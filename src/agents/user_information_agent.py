"""User Information 에이전트

사용자 프로필/자산 정보를 조회·입력·수정하는 전용 에이전트.

- MariaDB `user_info` 테이블을 사용
- 툴: `user_info_get`, `user_info_upsert`
"""

from __future__ import annotations

import json
import time
import re
from typing import Any, Dict, List, Optional
from decimal import Decimal
from datetime import datetime

from src.utils.agents import BaseAgent
from src.utils.logger import get_logger
from src.utils.llm import call_gpt
from src.tools.rdb import user_info_get, user_info_upsert, user_asset_log_get
from src.tools.date_parser import date_parse
from src.prompts.user_information_instruction import USER_INFORMATION_INSTRUCTION
from src.utils.state import Reference, Action, create_reference_and_action_from_tool_result


logger = get_logger(__name__)


class UserInformationAgent(BaseAgent):
    """사용자 프로필 및 자산 정보를 관리하는 에이전트."""

    def __init__(self) -> None:
        super().__init__(
            name="user_information",
            description=(
                "사용자 프로필(이름, 원화 자산, 달러 자산)을 "
                "MariaDB user_info 테이블에 저장/수정하고, 필요한 에이전트에게 제공하는 에이전트"
            ),
        )

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """사용자 정보 조회 또는 수정/등록을 수행한다."""
        execution_start_time = time.time()
        execution_id = f"user_info_{int(execution_start_time * 1000)}"
        
        logger.info(
            f"🚀 [EXECUTION START] 사용자 정보 에이전트 실행 시작",
            {
                "execution_id": execution_id,
                "agent_name": self.name,
                "task": task,
                "task_length": len(task),
                "has_context": context is not None,
                "context_keys": list(context.keys()) if context else [],
                "timestamp": execution_start_time
            }
        )
        
        context = context or {}
        state = context.get("state") or {}
        user_id: Optional[str] = (
            context.get("user_id")
            or state.get("user_id")
            or state.get("current_context", {}).get("user_id")
        )

        if not user_id:
            msg = "user_id가 제공되지 않아 사용자 정보를 조회/수정할 수 없습니다."
            logger.error(
                f"❌ [EXECUTION ERROR] UserInformationAgent.execute - user_id 누락",
                {
                    "execution_id": execution_id,
                    "task": task,
                    "context_keys": list(context.keys())
                },
            )
            return {
                "agent": self.name,
                "status": "error",
                "mode": None,
                "answer": msg,
                "user_info": None,
                "reasoning": "context/state에 user_id가 없어 작업을 수행할 수 없음",
                "reference": [],
                "action": []
            }

        # Reference와 Action 추적
        references: List[Reference] = []
        actions: List[Action] = []

        try:
            # 1) LLM에 어떤 작업을 할지 판단 요청 (조회 vs 수정/등록, 어떤 필드인지 등)
            analysis_start_time = time.time()
            logger.info(
                f"🔍 [ANALYSIS START] 요청 분석 단계 시작",
                {
                    "execution_id": execution_id,
                    "task": task,
                    "user_id": user_id
                }
            )
            
            plan = await self._analyze_request(task=task, user_id=user_id, context=context)
            
            analysis_elapsed = time.time() - analysis_start_time
            logger.info(
                f"✅ [ANALYSIS COMPLETE] 요청 분석 완료",
                {
                    "execution_id": execution_id,
                    "analysis_elapsed_seconds": round(analysis_elapsed, 3),
                    "mode": plan.get("mode"),
                    "reasoning": plan.get("reasoning", "")[:100]
                }
            )

            mode = plan.get("mode") or plan.get("action")  # "get" | "upsert" | "compare"
            reasoning = plan.get("reasoning", "")
            fields: Dict[str, Any] = plan.get("fields", {}) or {}

            if mode not in ("get", "upsert", "compare"):
                logger.warning(
                    f"⚠️ [MODE FALLBACK] mode가 명확하지 않아 기본값(get) 사용",
                    {
                        "execution_id": execution_id,
                        "mode": mode,
                        "plan": plan
                    },
                )
                mode = "get"

            # 2) 실제 DB 작업 수행
            db_operation_start_time = time.time()
            
            if mode == "get":
                logger.info(
                    f"📖 [DB OPERATION START] 사용자 정보 조회 시작",
                    {
                        "execution_id": execution_id,
                        "mode": "get",
                        "user_id": user_id
                    }
                )
                
                # LangChain @tool 로 래핑된 StructuredTool 이므로 .ainvoke(...) 로 호출
                result = await user_info_get.ainvoke({"user_id": user_id})
                db_operation_elapsed = time.time() - db_operation_start_time
                
                # Reference와 Action 생성
                reference, action = create_reference_and_action_from_tool_result(
                    tool_name="user_info_get",
                    tool_result=result,
                    source="rdb",
                    query=f"user_id={user_id}",
                    input_params={"user_id": user_id},
                    metadata={"mode": "get", "found": result.get("found", False)}
                )
                references.append(reference)
                actions.append(action)
                
                if result.get("found"):
                    user_info_data = result.get("user_info")
                    answer = self._format_user_info_answer(user_info_data, prefix="현재 등록된 사용자 정보입니다.")
                    logger.info(
                        f"✅ [DB OPERATION COMPLETE] 사용자 정보 조회 완료",
                        {
                            "execution_id": execution_id,
                            "elapsed_seconds": round(db_operation_elapsed, 3),
                            "found": True,
                            "has_user_info": user_info_data is not None
                        }
                    )
                else:
                    user_info_data = None
                    answer = (
                        "현재 이 user_id에 해당하는 사용자 정보가 등록되어 있지 않습니다. "
                        "필요하다면 이름, 원화 자산, 달러 자산을 알려주면 새로 등록할 수 있습니다."
                    )
                    logger.info(
                        f"ℹ️ [DB OPERATION COMPLETE] 사용자 정보 없음",
                        {
                            "execution_id": execution_id,
                            "elapsed_seconds": round(db_operation_elapsed, 3),
                            "found": False
                        }
                    )

                execution_elapsed = time.time() - execution_start_time
                logger.info(
                    f"🎉 [EXECUTION COMPLETE] 사용자 정보 에이전트 실행 완료",
                    {
                        "execution_id": execution_id,
                        "total_elapsed_seconds": round(execution_elapsed, 3),
                        "mode": "get",
                        "status": "success"
                    }
                )

                return {
                    "agent": self.name,
                    "status": "success",
                    "mode": "get",
                    "answer": answer,
                    "user_info": user_info_data,
                    "reasoning": reasoning,
                    "reference": [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in references],
                    "action": [act.__dict__ if hasattr(act, '__dict__') else act for act in actions]
                }

            # mode == "compare" (날짜 비교)
            if mode == "compare":
                compare_date = fields.get("compare_date")
                if not compare_date:
                    answer = "날짜 비교를 위해서는 비교할 날짜가 필요합니다. 예: '10/01기준 내 자산과 지금 내 자산을 비교하면 어때?'"
                    logger.warning(
                        f"⚠️ [COMPARE MODE] 비교 날짜가 제공되지 않음",
                        {
                            "execution_id": execution_id,
                            "fields": fields
                        }
                    )
                    return {
                        "agent": self.name,
                        "status": "error",
                        "mode": "compare",
                        "answer": answer,
                        "user_info": None,
                        "reasoning": reasoning,
                        "reference": [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in references],
                        "action": [act.__dict__ if hasattr(act, '__dict__') else act for act in actions]
                    }
                
                # 날짜 형식 변환 (다양한 형식 지원)
                try:
                    logger.info(
                        f"🔍 [COMPARE MODE] 날짜 파싱 시작",
                        {
                            "execution_id": execution_id,
                            "input_date_string": compare_date
                        }
                    )
                    parse_result = await date_parse.ainvoke({
                        "date_string": compare_date,
                        "default_year": 2025
                    })
                    if parse_result.get("status") == "success":
                        compare_date = parse_result.get("parsed_date")
                        logger.info(
                            f"✅ [COMPARE MODE] 날짜 파싱 성공",
                            {
                                "execution_id": execution_id,
                                "original": parse_result.get("original"),
                                "parsed_date": compare_date,
                                "parsed_date_type": type(compare_date).__name__
                            }
                        )
                        logger.info(
                            f"📅 [DEBUG] 파싱된 날짜: '{compare_date}' (원본: '{parse_result.get('original')}')"
                        )
                    else:
                        logger.warning(
                            f"⚠️ [COMPARE MODE] 날짜 파싱 실패",
                            {
                                "execution_id": execution_id,
                                "compare_date": compare_date,
                                "error": parse_result.get("error")
                            }
                        )
                        answer = f"날짜 형식을 인식할 수 없습니다: {compare_date}. 예: '10/02', '10월 2일', '2025-10-02'"
                        return {
                            "agent": self.name,
                            "status": "error",
                            "mode": "compare",
                            "answer": answer,
                            "user_info": None,
                            "reasoning": reasoning,
                            "reference": [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in references],
                            "action": [act.__dict__ if hasattr(act, '__dict__') else act for act in actions]
                        }
                except Exception as e:
                    logger.warning(
                        f"⚠️ [COMPARE MODE] 날짜 형식 변환 실패",
                        {
                            "execution_id": execution_id,
                            "compare_date": compare_date,
                            "error": str(e)
                        }
                    )
                    answer = f"날짜 파싱 중 오류가 발생했습니다: {str(e)}"
                    return {
                        "agent": self.name,
                        "status": "error",
                        "mode": "compare",
                        "answer": answer,
                        "user_info": None,
                        "reasoning": reasoning,
                        "reference": [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in references],
                        "action": [act.__dict__ if hasattr(act, '__dict__') else act for act in actions]
                    }
                
                logger.info(
                    f"📊 [COMPARE MODE] 자산 비교 시작",
                    {
                        "execution_id": execution_id,
                        "user_id": user_id,
                        "compare_date": compare_date,
                        "request_date": context.get("date")
                    }
                )
                
                # 특정 날짜의 자산 로그 조회 (과거 기준 날짜)
                past_log_result = await user_asset_log_get.ainvoke({"user_id": user_id, "date": compare_date})
                past_log = past_log_result.get("asset_log") if past_log_result.get("found") else None
                
                # 현재 기준 날짜 결정: 요청 body의 date가 있고 user_asset_log에 있으면 그 날짜 사용, 없으면 최신 행 사용
                request_date = context.get("date")
                current_date = None
                if request_date:
                    # 요청 body의 date가 user_asset_log에 있는지 확인
                    request_log_result = await user_asset_log_get.ainvoke({"user_id": user_id, "date": request_date})
                    if request_log_result.get("found"):
                        current_date = request_date
                        latest_log_result = request_log_result
                        latest_log = request_log_result.get("asset_log")
                        logger.info(
                            f"📅 [COMPARE MODE] 요청 body의 date 사용",
                            {
                                "execution_id": execution_id,
                                "request_date": request_date,
                                "current_date": current_date
                            }
                        )
                    else:
                        # 요청 body의 date가 테이블에 없으면 최신 행 사용
                        latest_log_result = await user_asset_log_get.ainvoke({"user_id": user_id, "date": None})
                        latest_log = latest_log_result.get("asset_log") if latest_log_result.get("found") else None
                        logger.info(
                            f"📅 [COMPARE MODE] 요청 body의 date가 테이블에 없어 최신 행 사용",
                            {
                                "execution_id": execution_id,
                                "request_date": request_date,
                                "current_date": "latest"
                            }
                        )
                else:
                    # 요청 body에 date가 없으면 최신 행 사용
                    latest_log_result = await user_asset_log_get.ainvoke({"user_id": user_id, "date": None})
                    latest_log = latest_log_result.get("asset_log") if latest_log_result.get("found") else None
                    logger.info(
                        f"📅 [COMPARE MODE] 요청 body에 date가 없어 최신 행 사용",
                        {
                            "execution_id": execution_id,
                            "current_date": "latest"
                        }
                    )
                
                # Reference와 Action 생성
                if past_log_result.get("found"):
                    reference, action = create_reference_and_action_from_tool_result(
                        tool_name="user_asset_log_get",
                        tool_result=past_log_result,
                        source="rdb",
                        query=f"user_id={user_id}, date={compare_date}",
                        input_params={"user_id": user_id, "date": compare_date},
                        metadata={"mode": "compare", "type": "past"}
                    )
                    references.append(reference)
                    actions.append(action)
                
                if latest_log_result.get("found"):
                    current_date_label = current_date if current_date else "latest"
                    reference, action = create_reference_and_action_from_tool_result(
                        tool_name="user_asset_log_get",
                        tool_result=latest_log_result,
                        source="rdb",
                        query=f"user_id={user_id}, date={current_date_label}",
                        input_params={"user_id": user_id, "date": current_date if current_date else None},
                        metadata={"mode": "compare", "type": "current", "date": current_date}
                    )
                    references.append(reference)
                    actions.append(action)
                
                if not past_log or not latest_log:
                    if not past_log:
                        answer = f"{compare_date} 기준의 자산 로그를 찾을 수 없습니다."
                    elif not latest_log:
                        answer = "최신 자산 로그를 찾을 수 없습니다."
                    else:
                        answer = "자산 로그를 찾을 수 없습니다."
                    
                    logger.warning(
                        f"⚠️ [COMPARE MODE] 자산 로그 없음",
                        {
                            "execution_id": execution_id,
                            "has_past_log": past_log is not None,
                            "has_latest_log": latest_log is not None
                        }
                    )
                else:
                    # 수익률 계산
                    past_total = (past_log.get("hedged_etf") or 0.0) + (past_log.get("unhedged_etf") or 0.0)
                    current_total = (latest_log.get("hedged_etf") or 0.0) + (latest_log.get("unhedged_etf") or 0.0)
                    
                    if past_total > 0:
                        return_rate = ((current_total - past_total) / past_total) * 100
                        current_date_label = current_date if current_date else "현재"
                        answer = f"{compare_date} 기준 자산(USD {past_total:,.2f})과 {current_date_label} 자산(USD {current_total:,.2f})을 비교한 결과, **{return_rate:+.2f}%**의 수익률을 기록했습니다."
                    else:
                        answer = f"{compare_date} 기준 자산이 0이어서 수익률을 계산할 수 없습니다."
                    
                    logger.info(
                        f"✅ [COMPARE MODE] 자산 비교 완료",
                        {
                            "execution_id": execution_id,
                            "compare_date": compare_date,
                            "current_date": current_date,
                            "past_total": past_total,
                            "current_total": current_total,
                            "return_rate": return_rate if past_total > 0 else None
                        }
                    )
                
                execution_elapsed = time.time() - execution_start_time
                logger.info(
                    f"🎉 [EXECUTION COMPLETE] 사용자 정보 에이전트 실행 완료 (비교 모드)",
                    {
                        "execution_id": execution_id,
                        "total_elapsed_seconds": round(execution_elapsed, 3),
                        "mode": "compare",
                        "status": "success"
                    }
                )
                
                return {
                    "agent": self.name,
                    "status": "success",
                    "mode": "compare",
                    "answer": answer,
                    "user_info": None,
                    "reasoning": reasoning,
                    "reference": [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in references],
                    "action": [act.__dict__ if hasattr(act, '__dict__') else act for act in actions]
                }

            # mode == "upsert"
            logger.info(
                f"💾 [DB OPERATION START] 사용자 정보 Upsert 시작",
                {
                    "execution_id": execution_id,
                    "mode": "upsert",
                    "user_id": user_id,
                    "fields_to_update": list(fields.keys())
                }
            )
            
            # 부족한 필드는 기존 DB 값으로 보완 (있다면)
            existing_start = time.time()
            # LangChain @tool 로 래핑된 StructuredTool 이므로 .ainvoke(...) 로 호출
            existing = await user_info_get.ainvoke({"user_id": user_id})
            existing_info: Dict[str, Any] = existing.get("user_info") or {}
            existing_elapsed = time.time() - existing_start
            
            logger.debug(
                f"📋 [EXISTING DATA] 기존 데이터 조회 완료",
                {
                    "execution_id": execution_id,
                    "elapsed_seconds": round(existing_elapsed, 3),
                    "has_existing": bool(existing_info)
                }
            )

            merged_fields: Dict[str, Any] = {
                "user_id": user_id,
                "user_name": fields.get("user_name", existing_info.get("user_name", "")),
                "user_krw": fields.get("user_krw", existing_info.get("user_krw", 0.0)),
                "user_usd": fields.get("user_usd", existing_info.get("user_usd", 0.0)),
            }
            
            # 헷지 비율 적용 요청 처리
            # 사용자가 "저대로 내 자산을 바꿔줘", "헷지 비율 적용해줘" 등의 요청을 했을 때
            apply_hedge_ratio = fields.get("apply_hedge_ratio", False)
            if apply_hedge_ratio:
                # context나 state에서 이전 strategy_execute_agent 결과 찾기
                w_H_star = None
                state = context.get("state") or {}
                
                # 1. context의 collected_data에서 찾기 (다양한 agent 이름 시도)
                collected_data = context.get("collected_data", {})
                logger.info(
                    f"🔍 [HEDGE RATIO SEARCH] collected_data 키 확인",
                    {
                        "execution_id": execution_id,
                        "collected_data_keys": list(collected_data.keys()) if collected_data else [],
                        "collected_data_types": {k: type(v).__name__ for k, v in collected_data.items()} if collected_data else {}
                    }
                )
                
                for agent_key in ["strategy_execute", "strategy_execute_agent", "hedge_strategy"]:
                    if agent_key in collected_data:
                        strategy_result = collected_data.get(agent_key, {})
                        logger.info(
                            f"🔍 [HEDGE RATIO SEARCH] {agent_key} 결과 확인",
                            {
                                "execution_id": execution_id,
                                "strategy_result_keys": list(strategy_result.keys()) if isinstance(strategy_result, dict) else [],
                                "strategy_result_type": type(strategy_result).__name__,
                                "w_H_star": strategy_result.get("w_H_star") if isinstance(strategy_result, dict) else None
                            }
                        )
                        w_H_star = strategy_result.get("w_H_star") if isinstance(strategy_result, dict) else None
                        if w_H_star is not None:
                            logger.info(
                                f"✅ [HEDGE RATIO FOUND] collected_data에서 헷지 비율 발견",
                                {
                                    "execution_id": execution_id,
                                    "source": agent_key,
                                    "w_H_star": w_H_star
                                }
                            )
                            break
                
                # 2. state의 conversation_history에서 찾기 (agent_results 포함)
                if w_H_star is None and state:
                    conversation_history = state.get("conversation_history", [])
                    logger.info(
                        f"🔍 [HEDGE RATIO SEARCH] conversation_history 검색",
                        {
                            "execution_id": execution_id,
                            "history_length": len(conversation_history) if conversation_history else 0
                        }
                    )
                    for turn in reversed(conversation_history):  # 최근부터 역순으로 검색
                        # 2-1. agent_results에서 직접 찾기
                        agent_results = turn.get("agent_results", [])
                        if isinstance(agent_results, list):
                            for agent_result in agent_results:
                                if isinstance(agent_result, dict):
                                    if agent_result.get("agent") == "strategy_execute":
                                        w_H_star = agent_result.get("w_H_star")
                                        if w_H_star is not None:
                                            logger.info(
                                                f"✅ [HEDGE RATIO FOUND] conversation_history.agent_results에서 헷지 비율 발견",
                                                {
                                                    "execution_id": execution_id,
                                                    "source": "conversation_history.agent_results",
                                                    "w_H_star": w_H_star
                                                }
                                            )
                                            break
                            if w_H_star is not None:
                                break
                        
                        # 2-2. additional_info의 metadata에서 찾기
                        additional_info = turn.get("additional_info", {})
                        if isinstance(additional_info, dict):
                            # metadata에서 strategy_execute 결과 찾기
                            metadata = additional_info.get("metadata", {})
                            if metadata and metadata.get("agent") == "strategy_execute":
                                w_H_star = metadata.get("w_H_star")
                                if w_H_star is not None:
                                    logger.info(
                                        f"✅ [HEDGE RATIO FOUND] conversation_history.metadata에서 헷지 비율 발견",
                                        {
                                            "execution_id": execution_id,
                                            "source": "conversation_history.metadata",
                                            "w_H_star": w_H_star
                                        }
                                    )
                                    break
                            
                            # 2-3. action에서 찾기 (strategy_execute의 output에서 w_H_star 찾기)
                            actions = additional_info.get("action", [])
                            if isinstance(actions, list):
                                for action in actions:
                                    if isinstance(action, dict):
                                        # action의 tool이 "strategy_execute"인 경우
                                        if action.get("tool") == "strategy_execute":
                                            action_output = action.get("output", {})
                                            if isinstance(action_output, dict):
                                                w_H_star = action_output.get("w_H_star")
                                                if w_H_star is not None:
                                                    logger.info(
                                                        f"✅ [HEDGE RATIO FOUND] conversation_history.action.output에서 헷지 비율 발견",
                                                        {
                                                            "execution_id": execution_id,
                                                            "source": "conversation_history.action.output",
                                                            "w_H_star": w_H_star
                                                        }
                                                    )
                                                    break
                                        # metadata에서도 찾기 (기존 로직 유지)
                                        action_metadata = action.get("metadata", {})
                                        if action_metadata and action_metadata.get("w_H_star"):
                                            w_H_star = action_metadata.get("w_H_star")
                                            if w_H_star is not None:
                                                logger.info(
                                                    f"✅ [HEDGE RATIO FOUND] conversation_history.action.metadata에서 헷지 비율 발견",
                                                    {
                                                        "execution_id": execution_id,
                                                        "source": "conversation_history.action.metadata",
                                                        "w_H_star": w_H_star
                                                    }
                                                )
                                                break
                                    if w_H_star is not None:
                                        break
                            if w_H_star is not None:
                                break
                
                # 3. context에서 직접 찾기
                if w_H_star is None:
                    w_H_star = context.get("w_H_star") or context.get("hedge_ratio") or context.get("w_H*")
                    if w_H_star is not None:
                        logger.info(
                            f"✅ [HEDGE RATIO FOUND] context에서 직접 헷지 비율 발견",
                            {
                                "execution_id": execution_id,
                                "source": "context",
                                "w_H_star": w_H_star
                            }
                        )
                
                if w_H_star is not None and 0.0 <= w_H_star <= 1.0:
                    # user_usd 가져오기 (기존 값 사용)
                    user_usd = merged_fields.get("user_usd", 0.0)
                    if user_usd > 0:
                        # Decimal로 변환하여 계산 (정밀도 보장)
                        user_usd_decimal = Decimal(str(user_usd))
                        w_H_star_decimal = Decimal(str(w_H_star))
                        
                        # hedged_etf, unhedged_etf 계산
                        hedged_etf = user_usd_decimal * w_H_star_decimal
                        unhedged_etf = user_usd_decimal * (Decimal("1") - w_H_star_decimal)
                        
                        # 소수점 2자리로 반올림 (DECIMAL(15, 2)에 맞춤)
                        hedged_etf = hedged_etf.quantize(Decimal("0.01"))
                        unhedged_etf = unhedged_etf.quantize(Decimal("0.01"))
                        
                        merged_fields["hedged_etf"] = hedged_etf
                        merged_fields["unhedged_etf"] = unhedged_etf
                        
                        logger.info(
                            f"✅ [HEDGE RATIO APPLY] 헷지 비율 적용",
                            {
                                "execution_id": execution_id,
                                "w_H_star": w_H_star,
                                "user_usd": user_usd,
                                "hedged_etf": hedged_etf,
                                "unhedged_etf": unhedged_etf
                            }
                        )
                    else:
                        logger.warning(
                            f"⚠️ [HEDGE RATIO APPLY] user_usd가 0이어서 헷지 비율을 적용할 수 없음",
                            {
                                "execution_id": execution_id,
                                "w_H_star": w_H_star,
                                "user_usd": user_usd
                            }
                        )
                else:
                    logger.warning(
                        f"⚠️ [HEDGE RATIO APPLY] 유효한 헷지 비율을 찾을 수 없음",
                        {
                            "execution_id": execution_id,
                            "w_H_star": w_H_star,
                            "context_keys": list(context.keys()) if context else []
                        }
                    )
            else:
                # 명시적으로 hedged_etf, unhedged_etf가 제공된 경우
                if "hedged_etf" in fields:
                    merged_fields["hedged_etf"] = fields["hedged_etf"]
                if "unhedged_etf" in fields:
                    merged_fields["unhedged_etf"] = fields["unhedged_etf"]

            # context에서 date 가져오기 (요청 body의 date 필드)
            request_date = context.get("date")
            if request_date:
                merged_fields["date"] = request_date
                logger.info(
                    f"📅 [DATE UPDATE] user_info의 date 업데이트",
                    {
                        "execution_id": execution_id,
                        "date": request_date,
                        "user_id": user_id
                    }
                )

            upsert_start = time.time()
            # LangChain @tool 로 래핑된 StructuredTool 이므로 .ainvoke(...) 로 호출
            upsert_result = await user_info_upsert.ainvoke(merged_fields)
            upsert_elapsed = time.time() - upsert_start
            db_operation_elapsed = time.time() - db_operation_start_time
            
            saved_info = upsert_result.get("user_info")
            operation = upsert_result.get("operation")

            # Reference와 Action 생성
            reference, action = create_reference_and_action_from_tool_result(
                tool_name="user_info_upsert",
                tool_result=upsert_result,
                source="rdb",
                query=f"user_id={user_id}",
                input_params=merged_fields,
                metadata={"mode": "upsert", "operation": operation}
            )
            references.append(reference)
            actions.append(action)

            if operation == "insert":
                prefix = "새로운 사용자 정보를 등록했습니다."
            else:
                prefix = "기존 사용자 정보를 업데이트했습니다."

            answer = self._format_user_info_answer(saved_info, prefix=prefix)
            
            logger.info(
                f"✅ [DB OPERATION COMPLETE] 사용자 정보 Upsert 완료",
                {
                    "execution_id": execution_id,
                    "elapsed_seconds": round(db_operation_elapsed, 3),
                    "upsert_elapsed_seconds": round(upsert_elapsed, 3),
                    "operation": operation,
                    "has_saved_info": saved_info is not None
                }
            )

            execution_elapsed = time.time() - execution_start_time
            logger.info(
                f"🎉 [EXECUTION COMPLETE] 사용자 정보 에이전트 실행 완료",
                {
                    "execution_id": execution_id,
                    "total_elapsed_seconds": round(execution_elapsed, 3),
                    "mode": "upsert",
                    "status": "success"
                }
            )

            return {
                "agent": self.name,
                "status": "success",
                "mode": "upsert",
                "answer": answer,
                "user_info": saved_info,
                "reasoning": reasoning,
                "reference": [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in references],
                "action": [act.__dict__ if hasattr(act, '__dict__') else act for act in actions]
            }

        except Exception as e:
            execution_elapsed = time.time() - execution_start_time
            logger.error(
                f"❌ [EXECUTION ERROR] UserInformationAgent.execute 실패",
                {
                    "execution_id": execution_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "total_elapsed_seconds": round(execution_elapsed, 3),
                    "task": task
                },
                exc_info=True,
            )
            return {
                "agent": self.name,
                "status": "error",
                "mode": None,
                "answer": f"사용자 정보 처리 중 오류가 발생했습니다: {e}",
                "user_info": None,
                "reasoning": "",
                "reference": [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in references],
                "action": [act.__dict__ if hasattr(act, '__dict__') else act for act in actions]
            }

    async def _analyze_request(
        self,
        task: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """LLM으로부터 '조회' vs '수정/등록' 및 필드 추출 계획을 받아온다."""
        context = context or {}

        system_prompt = f"""{USER_INFORMATION_INSTRUCTION}

## 응답 형식
반드시 다음 JSON 형식으로만 응답하세요 (설명 텍스트 금지):
{{
  "mode": "get" | "upsert" | "compare",
  "fields": {{
    "user_name": "<사용자 이름 또는 null>",
    "user_krw": <한국 원화 자산 또는 null>,
    "user_usd": <미국 달러 자산 또는 null>,
    "hedged_etf": <헷지된 ETF 금액 (USD) 또는 null>,
    "unhedged_etf": <비헷지 ETF 금액 (USD) 또는 null>,
    "apply_hedge_ratio": <true/false>,  # "저대로 내 자산을 바꿔줘", "헷지 비율 적용해줘" 등의 요청일 때 true
    "compare_date": "<YYYY-MM-DD 형식의 날짜 (예: '2025-10-02', '2023-10-02') 또는 null>"  # "10/01기준 내 자산과 지금 내 자산을 비교" 등의 요청일 때 날짜 추출. 연도가 없으면 기본 연도는 2025년입니다.
  }},
  "reasoning": "<당신의 판단 근거>"
}}

필드 값이 요청에서 전혀 언급되지 않았다면 null 또는 생략해도 됩니다.
"저대로 내 자산을 바꿔줘", "헷지 비율 적용해줘", "계산된 비율로 자산을 업데이트해줘" 등의 요청은 apply_hedge_ratio를 true로 설정하세요.
"10/01기준 내 자산과 지금 내 자산을 비교하면 어때?", "2025-10-01과 현재 자산 비교" 등의 요청은 mode를 "compare"로 설정하고 compare_date에 날짜를 설정하세요.

**중요: compare_date 필드 추출 규칙**
- compare_date는 반드시 YYYY-MM-DD 형식으로 추출해야 합니다 (예: "2025-10-02", "2023-10-02")
- 사용자가 "10/02", "10월 2일", "10.02" 등으로 입력했다면 → 연도가 없으므로 기본 연도인 **2025년**을 사용하여 "2025-10-02"로 변환
- 사용자가 "2023-10-02"처럼 연도를 포함해서 입력했다면 → 그대로 "2023-10-02"로 추출 (2023년 데이터를 찾음)
- 사용자가 "2023/10/02"처럼 다른 형식으로 연도를 포함했다면 → "2023-10-02"로 변환
- 기본 연도는 **2025년**입니다. 연도가 명시되지 않은 경우 항상 2025년을 사용하세요.
"""

        user_context_str = ""
        if context:
            try:
                user_context_str = json.dumps(
                    {"user_id": user_id, "context_keys": list(context.keys())},
                    ensure_ascii=False,
                )
            except Exception:
                user_context_str = f'{{"user_id": "{user_id}"}}'

        user_prompt = f"""사용자(또는 다른 에이전트)의 요청:
{task}

현재 user_id: {user_id}
간단한 컨텍스트 정보: {user_context_str}

이 요청이 사용자 정보 "조회"인지 "수정/등록(upsert)"인지 판단하고,
필요하다면 어떤 필드(user_name, user_krw, user_usd)를
어떤 값으로 저장/수정해야 하는지 위 응답 형식에 맞추어 JSON으로만 반환하세요."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_call_start = time.time()
        logger.debug(
            f"📤 [LLM CALL] LLM 호출 시작 (요청 분석)",
            {
                "task": task,
                "user_id": user_id,
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
                "temperature": 0.2
            }
        )

        response = await call_gpt(
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        llm_call_elapsed = time.time() - llm_call_start
        logger.debug(
            f"📥 [LLM RESPONSE] LLM 응답 수신",
            {
                "llm_call_elapsed_seconds": round(llm_call_elapsed, 3),
                "response_length": len(response),
                "response_preview": response[:200]
            }
        )

        try:
            plan = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(
                f"⚠️ [JSON PARSE ERROR] JSON 파싱 실패, 기본값 사용",
                {
                    "response_preview": response[:200],
                    "response_length": len(response)
                },
            )
            plan = {
                "mode": "get",
                "fields": {},
                "reasoning": "LLM 응답 파싱 실패로 기본 조회 모드를 사용함",
            }

        if "fields" not in plan or not isinstance(plan["fields"], dict):
            plan["fields"] = {}

        logger.debug(
            f"✅ [ANALYSIS RESULT] 분석 결과",
            {
                "mode": plan.get("mode"),
                "fields_count": len(plan.get("fields", {})),
                "fields_keys": list(plan.get("fields", {}).keys()),
                "reasoning_length": len(plan.get("reasoning", ""))
            }
        )

        return plan

    def _parse_date_string(self, date_str: str, context_date: Optional[str] = None) -> str:
        """
        다양한 날짜 형식을 YYYY-MM-DD 형식으로 변환
        
        지원 형식:
        - "10/02" → "2025-10-02"
        - "10월 2일" → "2025-10-02"
        - "10월 02일" → "2025-10-02"
        - "2025-10-02" → "2025-10-02" (그대로 반환)
        - "2025/10/02" → "2025-10-02"
        
        Args:
            date_str: 파싱할 날짜 문자열
            context_date: 컨텍스트 날짜 (연도 추출용, 예: "2025-01-01")
        
        Returns:
            YYYY-MM-DD 형식의 날짜 문자열
        
        Raises:
            ValueError: 날짜를 파싱할 수 없는 경우
        """
        if not date_str:
            raise ValueError("날짜 문자열이 비어있습니다.")
        
        date_str = date_str.strip()
        
        # 이미 YYYY-MM-DD 형식인 경우
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # YYYY/MM/DD 형식
        if re.match(r'^\d{4}/\d{1,2}/\d{1,2}$', date_str):
            parts = date_str.split("/")
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        
        # MM/DD 형식 (연도는 2025로 고정)
        if re.match(r'^\d{1,2}/\d{1,2}$', date_str):
            parts = date_str.split("/")
            return f"2025-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
        
        # "10월 2일", "10월 02일" 형식
        month_day_match = re.match(r'^(\d{1,2})월\s*(\d{1,2})일', date_str)
        if month_day_match:
            month = month_day_match.group(1).zfill(2)
            day = month_day_match.group(2).zfill(2)
            return f"2025-{month}-{day}"
        
        # "10월2일" (공백 없음)
        month_day_match = re.match(r'^(\d{1,2})월(\d{1,2})일', date_str)
        if month_day_match:
            month = month_day_match.group(1).zfill(2)
            day = month_day_match.group(2).zfill(2)
            return f"2025-{month}-{day}"
        
        # "10.02" 형식
        if re.match(r'^\d{1,2}\.\d{1,2}$', date_str):
            parts = date_str.split(".")
            return f"2025-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
        
        # 마지막 시도: datetime 파싱
        try:
            # 다양한 형식 시도
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m/%d", "%Y.%m.%d", "%m.%d"]:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    if fmt in ["%m/%d", "%m.%d"]:
                        # 연도가 없으면 2025로 설정
                        return f"2025-{parsed.month:02d}-{parsed.day:02d}"
                    else:
                        return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        except Exception:
            pass
        
        raise ValueError(f"날짜 형식을 인식할 수 없습니다: {date_str}")

    def _format_user_info_answer(
        self,
        user_info: Optional[Dict[str, Any]],
        prefix: str = "",
    ) -> str:
        """user_info 딕셔너리를 사람이 읽기 좋은 한국어 문장으로 변환."""
        if not user_info:
            return prefix or "사용자 정보가 없습니다."

        user_name = user_info.get("user_name") or "이름 미등록"
        user_krw = user_info.get("user_krw")
        user_usd = user_info.get("user_usd")
        risk_level = user_info.get("risk_level")
        hedged_etf = user_info.get("hedged_etf")
        unhedged_etf = user_info.get("unhedged_etf")

        def fmt_krw(v: Any) -> str:
            try:
                return f"{float(v):,.0f}원"
            except Exception:
                return "정보 없음"

        def fmt_usd(v: Any) -> str:
            try:
                return f"${float(v):,.2f}"
            except Exception:
                return "정보 없음"

        def fmt_risk(v: Any) -> str:
            try:
                return f"{float(v):.2f}"
            except Exception:
                return "정보 없음"

        parts = []
        if prefix:
            parts.append(prefix)

        parts.append(f"이름: {user_name}")
        if user_krw is not None:
            parts.append(f"원화 자산: {fmt_krw(user_krw)}")
        if user_usd is not None:
            parts.append(f"달러 자산: {fmt_usd(user_usd)}")
        if risk_level is not None:
            parts.append(f"리스크 레벨: {fmt_risk(risk_level)}")
        if hedged_etf is not None:
            parts.append(f"헷지된 ETF: {fmt_usd(hedged_etf)}")
        if unhedged_etf is not None:
            parts.append(f"비헷지 ETF: {fmt_usd(unhedged_etf)}")

        return " / ".join(parts)


