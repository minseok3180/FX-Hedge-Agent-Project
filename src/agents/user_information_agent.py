"""User Information 에이전트

사용자 프로필/자산 정보를 조회·입력·수정하는 전용 에이전트.

- MariaDB `user_info` 테이블을 사용
- 툴: `user_info_get`, `user_info_upsert`
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from src.utils.agents import BaseAgent
from src.utils.logger import get_logger
from src.utils.llm import call_gpt
from src.tools.rdb import user_info_get, user_info_upsert
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

            mode = plan.get("mode") or plan.get("action")  # "get" | "upsert"
            reasoning = plan.get("reasoning", "")
            fields: Dict[str, Any] = plan.get("fields", {}) or {}

            if mode not in ("get", "upsert"):
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
                    "reference": [ref.__dict__ for ref in references],
                    "action": [act.__dict__ for act in actions]
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
                "reference": [ref.__dict__ for ref in references],
                "action": [act.__dict__ for act in actions]
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
                "reference": [ref.__dict__ for ref in references],
                "action": [act.__dict__ for act in actions]
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
  "mode": "get" | "upsert",
  "fields": {{
    "user_name": "<사용자 이름 또는 null>",
    "user_krw": <한국 원화 자산 또는 null>,
    "user_usd": <미국 달러 자산 또는 null>
  }},
  "reasoning": "<당신의 판단 근거>"
}}

필드 값이 요청에서 전혀 언급되지 않았다면 null 또는 생략해도 됩니다.
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

        return " / ".join(parts)


