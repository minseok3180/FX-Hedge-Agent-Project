"""시장 정보 에이전트"""
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from src.utils.agents import BaseAgent
from src.tools.rdb import rdb_query_hard, rdb_query_llm, rdb_get_latest_ecos_date
from src.tools.web_search import web_search
from src.prompts.market_information_instruction import MARKET_INFORMATION_INSTRUCTION
from src.utils.llm import call_gpt
from src.utils.state import Reference, Action, create_reference_and_action_from_tool_result


class MarketInformationAgent(BaseAgent):
    """시장 정보를 수집하고 분석하는 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="market_information",
            description="시장 정보를 수집하고 분석하여 헤지 전략 수립에 필요한 인사이트를 제공하는 에이전트"
        )
        
        # 툴은 함수형으로 사용 (초기화 불필요)
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        시장 정보 수집 및 분석 수행
        
        Args:
            task: 수행할 작업 설명
            context: 추가 컨텍스트 정보 (state 포함)
            
        Returns:
            실행 결과 딕셔너리
        """
        execution_start_time = time.time()
        execution_id = f"market_info_{int(execution_start_time * 1000)}"
        
        self.logger.info(
            f"🚀 [EXECUTION START] 시장 정보 에이전트 실행 시작",
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
        
        try:
            # Context에서 state, collected_data 추출
            state = context.get("state") if context else None
            previous_collected_data = context.get("collected_data", {}) if context else {}
            user_query = task
            
            self.logger.debug(
                f"📋 [CONTEXT EXTRACTION] 컨텍스트 추출 완료",
                {
                    "execution_id": execution_id,
                    "has_state": state is not None,
                    "state_keys": list(state.keys()) if state else [],
                    "state_date": state.get("date") if state else None,
                    "state_user_id": state.get("user_id") if state else None,
                    "has_previous_data": bool(previous_collected_data),
                    "previous_data_keys": list(previous_collected_data.keys()) if previous_collected_data else [],
                    "previous_data_summary": {
                        k: f"{type(v).__name__}({len(v) if isinstance(v, (list, dict)) else 'N/A'})"
                        for k, v in previous_collected_data.items()
                    } if previous_collected_data else {}
                }
            )
            
            # Reference와 Action 추적
            references: List[Reference] = []
            actions: List[Action] = []
            
            # 1. LLM을 통해 필요한 데이터 파악 및 쿼리 결정
            # 이전 에이전트의 collected_data를 고려하여 분석
            analysis_start_time = time.time()
            self.logger.info(
                f"🔍 [ANALYSIS START] 질문 분석 단계 시작",
                {
                    "execution_id": execution_id,
                    "user_query": user_query,
                    "query_length": len(user_query),
                    "has_state": state is not None,
                    "has_previous_data": bool(previous_collected_data)
                }
            )
            
            analysis_result = await self._analyze_query(user_query, state, previous_collected_data)
            
            analysis_elapsed = time.time() - analysis_start_time
            self.logger.info(
                f"✅ [ANALYSIS COMPLETE] 질문 분석 완료",
                {
                    "execution_id": execution_id,
                    "analysis_elapsed_seconds": round(analysis_elapsed, 3),
                    "needs_rdb_query": analysis_result.get("needs_rdb_query", False),
                    "needs_web_search": analysis_result.get("needs_web_search", False),
                    "query_type": analysis_result.get("query_type"),
                    "query_params": analysis_result.get("query_params", {}),
                    "web_search_query": analysis_result.get("web_search_query"),
                    "web_search_num_results": analysis_result.get("web_search_num_results"),
                    "reasoning": analysis_result.get("reasoning", ""),
                    "full_analysis_result": analysis_result
                }
            )
            
            # 2. 데이터 수집 (이전 에이전트의 데이터와 병합)
            data_collection_start_time = time.time()
            collected_data = previous_collected_data.copy()
            
            self.logger.info(
                f"📦 [DATA COLLECTION START] 데이터 수집 단계 시작",
                {
                    "execution_id": execution_id,
                    "initial_data_keys": list(collected_data.keys()),
                    "initial_data_summary": {
                        k: f"{type(v).__name__}({len(v) if isinstance(v, (list, dict)) else 'N/A'})"
                        for k, v in collected_data.items()
                    } if collected_data else {}
                }
            )
            
            # 2-1. 웹 검색이 필요한 경우
            if analysis_result.get("needs_web_search"):
                search_query = analysis_result.get("web_search_query", user_query)
                num_results = analysis_result.get("web_search_num_results", 5)
                
                web_search_start_time = time.time()
                self.logger.info(
                    f"🌐 [WEB SEARCH START] 웹 검색 시작",
                    {
                        "execution_id": execution_id,
                        "tool_name": "web_search",
                        "search_query": search_query,
                        "query_length": len(search_query),
                        "num_results": num_results,
                        "agent": "market_information",
                        "triggered_by": "analysis_result.needs_web_search"
                    }
                )
                
                try:
                    # @tool 데코레이터로 래핑된 함수는 .ainvoke()로 호출
                    web_results = await web_search.ainvoke({
                        "query": search_query,
                        "num_results": num_results
                    })
                    web_search_elapsed = time.time() - web_search_start_time
                    
                    self.logger.info(
                        f"✅ [WEB SEARCH COMPLETE] 웹 검색 완료",
                        {
                            "execution_id": execution_id,
                            "tool_name": "web_search",
                            "search_query": search_query,
                            "elapsed_seconds": round(web_search_elapsed, 3),
                            "results_count": len(web_results) if web_results else 0,
                            "results_preview": [
                                {
                                    "title": r.get("title", "")[:50],
                                    "snippet_length": len(r.get("snippet", "")),
                                    "has_link": bool(r.get("link"))
                                }
                                for r in web_results[:3]
                            ] if web_results else [],
                            "full_results_count": len(web_results) if web_results else 0
                        }
                    )
                    
                    # Reference와 Action 생성
                    reference_creation_start = time.time()
                    reference, action = create_reference_and_action_from_tool_result(
                        tool_name="web_search",
                        tool_result=web_results,
                        source="web_search",
                        query=search_query,
                        input_params={"query": search_query, "num_results": num_results},
                        metadata={"num_results": num_results}
                    )
                    references.append(reference)
                    actions.append(action)
                    collected_data["web_search"] = web_results
                    
                    self.logger.debug(
                        f"📝 [REFERENCE CREATED] Reference 및 Action 생성 완료",
                        {
                            "execution_id": execution_id,
                            "reference_creation_elapsed": round(time.time() - reference_creation_start, 4),
                            "reference_source": reference.source if hasattr(reference, 'source') else None,
                            "reference_query": reference.query if hasattr(reference, 'query') else None,
                            "action_type": action.action_type if hasattr(action, 'action_type') else None,
                            "total_references": len(references),
                            "total_actions": len(actions)
                        }
                    )
                    
                except Exception as e:
                    web_search_elapsed = time.time() - web_search_start_time
                    self.logger.error(
                        f"❌ [WEB SEARCH ERROR] 웹 검색 실행 실패",
                        {
                            "execution_id": execution_id,
                            "tool_name": "web_search",
                            "search_query": search_query,
                            "elapsed_seconds": round(web_search_elapsed, 3),
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "error_traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
                        },
                        exc_info=True
                    )
                    collected_data["web_search_error"] = str(e)
                    collected_data["web_search_error_type"] = type(e).__name__
            else:
                self.logger.debug(
                    f"⏭️  [WEB SEARCH SKIPPED] 웹 검색 건너뜀",
                    {
                        "execution_id": execution_id,
                        "reason": "analysis_result.needs_web_search is False"
                    }
                )
            
            # 2-2. RDB 쿼리가 필요한 경우
            if analysis_result.get("needs_rdb_query"):
                query_type = analysis_result.get("query_type")
                query_params = analysis_result.get("query_params", {})
                
                rdb_query_start_time = time.time()
                self.logger.info(
                    f"🗄️  [RDB QUERY START] RDB 쿼리 시작",
                    {
                        "execution_id": execution_id,
                        "query_type": query_type,
                        "query_params": query_params,
                        "query_params_keys": list(query_params.keys()) if query_params else [],
                        "agent": "market_information",
                        "triggered_by": "analysis_result.needs_rdb_query"
                    }
                )
                
                try:
                    if query_type == "get_by_date":
                        # get_by_date는 {date} placeholder를 사용
                        # query_params에 date가 있으면 사용, 없으면 state에서 가져옴
                        date_value = query_params.get("date") or (state.get("date") if state else None)
                        
                        self.logger.debug(
                            f"📅 [DATE RESOLUTION] 날짜 값 확인",
                            {
                                "execution_id": execution_id,
                                "query_type": "get_by_date",
                                "date_from_params": query_params.get("date"),
                                "date_from_state": state.get("date") if state else None,
                                "resolved_date": date_value,
                                "has_date": bool(date_value)
                            }
                        )
                        
                        if not date_value:
                            # date가 없으면 get_latest로 대체
                            self.logger.warning(
                                f"⚠️  [FALLBACK] get_by_date에 date가 없어 get_latest로 대체",
                                {
                                    "execution_id": execution_id,
                                    "original_query_type": "get_by_date",
                                    "fallback_to": "get_latest",
                                    "fallback_limit": 10,
                                    "reason": "date_value is None"
                                }
                            )
                            
                            get_latest_start = time.time()
                            # @tool 데코레이터로 래핑된 함수는 .ainvoke()로 호출
                            results = await rdb_query_hard.ainvoke({
                                "query_key": "get_latest",
                                "params": (10,),
                                "state": state
                            })
                            get_latest_elapsed = time.time() - get_latest_start
                            
                            self.logger.info(
                                f"✅ [FALLBACK COMPLETE] get_latest 실행 완료",
                                {
                                    "execution_id": execution_id,
                                    "elapsed_seconds": round(get_latest_elapsed, 3),
                                    "results_count": len(results) if results else 0
                                }
                            )
                            
                            # Reference와 Action 생성
                            reference_creation_start = time.time()
                            reference, action = create_reference_and_action_from_tool_result(
                                tool_name="rdb_hard",
                                tool_result=results,
                                source="rdb",
                                query="get_latest",
                                input_params={"query_type": "get_latest", "query_params": {"limit": 10}},
                                metadata={"limit": 10, "fallback_from": "get_by_date"}
                            )
                            references.append(reference)
                            actions.append(action)
                            collected_data["exchange_rate"] = results
                            
                            self.logger.debug(
                                f"📝 [REFERENCE CREATED] Reference 및 Action 생성 완료 (fallback)",
                                {
                                    "execution_id": execution_id,
                                    "reference_creation_elapsed": round(time.time() - reference_creation_start, 4),
                                    "total_references": len(references),
                                    "total_actions": len(actions),
                                    "collected_data_keys": list(collected_data.keys())
                                }
                            )
                        else:
                            # state에 date를 설정하여 placeholder 치환
                            if state:
                                state_with_date = state.copy()
                                state_with_date["date"] = date_value
                            else:
                                state_with_date = {"date": date_value}
                            
                            self.logger.debug(
                                f"🔧 [STATE PREPARATION] State 준비 완료",
                                {
                                    "execution_id": execution_id,
                                    "original_state_keys": list(state.keys()) if state else [],
                                    "state_with_date_keys": list(state_with_date.keys()),
                                    "date_value": date_value
                                }
                            )
                            
                            get_by_date_start = time.time()
                            # @tool 데코레이터로 래핑된 함수는 .ainvoke()로 호출
                            results = await rdb_query_hard.ainvoke({
                                "query_key": "get_by_date",
                                "state": state_with_date
                            })
                            get_by_date_elapsed = time.time() - get_by_date_start
                            
                            self.logger.info(
                                f"✅ [RDB QUERY COMPLETE] get_by_date 실행 완료",
                                {
                                    "execution_id": execution_id,
                                    "query_type": "get_by_date",
                                    "date": date_value,
                                    "elapsed_seconds": round(get_by_date_elapsed, 3),
                                    "results_count": len(results) if results else 0,
                                    "results_preview": [
                                        {
                                            "date": r.get("date"),
                                            "usdkrw": r.get("usdkrw"),
                                            "keys": list(r.keys())[:5]
                                        }
                                        for r in results[:2]
                                    ] if results else []
                                }
                            )
                        
                        # Reference와 Action 생성
                        reference_creation_start = time.time()
                        reference, action = create_reference_and_action_from_tool_result(
                            tool_name="rdb_hard",
                            tool_result=results,
                            source="rdb",
                            query="get_by_date",
                            input_params={"query_type": "get_by_date", "query_params": query_params},
                            metadata={"date": date_value}
                        )
                        references.append(reference)
                        actions.append(action)
                        collected_data["exchange_rate"] = results
                        
                        self.logger.debug(
                            f"📝 [REFERENCE CREATED] Reference 및 Action 생성 완료",
                            {
                                "execution_id": execution_id,
                                "reference_creation_elapsed": round(time.time() - reference_creation_start, 4),
                                "total_references": len(references),
                                "total_actions": len(actions),
                                "collected_data_keys": list(collected_data.keys())
                            }
                        )
                        
                    elif query_type == "get_by_range":
                        start_date = query_params.get("start_date")
                        end_date = query_params.get("end_date")
                        limit = query_params.get("limit", 100)
                        
                        self.logger.debug(
                            f"📊 [RANGE QUERY] 범위 쿼리 파라미터 확인",
                            {
                                "execution_id": execution_id,
                                "query_type": "get_by_range",
                                "start_date": start_date,
                                "end_date": end_date,
                                "limit": limit,
                                "date_range_days": (
                                    (end_date - start_date).days 
                                    if start_date and end_date and hasattr(end_date, '__sub__') 
                                    else None
                                )
                            }
                        )
                        
                        get_by_range_start = time.time()
                        # @tool 데코레이터로 래핑된 함수는 .ainvoke()로 호출
                        results = await rdb_query_hard.ainvoke({
                            "query_key": "get_by_range",
                            "params": (start_date, end_date, limit),
                            "state": state
                        })
                        get_by_range_elapsed = time.time() - get_by_range_start
                        
                        self.logger.info(
                            f"✅ [RDB QUERY COMPLETE] get_by_range 실행 완료",
                            {
                                "execution_id": execution_id,
                                "query_type": "get_by_range",
                                "start_date": start_date,
                                "end_date": end_date,
                                "limit": limit,
                                "elapsed_seconds": round(get_by_range_elapsed, 3),
                                "results_count": len(results) if results else 0,
                                "results_date_range": {
                                    "first_date": results[0].get("date") if results else None,
                                    "last_date": results[-1].get("date") if results else None
                                } if results else {}
                            }
                        )
                        
                        # Reference와 Action 생성
                        reference_creation_start = time.time()
                        reference, action = create_reference_and_action_from_tool_result(
                            tool_name="rdb_hard",
                            tool_result=results,
                            source="rdb",
                            query="get_by_range",
                            input_params={"query_type": "get_by_range", "query_params": query_params},
                            metadata={"start_date": start_date, "end_date": end_date, "limit": limit}
                        )
                        references.append(reference)
                        actions.append(action)
                        collected_data["exchange_rate"] = results
                        
                        self.logger.debug(
                            f"📝 [REFERENCE CREATED] Reference 및 Action 생성 완료",
                            {
                                "execution_id": execution_id,
                                "reference_creation_elapsed": round(time.time() - reference_creation_start, 4),
                                "total_references": len(references),
                                "total_actions": len(actions)
                            }
                        )
                        
                    elif query_type == "get_latest":
                        limit = query_params.get("limit", 10)
                        
                        self.logger.debug(
                            f"🔄 [LATEST QUERY] 최신 데이터 쿼리 파라미터 확인",
                            {
                                "execution_id": execution_id,
                                "query_type": "get_latest",
                                "limit": limit
                            }
                        )
                        
                        get_latest_start = time.time()
                        # @tool 데코레이터로 래핑된 함수는 .ainvoke()로 호출
                        results = await rdb_query_hard.ainvoke({
                            "query_key": "get_latest",
                            "params": (limit,),
                            "state": state
                        })
                        get_latest_elapsed = time.time() - get_latest_start
                        
                        self.logger.info(
                            f"✅ [RDB QUERY COMPLETE] get_latest 실행 완료",
                            {
                                "execution_id": execution_id,
                                "query_type": "get_latest",
                                "limit": limit,
                                "elapsed_seconds": round(get_latest_elapsed, 3),
                                "results_count": len(results) if results else 0,
                                "latest_date": results[0].get("date") if results else None
                            }
                        )
                        
                        # Reference와 Action 생성
                        reference_creation_start = time.time()
                        reference, action = create_reference_and_action_from_tool_result(
                            tool_name="rdb_hard",
                            tool_result=results,
                            source="rdb",
                            query="get_latest",
                            input_params={"query_type": "get_latest", "query_params": query_params},
                            metadata={"limit": limit}
                        )
                        references.append(reference)
                        actions.append(action)
                        collected_data["exchange_rate"] = results
                        
                        self.logger.debug(
                            f"📝 [REFERENCE CREATED] Reference 및 Action 생성 완료",
                            {
                                "execution_id": execution_id,
                                "reference_creation_elapsed": round(time.time() - reference_creation_start, 4),
                                "total_references": len(references),
                                "total_actions": len(actions)
                            }
                        )
                        
                    elif query_type == "rdb_llm":
                        # LLM이 쿼리를 생성하여 실행
                        rdb_llm_start_time = time.time()
                        self.logger.info(
                            f"🤖 [RDB LLM START] LLM 기반 쿼리 생성 시작",
                            {
                                "execution_id": execution_id,
                                "tool_name": "rdb_query_llm",
                                "user_query": user_query,
                                "query_length": len(user_query),
                                "has_context": context is not None,
                                "context_keys": list(context.keys()) if context else [],
                                "agent": "market_information"
                            }
                        )
                        
                        # @tool 데코레이터로 래핑된 함수는 .ainvoke()로 호출
                        rdb_result = await rdb_query_llm.ainvoke({
                            "user_request": user_query,
                            "context": context
                        })
                        rdb_llm_elapsed = time.time() - rdb_llm_start_time
                        
                        self.logger.info(
                            f"✅ [RDB LLM COMPLETE] LLM 기반 쿼리 생성 완료",
                            {
                                "execution_id": execution_id,
                                "elapsed_seconds": round(rdb_llm_elapsed, 3),
                                "status": rdb_result.get("status"),
                                "has_sql_query": bool(rdb_result.get("sql_query")),
                                "sql_query": rdb_result.get("sql_query", "")[:200],
                                "sql_query_length": len(rdb_result.get("sql_query", "")),
                                "explanation": rdb_result.get("explanation", "")[:200],
                                "results_count": rdb_result.get("results_count", 0)
                            }
                        )
                        
                        # rdb_llm은 "results" 키를 반환하므로 수정
                        if rdb_result.get("status") == "success":
                            results = rdb_result.get("results", [])
                            
                            self.logger.debug(
                                f"📊 [RDB LLM RESULTS] 결과 데이터 확인",
                                {
                                    "execution_id": execution_id,
                                    "results_count": len(results) if results else 0,
                                    "results_preview": [
                                        {
                                            "date": r.get("date"),
                                            "keys": list(r.keys())[:5]
                                        }
                                        for r in results[:2]
                                    ] if results else []
                                }
                            )
                            
                            # Reference와 Action 생성
                            reference_creation_start = time.time()
                            reference, action = create_reference_and_action_from_tool_result(
                                tool_name="rdb_soft",
                                tool_result=results,
                                source="rdb",
                                query=rdb_result.get("sql_query", ""),
                                input_params={"user_request": user_query, "method": "llm_generated"},
                                metadata={"method": "llm_generated", "sql_query": rdb_result.get("sql_query", "")}
                            )
                            references.append(reference)
                            actions.append(action)
                            collected_data["exchange_rate"] = results
                            
                            self.logger.debug(
                                f"📝 [REFERENCE CREATED] Reference 및 Action 생성 완료",
                                {
                                    "execution_id": execution_id,
                                    "reference_creation_elapsed": round(time.time() - reference_creation_start, 4),
                                    "total_references": len(references),
                                    "total_actions": len(actions)
                                }
                            )
                        else:
                            self.logger.warning(
                                f"⚠️  [RDB LLM ERROR] LLM 쿼리 실행 실패",
                                {
                                    "execution_id": execution_id,
                                    "status": rdb_result.get("status"),
                                    "error": rdb_result.get("error", "알 수 없는 오류"),
                                    "rdb_result": rdb_result
                                }
                            )
                            collected_data["rdb_llm_error"] = rdb_result.get("error", "알 수 없는 오류")
                            collected_data["rdb_llm_status"] = rdb_result.get("status")
                        
                except Exception as e:
                    rdb_query_elapsed = time.time() - rdb_query_start_time
                    
                    # 에러 상세 정보 수집
                    error_details = {
                        "execution_id": execution_id,
                        "query_type": query_type,
                        "query_params": query_params,
                        "elapsed_seconds": round(rdb_query_elapsed, 3),
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "error_args": e.args if hasattr(e, 'args') else None,
                    }
                    
                    # ToolError인 경우 추가 정보
                    if hasattr(e, 'tool_name'):
                        error_details["tool_name"] = e.tool_name
                    if hasattr(e, 'original_error'):
                        error_details["original_error"] = str(e.original_error)
                        error_details["original_error_type"] = type(e.original_error).__name__
                    
                    # 쿼리 실행 컨텍스트 정보
                    if query_type == "get_by_date":
                        date_value = query_params.get("date") or (state.get("date") if state else None)
                        error_details["date_value"] = date_value
                        error_details["state_date"] = state.get("date") if state else None
                        error_details["state_keys"] = list(state.keys()) if state else []
                    elif query_type == "get_by_range":
                        error_details["start_date"] = query_params.get("start_date")
                        error_details["end_date"] = query_params.get("end_date")
                        error_details["limit"] = query_params.get("limit")
                    elif query_type == "get_latest":
                        error_details["limit"] = query_params.get("limit")
                    
                    self.logger.error(
                        f"❌ [RDB QUERY ERROR] RDB 쿼리 실행 실패",
                        error_details,
                        exc_info=True
                    )
                    
                    # 터미널에 직접 출력 (로그 레벨과 관계없이)
                    import traceback
                    error_traceback = traceback.format_exc()
                    print(f"\n{'='*80}")
                    print(f"❌ [RDB QUERY ERROR] 상세 에러 정보")
                    print(f"{'='*80}")
                    print(f"Execution ID: {execution_id}")
                    print(f"Query Type: {query_type}")
                    print(f"Query Params: {query_params}")
                    print(f"Error Type: {type(e).__name__}")
                    print(f"Error Message: {str(e)}")
                    if hasattr(e, 'tool_name'):
                        print(f"Tool Name: {e.tool_name}")
                    if hasattr(e, 'original_error'):
                        print(f"Original Error: {type(e.original_error).__name__}: {str(e.original_error)}")
                    print(f"\nFull Traceback:")
                    print(error_traceback)
                    print(f"{'='*80}\n")
                    
                    collected_data["error"] = str(e)
                    collected_data["error_type"] = type(e).__name__
                    collected_data["error_query_type"] = query_type
                    collected_data["error_details"] = error_details
            
            data_collection_elapsed = time.time() - data_collection_start_time
            self.logger.info(
                f"✅ [DATA COLLECTION COMPLETE] 데이터 수집 완료",
                {
                    "execution_id": execution_id,
                    "elapsed_seconds": round(data_collection_elapsed, 3),
                    "collected_data_keys": list(collected_data.keys()),
                    "collected_data_summary": {
                        k: f"{type(v).__name__}({len(v) if isinstance(v, (list, dict)) else 'N/A'})"
                        for k, v in collected_data.items()
                    },
                    "total_references": len(references),
                    "total_actions": len(actions),
                    "has_exchange_rate": "exchange_rate" in collected_data,
                    "has_web_search": "web_search" in collected_data,
                    "has_errors": any("error" in k for k in collected_data.keys())
                }
            )
            
            # 3. 수집한 데이터를 바탕으로 답변 생성
            answer_generation_start_time = time.time()
            self.logger.info(
                f"✍️  [ANSWER GENERATION START] 답변 생성 단계 시작",
                {
                    "execution_id": execution_id,
                    "user_query": user_query,
                    "collected_data_keys": list(collected_data.keys()),
                    "collected_data_size": sum(
                        len(str(v)) if isinstance(v, (list, dict, str)) else 1
                        for v in collected_data.values()
                    )
                }
            )
            
            answer = await self._generate_answer(user_query, collected_data, state)
            
            answer_generation_elapsed = time.time() - answer_generation_start_time
            execution_elapsed = time.time() - execution_start_time
            
            self.logger.info(
                f"✅ [ANSWER GENERATION COMPLETE] 답변 생성 완료",
                {
                    "execution_id": execution_id,
                    "answer_generation_elapsed_seconds": round(answer_generation_elapsed, 3),
                    "answer_length": len(answer),
                    "answer_preview": answer[:200] if answer else None,
                    "answer_word_count": len(answer.split()) if answer else 0
                }
            )
            
            # Action은 이미 Command에서 추가되었으므로 여기서는 추가하지 않음
            
            self.logger.info(
                f"🎉 [EXECUTION COMPLETE] 시장 정보 에이전트 실행 완료",
                {
                    "execution_id": execution_id,
                    "agent_name": self.name,
                    "total_elapsed_seconds": round(execution_elapsed, 3),
                    "analysis_elapsed": round(analysis_elapsed, 3),
                    "data_collection_elapsed": round(data_collection_elapsed, 3),
                    "answer_generation_elapsed": round(answer_generation_elapsed, 3),
                    "answer_length": len(answer),
                    "total_references": len(references),
                    "total_actions": len(actions),
                    "status": "success",
                    "performance_breakdown": {
                        "analysis_percentage": round((analysis_elapsed / execution_elapsed * 100), 1) if execution_elapsed > 0 else 0,
                        "data_collection_percentage": round((data_collection_elapsed / execution_elapsed * 100), 1) if execution_elapsed > 0 else 0,
                        "answer_generation_percentage": round((answer_generation_elapsed / execution_elapsed * 100), 1) if execution_elapsed > 0 else 0
                    }
                }
            )
            
            return {
                "agent": self.name,
                "task": task,
                "status": "success",
                "answer": answer,
                "reference": [ref.__dict__ for ref in references],
                "action": [act.__dict__ for act in actions]
            }
            
        except Exception as e:
            execution_elapsed = time.time() - execution_start_time
            self.logger.error(
                f"❌ [EXECUTION ERROR] 시장 정보 에이전트 실행 실패",
                {
                    "execution_id": execution_id,
                    "agent_name": self.name,
                    "task": task,
                    "total_elapsed_seconds": round(execution_elapsed, 3),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None,
                    "context_at_error": {
                        "has_state": state is not None if 'state' in locals() else False,
                        "has_collected_data": bool(collected_data) if 'collected_data' in locals() else False,
                        "collected_data_keys": list(collected_data.keys()) if 'collected_data' in locals() and collected_data else [],
                        "total_references": len(references) if 'references' in locals() else 0,
                        "total_actions": len(actions) if 'actions' in locals() else 0
                    }
                },
                exc_info=True
            )
            return {
                "agent": self.name,
                "task": task,
                "error": str(e),
                "status": "error"
            }
    
    def _extract_defaults_from_query(self, user_query: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        사용자 질문에서 기본값 추출
        
        Args:
            user_query: 사용자 질문
            state: 현재 state
            
        Returns:
            추출된 기본값 딕셔너리
        """
        defaults = {}
        query_lower = user_query.lower()
        
        # 오늘 날짜 추출
        today_keywords = ["오늘", "현재", "지금", "today", "now", "현재시점"]
        if any(keyword in query_lower for keyword in today_keywords):
            today = datetime.now().strftime("%Y-%m-%d")
            defaults["date"] = today
            self.logger.debug(f"📅 [DEFAULT] '오늘' 키워드 감지 → 날짜: {today}")
        
        # state에서 날짜 가져오기
        if not defaults.get("date") and state:
            state_date = state.get("date")
            if state_date:
                defaults["date"] = state_date
                self.logger.debug(f"📅 [DEFAULT] State에서 날짜 추출: {state_date}")
        
        # 환율 기본값 (USD/KRW)
        exchange_keywords = ["환율", "exchange", "rate", "달러", "원화", "usd", "krw", "원-달러", "달러-원"]
        if any(keyword in query_lower for keyword in exchange_keywords):
            defaults["currency_pair"] = "USD/KRW"
            defaults["exchange_rate"] = True
            self.logger.debug(f"💱 [DEFAULT] 환율 키워드 감지 → USD/KRW 기본값 사용")
        
        return defaults
    
    async def _analyze_query(self, user_query: str, state: Optional[Dict[str, Any]] = None, previous_collected_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """사용자 질문을 분석하여 필요한 데이터와 쿼리 타입 결정"""
        analysis_id = f"analysis_{int(time.time() * 1000)}"
        
        # 기본값 추출
        inferred_defaults = self._extract_defaults_from_query(user_query, state)
        
        self.logger.debug(
            f"🔍 [ANALYSIS DETAIL] 질문 분석 상세 정보",
            {
                "analysis_id": analysis_id,
                "user_query": user_query,
                "query_length": len(user_query),
                "has_state": state is not None,
                "state_keys": list(state.keys()) if state else [],
                "state_date": state.get("date") if state else None,
                "has_previous_data": bool(previous_collected_data),
                "previous_data_keys": list(previous_collected_data.keys()) if previous_collected_data else [],
                "inferred_defaults": inferred_defaults
            }
        )
        
        # 오늘 날짜 정보 추가
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt = f"""{MARKET_INFORMATION_INSTRUCTION}

사용자의 질문을 분석하여 다음 JSON 형식으로 응답하세요:
{{
    "needs_rdb_query": true/false,
    "needs_web_search": true/false,
    "query_type": "get_by_date" | "get_by_range" | "get_latest" | "rdb_llm" | null,
    "query_params": {{
        "date": "YYYY-MM-DD" (get_by_date인 경우),
        "start_date": "YYYY-MM-DD" (get_by_range인 경우),
        "end_date": "YYYY-MM-DD" (get_by_range인 경우),
        "limit": number (get_by_range 또는 get_latest인 경우)
    }},
    "web_search_query": "검색 쿼리 (needs_web_search가 true인 경우)",
    "web_search_num_results": number (기본값: 5, 최대: 10),
    "reasoning": "분석 이유"
}}

## 중요: 기본값 사용 원칙
**가능한 한 기본값을 사용하여 바로 답변할 수 있도록 분석하세요.**

### 기본값 규칙
1. **날짜 관련**:
   - "오늘", "현재", "지금" 등의 단어가 있으면 → 오늘 날짜({today_date}) 사용
   - **요청 바디나 state에 date가 주어져 있고**, 질문에 "최근", "지난", "일주일" 등의 표현이 있으면
     → 해당 date를 기준으로 상대 기간을 계산 (예: date가 2025-11-24이고 "최근 일주일"이면
        2025-11-18 ~ 2025-11-24 구간을 get_by_range로 조회)
   - 날짜가 명시되지 않았지만 환율/경제지표 조회인 경우 → "get_latest" 사용 (최신 데이터)
   - "어제", "지난주" 등 상대적 표현도 기본값으로 처리

2. **환율 관련**:
   - "환율"만 언급된 경우 → USD/KRW (원-달러)로 처리
   - "달러", "원화" 등이 언급되면 → USD/KRW로 처리
   - 특정 통화쌍이 명시되지 않았지만 환율 문맥이면 → USD/KRW 기본값 사용

3. **쿼리 타입 선택**:
   - 날짜가 명시되지 않았지만 "오늘" 키워드가 있으면 → "get_by_date" + 오늘 날짜
   - 날짜가 전혀 없으면 → "get_latest" (최신 데이터 조회)
   - 날짜 범위가 명시되면 → "get_by_range"
   - 복잡한 조건이 있으면 → "rdb_llm"

### 추론된 기본값
{json.dumps(inferred_defaults, ensure_ascii=False, indent=2, default=str) if inferred_defaults else '없음'}

사용 가능한 쿼리:
- get_by_date: 특정 날짜의 환율 및 경제 지표 조회
- get_by_range: 날짜 범위의 환율 및 경제 지표 조회
- get_latest: 최신 환율 및 경제 지표 조회 (날짜가 없을 때 기본값)
- rdb_llm: 복잡한 쿼리가 필요한 경우 LLM이 쿼리 생성

웹 검색 사용 시기:
- 최신 뉴스나 실시간 정보가 필요할 때
- RDB에 없는 최근 시장 동향이나 뉴스가 필요할 때
- 특정 이벤트나 뉴스에 대한 정보가 필요할 때

현재 state의 date: {state.get('date') if state else 'None'}
오늘 날짜: {today_date}

이전 에이전트가 수집한 데이터:
{json.dumps(previous_collected_data, ensure_ascii=False, indent=2, default=str) if previous_collected_data else '없음'}
"""
        
        user_prompt = f"""사용자 질문: {user_query}

## 분석 지침
1. **기본값 우선 사용**: 질문에서 기본값을 추론할 수 있으면 바로 사용하세요.
   - "오늘" 키워드가 있으면 → 오늘 날짜({today_date}) 사용
   - "환율"만 언급되면 → USD/KRW 기본값 사용
   - 날짜가 없으면 → "get_latest" 사용

2. **추론된 기본값 활용**: 위에서 추론된 기본값을 적극 활용하세요.
{json.dumps(inferred_defaults, ensure_ascii=False, indent=2, default=str) if inferred_defaults else '추론된 기본값 없음'}

3. **이전 데이터 활용**: 이전 에이전트가 수집한 데이터가 있다면 이를 활용하여 추가로 필요한 데이터만 조회하세요.

위 질문을 분석하여 필요한 데이터와 쿼리 타입을 결정해주세요. 가능한 한 기본값을 사용하여 바로 답변할 수 있도록 분석하세요."""
        
        try:
            llm_call_start = time.time()
            self.logger.debug(
                f"📤 [LLM CALL] LLM 호출 시작 (질문 분석)",
                {
                    "analysis_id": analysis_id,
                    "system_prompt_length": len(system_prompt),
                    "user_prompt_length": len(user_prompt),
                    "temperature": 0.3,
                    "response_format": "json_object"
                }
            )
            
            response = await call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            llm_call_elapsed = time.time() - llm_call_start
            self.logger.debug(
                f"📥 [LLM RESPONSE] LLM 응답 수신",
                {
                    "analysis_id": analysis_id,
                    "llm_call_elapsed_seconds": round(llm_call_elapsed, 3),
                    "response_length": len(response),
                    "response_preview": response[:200]
                }
            )
            
            parsing_start = time.time()
            result = json.loads(response)
            parsing_elapsed = time.time() - parsing_start
            
            # 기본값 적용 (추론된 기본값이 있으면 적용)
            if inferred_defaults:
                # 날짜 기본값 적용
                if inferred_defaults.get("date") and not result.get("query_params", {}).get("date"):
                    if "query_params" not in result:
                        result["query_params"] = {}
                    result["query_params"]["date"] = inferred_defaults["date"]
                    result["query_type"] = result.get("query_type") or "get_by_date"
                    result["needs_rdb_query"] = result.get("needs_rdb_query", True)
                    self.logger.debug(
                        f"📅 [DEFAULT APPLIED] 날짜 기본값 적용",
                        {
                            "analysis_id": analysis_id,
                            "applied_date": inferred_defaults["date"]
                        }
                    )
                
                # 날짜가 없고 get_by_date가 선택된 경우 get_latest로 변경
                if result.get("query_type") == "get_by_date" and not result.get("query_params", {}).get("date"):
                    result["query_type"] = "get_latest"
                    result["query_params"] = {"limit": 10}
                    self.logger.debug(
                        f"🔄 [DEFAULT FALLBACK] 날짜 없음 → get_latest로 변경",
                        {
                            "analysis_id": analysis_id
                        }
                    )
            
            # 날짜가 없으면 get_latest 사용
            if result.get("needs_rdb_query") and not result.get("query_type"):
                result["query_type"] = "get_latest"
                result["query_params"] = {"limit": 10}
                self.logger.debug(
                    f"🔄 [DEFAULT] 쿼리 타입 없음 → get_latest로 설정",
                    {
                        "analysis_id": analysis_id
                    }
                )
            
            self.logger.debug(
                f"✅ [ANALYSIS PARSING] 분석 결과 파싱 완료",
                {
                    "analysis_id": analysis_id,
                    "parsing_elapsed_seconds": round(parsing_elapsed, 4),
                    "result_keys": list(result.keys()),
                    "full_result": result,
                    "inferred_defaults_applied": bool(inferred_defaults)
                }
            )
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(
                f"❌ [ANALYSIS JSON ERROR] JSON 파싱 실패",
                {
                    "analysis_id": analysis_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_preview": response[:500] if 'response' in locals() else None,
                    "response_length": len(response) if 'response' in locals() else 0
                },
                exc_info=True
            )
            # 기본값: get_by_date 사용
            date = state.get("date") if state else None
            fallback_result = {
                "needs_rdb_query": True,
                "query_type": "get_by_date" if date else "get_latest",
                "query_params": {"date": date} if date else {"limit": 10},
                "reasoning": f"기본 쿼리 사용 (JSON 파싱 실패: {str(e)})"
            }
            self.logger.warning(
                f"⚠️  [FALLBACK] 기본 쿼리 사용",
                {
                    "analysis_id": analysis_id,
                    "fallback_result": fallback_result
                }
            )
            return fallback_result
            
        except Exception as e:
            self.logger.error(
                f"❌ [ANALYSIS ERROR] 질문 분석 실패",
                {
                    "analysis_id": analysis_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
                },
                exc_info=True
            )
            # 기본값: get_by_date 사용
            date = state.get("date") if state else None
            fallback_result = {
                "needs_rdb_query": True,
                "query_type": "get_by_date" if date else "get_latest",
                "query_params": {"date": date} if date else {"limit": 10},
                "reasoning": f"기본 쿼리 사용 (분석 실패: {str(e)})"
            }
            self.logger.warning(
                f"⚠️  [FALLBACK] 기본 쿼리 사용",
                {
                    "analysis_id": analysis_id,
                    "fallback_result": fallback_result
                }
            )
            return fallback_result
    
    async def _generate_answer(self, user_query: str, collected_data: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> str:
        """수집한 데이터를 바탕으로 답변 생성"""
        answer_id = f"answer_{int(time.time() * 1000)}"
        
        self.logger.debug(
            f"✍️  [ANSWER GENERATION DETAIL] 답변 생성 상세 정보",
            {
                "answer_id": answer_id,
                "user_query": user_query,
                "query_length": len(user_query),
                "collected_data_keys": list(collected_data.keys()),
                "collected_data_summary": {
                    k: {
                        "type": type(v).__name__,
                        "size": len(v) if isinstance(v, (list, dict, str)) else 1,
                        "preview": str(v)[:100] if not isinstance(v, (list, dict)) else None
                    }
                    for k, v in collected_data.items()
                },
                "has_state": state is not None
            }
        )
        
        system_prompt = f"""{MARKET_INFORMATION_INSTRUCTION}

수집한 데이터를 바탕으로 사용자의 질문에 대한 명확하고 정확한 답변을 생성하세요.

## 답변 작성 규칙
1. 수집한 데이터를 정확하게 인용
2. 숫자는 소수점과 단위를 포함하여 표시 (예: "1,157.8원")
3. 데이터가 없는 경우 "해당 날짜의 데이터를 찾을 수 없습니다"라고 명확히 안내
4. 답변은 자연스럽고 이해하기 쉽게 작성
5. 불필요한 설명은 생략하고 핵심 정보만 제공
"""
        
        # 데이터 포맷팅
        data_summary_parts = []
        
        # RDB 데이터
        if collected_data.get("exchange_rate"):
            data = collected_data["exchange_rate"]
            if isinstance(data, list) and len(data) > 0:
                data_summary_parts.append(f"RDB 데이터:\n{json.dumps(data, ensure_ascii=False, indent=2, default=str)}")
            else:
                data_summary_parts.append("RDB 데이터: 없음 (해당 날짜의 데이터를 찾을 수 없습니다)")
        elif collected_data.get("error"):
            data_summary_parts.append(f"RDB 데이터 수집 중 오류 발생: {collected_data['error']}")
        
        # 웹 검색 결과
        if collected_data.get("web_search"):
            web_results = collected_data["web_search"]
            if isinstance(web_results, list) and len(web_results) > 0:
                web_summary = "웹 검색 결과:\n"
                for i, result in enumerate(web_results, 1):
                    web_summary += f"{i}. {result.get('title', '')}\n"
                    web_summary += f"   {result.get('snippet', '')}\n"
                    web_summary += f"   링크: {result.get('link', '')}\n\n"
                data_summary_parts.append(web_summary)
            else:
                data_summary_parts.append("웹 검색 결과: 없음")
        elif collected_data.get("web_search_error"):
            data_summary_parts.append(f"웹 검색 중 오류 발생: {collected_data['web_search_error']}")
        
        # 데이터 요약 결합
        if data_summary_parts:
            data_summary = "\n\n".join(data_summary_parts)
        else:
            data_summary = "수집한 데이터: 없음"
        
        user_prompt = f"""사용자 질문: {user_query}

{data_summary}

위 데이터를 바탕으로 사용자의 질문에 대한 답변을 생성해주세요."""
        
        try:
            llm_call_start = time.time()
            self.logger.debug(
                f"📤 [LLM CALL] LLM 호출 시작 (답변 생성)",
                {
                    "answer_id": answer_id,
                    "system_prompt_length": len(system_prompt),
                    "user_prompt_length": len(user_prompt),
                    "data_summary_length": len(data_summary),
                    "temperature": 0.7
                }
            )
            
            answer = await call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            llm_call_elapsed = time.time() - llm_call_start
            answer_cleaned = answer.strip()
            
            self.logger.debug(
                f"📥 [LLM RESPONSE] LLM 응답 수신 및 처리 완료",
                {
                    "answer_id": answer_id,
                    "llm_call_elapsed_seconds": round(llm_call_elapsed, 3),
                    "original_answer_length": len(answer),
                    "cleaned_answer_length": len(answer_cleaned),
                    "answer_word_count": len(answer_cleaned.split()),
                    "answer_preview": answer_cleaned[:200]
                }
            )
            
            return answer_cleaned
            
        except Exception as e:
            self.logger.error(
                f"❌ [ANSWER GENERATION ERROR] 답변 생성 실패",
                {
                    "answer_id": answer_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None,
                    "collected_data_keys": list(collected_data.keys()),
                    "has_exchange_rate": "exchange_rate" in collected_data
                },
                exc_info=True
            )
            
            # Fallback: 간단한 답변 생성
            fallback_start = time.time()
            if collected_data.get("exchange_rate"):
                data = collected_data["exchange_rate"]
                if isinstance(data, list) and len(data) > 0:
                    latest = data[0]
                    usdkrw = latest.get("usdkrw")
                    date = latest.get("date")
                    if usdkrw:
                        fallback_answer = f"{date} 기준 USD/KRW 환율은 {usdkrw:,.2f}원입니다."
                        self.logger.info(
                            f"✅ [FALLBACK ANSWER] Fallback 답변 생성 완료",
                            {
                                "answer_id": answer_id,
                                "fallback_elapsed_seconds": round(time.time() - fallback_start, 4),
                                "fallback_answer": fallback_answer,
                                "used_data": {"date": date, "usdkrw": usdkrw}
                            }
                        )
                        return fallback_answer
            
            fallback_answer = "데이터를 조회했지만 답변을 생성하는 중 오류가 발생했습니다."
            self.logger.warning(
                f"⚠️  [FALLBACK ANSWER] 기본 Fallback 답변 사용",
                {
                    "answer_id": answer_id,
                    "fallback_elapsed_seconds": round(time.time() - fallback_start, 4),
                    "fallback_answer": fallback_answer
                }
            )
            return fallback_answer

