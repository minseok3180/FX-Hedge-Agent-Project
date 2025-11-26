"""시장 정보 에이전트"""
import json
from typing import Dict, Any, Optional, List
from src.utils.agents import BaseAgent
from src.tools.rdb import rdb_query_hard, rdb_query_llm
from src.tools.vdb import vdb_search
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
        self.logger.info(f"📊 시장 정보 에이전트 실행 시작 - task: {task}")
        
        try:
            # Context에서 state, collected_data 추출
            state = context.get("state") if context else None
            previous_collected_data = context.get("collected_data", {}) if context else {}
            user_query = task
            
            # Reference와 Action 추적
            references: List[Reference] = []
            actions: List[Action] = []
            
            # 1. LLM을 통해 필요한 데이터 파악 및 쿼리 결정
            # 이전 에이전트의 collected_data를 고려하여 분석
            analysis_result = await self._analyze_query(user_query, state, previous_collected_data)
            
            # 2. 데이터 수집 (이전 에이전트의 데이터와 병합)
            collected_data = previous_collected_data.copy()
            
            # 2-1. 웹 검색이 필요한 경우
            if analysis_result.get("needs_web_search"):
                search_query = analysis_result.get("web_search_query", user_query)
                num_results = analysis_result.get("web_search_num_results", 5)
                
                self.logger.info(f"🔍 웹 검색 실행 - query: {search_query}, num_results: {num_results}")
                
                try:
                    web_results = await web_search(search_query, num_results=num_results)
                    
                    # Reference와 Action 생성
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
                    
                except Exception as e:
                    self.logger.error(f"❌ 웹 검색 실행 실패: {str(e)}", exc_info=True)
                    collected_data["web_search_error"] = str(e)
            
            # 2-2. RDB 쿼리가 필요한 경우
            if analysis_result.get("needs_rdb_query"):
                query_type = analysis_result.get("query_type")
                query_params = analysis_result.get("query_params", {})
                
                self.logger.info(f"🔍 RDB 쿼리 실행 - type: {query_type}, params: {query_params}")
                
                try:
                    if query_type == "get_by_date":
                        # get_by_date는 {date} placeholder를 사용
                        # query_params에 date가 있으면 사용, 없으면 state에서 가져옴
                        date_value = query_params.get("date") or (state.get("date") if state else None)
                        
                        if not date_value:
                            # date가 없으면 get_latest로 대체
                            self.logger.warning("⚠️ get_by_date에 date가 없어 get_latest로 대체")
                            results = await rdb_query_hard("get_latest", params=(10,), state=state)
                            
                            # Reference와 Action 생성
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
                        else:
                            # state에 date를 설정하여 placeholder 치환
                            if state:
                                state_with_date = state.copy()
                                state_with_date["date"] = date_value
                            else:
                                state_with_date = {"date": date_value}
                            results = await rdb_query_hard("get_by_date", state=state_with_date)
                        
                        # Reference와 Action 생성
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
                        
                    elif query_type == "get_by_range":
                        start_date = query_params.get("start_date")
                        end_date = query_params.get("end_date")
                        limit = query_params.get("limit", 100)
                        results = await rdb_query_hard(
                            "get_by_range",
                            params=(start_date, end_date, limit),
                            state=state
                        )
                        
                        # Reference와 Action 생성
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
                        
                    elif query_type == "get_latest":
                        limit = query_params.get("limit", 10)
                        results = await rdb_query_hard(
                            "get_latest",
                            params=(limit,),
                            state=state
                        )
                        
                        # Reference와 Action 생성
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
                        
                    elif query_type == "rdb_llm":
                        # LLM이 쿼리를 생성하여 실행
                        rdb_result = await rdb_query_llm(user_query, context)
                        
                        # rdb_llm은 "results" 키를 반환하므로 수정
                        if rdb_result.get("status") == "success":
                            results = rdb_result.get("results", [])
                            
                            # Reference와 Action 생성
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
                        else:
                            collected_data["rdb_llm_error"] = rdb_result.get("error", "알 수 없는 오류")
                        
                except Exception as e:
                    self.logger.error(f"❌ RDB 쿼리 실행 실패: {str(e)}", exc_info=True)
                    collected_data["error"] = str(e)
            
            # 3. 수집한 데이터를 바탕으로 답변 생성
            answer = await self._generate_answer(user_query, collected_data, state)
            
            # Action은 이미 Command에서 추가되었으므로 여기서는 추가하지 않음
            
            self.logger.info(f"✅ 시장 정보 에이전트 실행 완료 - answer_length: {len(answer)}")
            
            return {
                "agent": self.name,
                "task": task,
                "status": "success",
                "answer": answer,
                "reference": [ref.__dict__ for ref in references],
                "action": [act.__dict__ for act in actions]
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시장 정보 에이전트 실행 실패: {str(e)}", exc_info=True)
            return {
                "agent": self.name,
                "task": task,
                "error": str(e),
                "status": "error"
            }
    
    async def _analyze_query(self, user_query: str, state: Optional[Dict[str, Any]] = None, previous_collected_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """사용자 질문을 분석하여 필요한 데이터와 쿼리 타입 결정"""
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

사용 가능한 쿼리:
- get_by_date: 특정 날짜의 환율 및 경제 지표 조회
- get_by_range: 날짜 범위의 환율 및 경제 지표 조회
- get_latest: 최신 환율 및 경제 지표 조회
- rdb_llm: 복잡한 쿼리가 필요한 경우 LLM이 쿼리 생성

웹 검색 사용 시기:
- 최신 뉴스나 실시간 정보가 필요할 때
- RDB에 없는 최근 시장 동향이나 뉴스가 필요할 때
- 특정 이벤트나 뉴스에 대한 정보가 필요할 때

현재 state의 date: {state.get('date') if state else 'None'}

이전 에이전트가 수집한 데이터:
{json.dumps(previous_collected_data, ensure_ascii=False, indent=2) if previous_collected_data else '없음'}
"""
        
        user_prompt = f"""사용자 질문: {user_query}

이전 에이전트가 수집한 데이터가 있다면 이를 활용하여 추가로 필요한 데이터만 조회하세요.
위 질문을 분석하여 필요한 데이터와 쿼리 타입을 결정해주세요."""
        
        try:
            response = await call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response)
            self.logger.debug(f"📊 질문 분석 결과: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 질문 분석 실패: {str(e)}", exc_info=True)
            # 기본값: get_by_date 사용
            date = state.get("date") if state else None
            return {
                "needs_rdb_query": True,
                "query_type": "get_by_date" if date else "get_latest",
                "query_params": {"date": date} if date else {"limit": 10},
                "reasoning": f"기본 쿼리 사용 (분석 실패: {str(e)})"
            }
    
    async def _generate_answer(self, user_query: str, collected_data: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> str:
        """수집한 데이터를 바탕으로 답변 생성"""
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
                data_summary_parts.append(f"RDB 데이터:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
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
            answer = await call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            return answer.strip()
            
        except Exception as e:
            self.logger.error(f"❌ 답변 생성 실패: {str(e)}", exc_info=True)
            
            # Fallback: 간단한 답변 생성
            if collected_data.get("exchange_rate"):
                data = collected_data["exchange_rate"]
                if isinstance(data, list) and len(data) > 0:
                    latest = data[0]
                    usdkrw = latest.get("usdkrw")
                    date = latest.get("date")
                    if usdkrw:
                        return f"{date} 기준 USD/KRW 환율은 {usdkrw:,.2f}원입니다."
            
            return "데이터를 조회했지만 답변을 생성하는 중 오류가 발생했습니다."

