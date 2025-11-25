"""시장 정보 에이전트"""
import json
from typing import Dict, Any, Optional, List
from src.agents.base_agent import BaseAgent
from src.tools.rdb_query import RDBHardTool
from src.tools.rdb_llm import RDBSoftTool
from src.tools.vdb import QdrantTool
from src.tools.web_search import WebSearchTool
from src.prompts.market_information_instruction import MARKET_INFORMATION_INSTRUCTION
from src.utils.gpt_client import call_gpt
from src.utils.state import Reference, Action


class MarketInformationAgent(BaseAgent):
    """시장 정보를 수집하고 분석하는 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="market_information",
            description="시장 정보를 수집하고 분석하여 헤지 전략 수립에 필요한 인사이트를 제공하는 에이전트"
        )
        
        # 툴 초기화
        self.rdb_hard = RDBHardTool()  # Query 폴더의 쿼리를 사용하는 하드 쿼리 툴
        self.rdb_soft = RDBSoftTool()  # LLM이 쿼리를 생성하는 소프트 쿼리 툴
        self.vdb = QdrantTool()  # 벡터 데이터베이스 툴
        self.web_search = WebSearchTool()  # 웹 검색 툴
    
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
            # Context에서 state 추출
            state = context.get("state") if context else None
            user_query = task
            
            # Reference와 Action 추적
            references: List[Reference] = []
            actions: List[Action] = []
            
            # 1. LLM을 통해 필요한 데이터 파악 및 쿼리 결정
            analysis_result = await self._analyze_query(user_query, state)
            
            # 2. 데이터 수집
            collected_data = {}
            if analysis_result.get("needs_rdb_query"):
                query_type = analysis_result.get("query_type")
                query_params = analysis_result.get("query_params", {})
                
                self.logger.info(f"🔍 RDB 쿼리 실행 - type: {query_type}, params: {query_params}")
                
                try:
                    if query_type == "get_by_date":
                        results = await self.rdb_hard.execute("get_by_date", state=state)
                        collected_data["exchange_rate"] = results
                        
                        # Reference 추가
                        references.append(Reference(
                            source="rdb",
                            query="get_by_date",
                            results_count=len(results),
                            metadata={"date": state.get("date") if state else query_params.get("date")}
                        ))
                        
                    elif query_type == "get_by_range":
                        start_date = query_params.get("start_date")
                        end_date = query_params.get("end_date")
                        limit = query_params.get("limit", 100)
                        results = await self.rdb_hard.execute(
                            "get_by_range",
                            params=(start_date, end_date, limit),
                            state=state
                        )
                        collected_data["exchange_rate"] = results
                        
                        references.append(Reference(
                            source="rdb",
                            query="get_by_range",
                            results_count=len(results),
                            metadata={"start_date": start_date, "end_date": end_date}
                        ))
                        
                    elif query_type == "get_latest":
                        limit = query_params.get("limit", 10)
                        results = await self.rdb_hard.execute(
                            "get_latest",
                            params=(limit,),
                            state=state
                        )
                        collected_data["exchange_rate"] = results
                        
                        references.append(Reference(
                            source="rdb",
                            query="get_latest",
                            results_count=len(results),
                            metadata={"limit": limit}
                        ))
                        
                    elif query_type == "rdb_llm":
                        # LLM이 쿼리를 생성하여 실행
                        rdb_result = await self.rdb_soft.generate_and_execute(user_query, context)
                        collected_data.update(rdb_result.get("data", {}))
                        
                        references.append(Reference(
                            source="rdb",
                            query=rdb_result.get("sql_query", ""),
                            results_count=rdb_result.get("results_count", 0),
                            metadata={"method": "llm_generated"}
                        ))
                        
                except Exception as e:
                    self.logger.error(f"❌ RDB 쿼리 실행 실패: {str(e)}", exc_info=True)
                    collected_data["error"] = str(e)
            
            # 3. 수집한 데이터를 바탕으로 답변 생성
            answer = await self._generate_answer(user_query, collected_data, state)
            
            # Action 추가
            actions.append(Action(
                type="query",
                tool="rdb_query",
                description=f"환율 및 시장 정보 조회: {query_type if analysis_result.get('needs_rdb_query') else 'N/A'}",
                input={"query_type": analysis_result.get("query_type"), "query_params": analysis_result.get("query_params")},
                output={"results_count": len(collected_data.get("exchange_rate", []))}
            ))
            
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
    
    async def _analyze_query(self, user_query: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """사용자 질문을 분석하여 필요한 데이터와 쿼리 타입 결정"""
        system_prompt = f"""{MARKET_INFORMATION_INSTRUCTION}

사용자의 질문을 분석하여 다음 JSON 형식으로 응답하세요:
{{
    "needs_rdb_query": true/false,
    "query_type": "get_by_date" | "get_by_range" | "get_latest" | "rdb_llm" | null,
    "query_params": {{
        "date": "YYYY-MM-DD" (get_by_date인 경우),
        "start_date": "YYYY-MM-DD" (get_by_range인 경우),
        "end_date": "YYYY-MM-DD" (get_by_range인 경우),
        "limit": number (get_by_range 또는 get_latest인 경우)
    }},
    "reasoning": "분석 이유"
}}

사용 가능한 쿼리:
- get_by_date: 특정 날짜의 환율 및 경제 지표 조회
- get_by_range: 날짜 범위의 환율 및 경제 지표 조회
- get_latest: 최신 환율 및 경제 지표 조회
- rdb_llm: 복잡한 쿼리가 필요한 경우 LLM이 쿼리 생성

현재 state의 date: {state.get('date') if state else 'None'}
"""
        
        user_prompt = f"사용자 질문: {user_query}\n\n위 질문을 분석하여 필요한 데이터와 쿼리 타입을 결정해주세요."
        
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
        data_summary = ""
        if collected_data.get("exchange_rate"):
            data = collected_data["exchange_rate"]
            if isinstance(data, list) and len(data) > 0:
                data_summary = f"수집한 데이터:\n{json.dumps(data, ensure_ascii=False, indent=2)}"
            else:
                data_summary = "수집한 데이터: 없음 (해당 날짜의 데이터를 찾을 수 없습니다)"
        elif collected_data.get("error"):
            data_summary = f"데이터 수집 중 오류 발생: {collected_data['error']}"
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

