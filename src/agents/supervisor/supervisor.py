"""Supervisor - 멀티 에이전트 오케스트레이터"""
import os
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from src.utils.settings import settings
from src.agents.market_information_agent import MarketInformationAgent
from src.agents.reask_agent import ReAskAgent
from src.agents.react_agent import ReActAgent
from src.agents.handsoff_agent import HandsOffAgent
from src.prompts.supervisor_instruction import SUPERVISOR_INSTRUCTION
from src.prompts.supervisor_routing import SUPERVISOR_ROUTING_SYSTEM_PROMPT, SUPERVISOR_ROUTING_USER_PROMPT_TEMPLATE
from src.prompts.market_information_description import MARKET_INFORMATION_DESCRIPTION
from src.utils.logger import get_logger
from src.utils.openai_tracer import TracedOpenAIClient
from src.utils.state import AgentState, AdditionalInfo, Reference, Action

# LangSmith tracing 설정
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"


class Supervisor:
    """멀티 에이전트 시스템의 Supervisor"""
    
    def __init__(self):
        self.logger = get_logger("supervisor")
        # LangSmith 추적이 포함된 OpenAI 클라이언트 사용
        self.client = TracedOpenAIClient(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        
        self.logger.info("🔧 Supervisor 초기화 시작")
        
        # 하위 에이전트 초기화
        self.agents = {
            "market_information": MarketInformationAgent(),
            "reask": ReAskAgent(),
            "react": ReActAgent(),
            "handsoff": HandsOffAgent()
        }
        
        # 에이전트 설명 (라우팅에 사용)
        self.agent_descriptions = MARKET_INFORMATION_DESCRIPTION
        
        self.logger.info(
            f"✅ Supervisor 초기화 완료",
            {"available_agents": list(self.agents.keys())}
        )
    
    async def route_task(
        self,
        user_query: str,
        user_id: str,
        date: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        사용자 질문을 분석하여 적절한 에이전트에게 라우팅
        
        Args:
            user_query: 사용자 질문
            user_id: 사용자 아이디
            date: 질문하는 날짜 (YYYY-MM-DD)
            state: AgentState (대화 히스토리 및 상태 포함)
            
        Returns:
            에이전트 실행 결과 (additional_info 포함)
        """
        self.logger.info(
            f"🔄 작업 라우팅 시작",
            {
                "query_length": len(user_query),
                "query_preview": user_query[:100],
                "user_id": user_id,
                "date": date
            }
        )
        
        # State에서 대화 이력 가져오기
        conversation_history = state.get_recent_history(limit=10)
        
        # Reference와 Action 추적용 리스트
        references: List[Reference] = []
        actions: List[Action] = []
        
        try:
            # 1. ReAsk 노드: 정보 충분성 확인
            self.logger.debug("🔍 ReAsk 노드: 정보 충분성 확인 중...")
            reask_result = await self.agents["reask"].check_and_ask(
                user_query,
                context={"state": state.to_dict()} if state else None,
                conversation_history=conversation_history
            )
            
            if reask_result.get("needs_clarification", False):
                self.logger.info("❓ 정보 부족으로 재질문 필요")
                clarification_question = reask_result.get("clarification_question", "")
                
                # AdditionalInfo 생성
                additional_info = AdditionalInfo(
                    answer=f"[재질문] {clarification_question}",
                    reference=references,
                    action=actions
                )
                
                return {
                    "status": "needs_clarification",
                    "answer": clarification_question,
                    "additional_info": additional_info,
                    "missing_info": reask_result.get("missing_info", []),
                    "reasoning": reask_result.get("reasoning", ""),
                    "agent": "reask"
                }
            
            # 2. LLM을 통해 적절한 에이전트 라우팅 결정
            self.logger.debug("🤔 에이전트 라우팅 결정 중...")
            routing_decision = await self._select_agent(user_query, conversation_history)
            routing = routing_decision.get("routing", [])
            
            self.logger.info(
                f"✅ 라우팅 결정 완료",
                {
                    "routing": routing,
                    "reasoning": routing_decision.get("reasoning", ""),
                    "task_breakdown": routing_decision.get("task_breakdown", {})
                }
            )
            
            # 2. 라우팅에 따라 에이전트 실행 (Sequential 또는 Parallel)
            results = []
            if isinstance(routing, list) and len(routing) > 0:
                # 첫 번째 레벨이 리스트인지 확인 (Parallel)
                if isinstance(routing[0], list):
                    # Parallel 실행
                    self.logger.info("🔄 병렬 에이전트 실행 시작")
                    import asyncio
                    parallel_results = []
                    for agent_group in routing:
                        group_tasks = []
                        for agent_name in agent_group:
                            if agent_name in self.agents:
                                task = routing_decision.get("task_breakdown", {}).get(agent_name, user_query)
                                group_tasks.append(self.agents[agent_name].execute(task))
                            else:
                                self.logger.warning(f"⚠️ 알 수 없는 에이전트: {agent_name}")
                        if group_tasks:
                            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
                            parallel_results.extend(group_results)
                    results = parallel_results
                else:
                    # Sequential 실행
                    self.logger.info("🔄 순차 에이전트 실행 시작")
                    collected_data = {}
                    for agent_name in routing:
                        if agent_name in self.agents:
                            task = routing_decision.get("task_breakdown", {}).get(agent_name, user_query)
                            
                            # Context에 state 및 collected data 포함
                            context = {
                                "state": state.to_dict(),
                                "collected_data": collected_data,
                                "user_id": user_id,
                                "date": date
                            }
                            
                            result = await self.agents[agent_name].execute(task, context)
                            results.append(result)
                            
                            # Reference 추출 (rdb, vdb 사용 시)
                            if "reference" in result:
                                if isinstance(result["reference"], list):
                                    references.extend([Reference(**ref) for ref in result["reference"]])
                            
                            # Action 추출 (modify, calculate 등)
                            if "action" in result:
                                if isinstance(result["action"], list):
                                    actions.extend([Action(**act) for act in result["action"]])
                                elif isinstance(result["action"], dict):
                                    actions.append(Action(**result["action"]))
                            
                            # 수집된 데이터 업데이트
                            if result.get("status") == "success":
                                collected_data[agent_name] = result
                            
                            # Hands-off 체크: 에이전트가 직접 답변할지 결정
                            if agent_name != "handsoff" and agent_name != "reask":
                                handsoff_result = await self.agents["handsoff"].decide(
                                    user_query,
                                    collected_data=collected_data,
                                    context=context
                                )
                                
                                if handsoff_result.get("decision") == "forward":
                                    # 직접 답변
                                    self.logger.info("✅ 에이전트가 직접 답변 결정")
                                    answer = handsoff_result.get("answer", "")
                                    
                                    # AdditionalInfo 생성
                                    additional_info = AdditionalInfo(
                                        answer=answer,
                                        reference=references,
                                        action=actions
                                    )
                                    
                                    return {
                                        "agent": agent_name,
                                        "answer": answer,
                                        "additional_info": additional_info,
                                        "collected_data": collected_data,
                                        "handsoff_decision": handsoff_result,
                                        "status": "success"
                                    }
                                else:
                                    # Supervisor에게 넘김 (계속 진행)
                                    self.logger.info("🔄 Supervisor에게 넘김 결정, 계속 진행")
                        else:
                            self.logger.warning(f"⚠️ 알 수 없는 에이전트: {agent_name}")
            
            # 3. 결과 통합 및 AdditionalInfo 생성
            if len(results) == 1:
                final_result = results[0]
                answer = final_result.get("answer", final_result.get("message", ""))
            else:
                # 여러 결과를 통합
                answer = "여러 에이전트의 결과를 종합하여 답변을 생성했습니다."
                final_result = {
                    "agent": "multiple",
                    "results": results,
                    "status": "success"
                }
            
            # AdditionalInfo 생성 (answer, reference, action 포함)
            additional_info = AdditionalInfo(
                answer=answer,
                reference=references,
                action=actions
            )
            
            # 4. Supervisor 메타데이터 추가
            final_result["supervisor_decision"] = routing_decision
            final_result["additional_info"] = additional_info
            final_result["answer"] = answer
            
            self.logger.info(
                f"✅ 작업 라우팅 완료",
                {
                    "routing": routing,
                    "status": final_result.get("status"),
                    "results_count": len(results),
                    "references_count": len(references),
                    "actions_count": len(actions)
                }
            )
            
            return final_result
        except Exception as e:
            self.logger.error(
                f"❌ 작업 라우팅 실패",
                {"error": str(e)},
                exc_info=True
            )
            return {
                "error": str(e),
                "status": "error"
            }
    
    async def _select_agent(self, user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, str]:
        """LLM을 통해 적절한 에이전트 선택 및 라우팅"""
        messages = [
            {
                "role": "system", 
                "content": SUPERVISOR_ROUTING_SYSTEM_PROMPT.format(
                    agent_descriptions=self.agent_descriptions
                )
            }
        ]
        
        # Conversation history 추가
        if conversation_history:
            for msg in conversation_history[-10:]:  # 최근 10개만 포함
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        messages.append({
            "role": "user",
            "content": SUPERVISOR_ROUTING_USER_PROMPT_TEMPLATE.format(user_query=user_query)
        })
        
        self.logger.debug(
            "📤 Supervisor LLM 호출",
            {"model": self.model, "query_preview": user_query[:50]}
        )
        
        response = self.client.chat_completions_create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        self.logger.debug(f"📥 Supervisor LLM 응답 수신", {"result": result})
        return result
    
    def get_available_agents(self) -> Dict[str, Any]:
        """사용 가능한 에이전트 목록 반환"""
        return {
            name: agent.get_capabilities()
            for name, agent in self.agents.items()
        }
    

