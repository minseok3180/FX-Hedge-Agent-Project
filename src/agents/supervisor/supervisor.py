"""Supervisor - 멀티 에이전트 오케스트레이터 (LangGraph 기반)"""
from __future__ import annotations
import json
from typing import Dict, Any, Optional, List, Literal, Annotated, TYPE_CHECKING
from operator import add
from src.utils.settings import settings
from src.agents.market_information_agent import MarketInformationAgent
from src.agents.expert_information_agent import ExpertInformationAgent
from src.agents.user_information_agent import UserInformationAgent
from src.agents.reask_agent import ReAskAgent
from src.agents.react_agent import ReActAgent
from src.agents.handsoff_agent import HandsOffAgent
from src.prompts.supervisor_routing import SUPERVISOR_ROUTING_SYSTEM_PROMPT, SUPERVISOR_ROUTING_USER_PROMPT_TEMPLATE
from src.prompts.market_information_routing import MARKET_INFORMATION_ROUTING
from src.prompts.expert_information_routing import EXPERT_INFORMATION_ROUTING
from src.prompts.user_information_routing import USER_INFORMATION_ROUTING
from src.utils.logger import get_logger
from src.utils.llm import LANGCHAIN_OPENAI_AVAILABLE, convert_dict_messages_to_langchain
from src.utils.state import AgentState, AdditionalInfo, Reference, Action

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Command
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    Command = None
    StateGraph = None
    START = None
    END = None
    TypedDict = dict

# 타입 힌트용 (TYPE_CHECKING 사용)
if TYPE_CHECKING:
    from langgraph.types import Command as CommandType
    from langgraph.graph import StateGraph as StateGraphType
else:
    # 런타임 타입 (간소화)
    CommandType = Command if (LANGGRAPH_AVAILABLE and Command is not None) else Any
    StateGraphType = StateGraph if (LANGGRAPH_AVAILABLE and StateGraph is not None) else Any

# LangSmith tracing은 openai_tracer 모듈에서 환경 변수로 설정됨


class SupervisorState(TypedDict):
    """Supervisor 그래프의 State"""
    user_query: str
    user_id: str
    date: str
    agent_state: AgentState  # AgentState 객체
    conversation_history: List[Dict[str, str]]
    routing_decision: Optional[Dict[str, Any]]
    collected_data: Dict[str, Any]
    references: Annotated[List[Dict[str, Any]], add]  # LangGraph의 reducer 사용 (dict로 저장)
    actions: Annotated[List[Dict[str, Any]], add]  # LangGraph의 reducer 사용 (dict로 저장)
    current_agent: Optional[str]
    agent_results: List[Dict[str, Any]]
    needs_clarification: bool
    clarification_question: Optional[str]
    final_answer: Optional[str]
    handsoff_decision: Optional[Dict[str, Any]]  # Hands-off 결정
    additional_info: Optional[Dict[str, Any]]  # dict로 저장
    status: str


class Supervisor:
    """멀티 에이전트 시스템의 Supervisor (LangGraph 기반)"""
    
    def __init__(self):
        self.logger = get_logger("supervisor")
        self.model = settings.openai_model
        
        # LangChain OpenAI 클라이언트 사용 (LangSmith 자동 추적)
        if LANGCHAIN_OPENAI_AVAILABLE:
            from langchain_openai import ChatOpenAI
            self.client = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.3
            )
        else:
            # Fallback: OpenAI SDK 직접 사용
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
        
        self.logger.info("🔧 Supervisor 초기화 시작")
        
        # 하위 에이전트 초기화
        self.agents = {
            "market_information": MarketInformationAgent(),
            "expert_information": ExpertInformationAgent(),
            "user_information": UserInformationAgent(),
            "reask": ReAskAgent(),
            "react": ReActAgent(),
            "handsoff": HandsOffAgent()
        }
        
        # 에이전트 설명 (라우팅에 사용) - 모든 에이전트 라우팅 설명 결합
        self.agent_descriptions = (
            f"{MARKET_INFORMATION_ROUTING}\n\n"
            f"{EXPERT_INFORMATION_ROUTING}\n\n"
            f"{USER_INFORMATION_ROUTING}"
        )
        
        # LangGraph 그래프 구성
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
            self.logger.info("✅ LangGraph 그래프 구성 완료")
        else:
            self.graph = None
            self.logger.warning("⚠️ LangGraph가 설치되지 않아 기본 모드로 동작합니다")
        
        self.logger.info(
            f"✅ Supervisor 초기화 완료",
            {"available_agents": list(self.agents.keys())}
        )
    
    def _build_graph(self) -> StateGraph:
        """LangGraph StateGraph 구성"""
        graph = StateGraph(SupervisorState)
        
        # 노드 추가
        graph.add_node("reask", self._reask_node)
        graph.add_node("routing", self._routing_node)
        graph.add_node("agent_execution", self._agent_execution_node)
        graph.add_node("handsoff", self._handsoff_node)
        graph.add_node("final_answer", self._final_answer_node)
        
        # 엣지 추가
        graph.add_edge(START, "reask")
        graph.add_conditional_edges(
            "reask",
            self._should_clarify,
            {
                "clarify": END,  # 재질문 필요 시 종료
                "continue": "routing"
            }
        )
        graph.add_edge("routing", "agent_execution")
        graph.add_conditional_edges(
            "agent_execution",
            self._should_handsoff,
            {
                "handsoff": "handsoff",
                "continue": "final_answer"
            }
        )
        graph.add_conditional_edges(
            "handsoff",
            self._handsoff_decision,
            {
                "forward": "final_answer",
                "continue": "routing"  # 추가 정보 필요 시 다시 라우팅
            }
        )
        graph.add_edge("final_answer", END)
        
        # 그래프 컴파일 (Checkpoint 추가 - 상태 영속성)
        if LANGGRAPH_AVAILABLE:
            try:
                from langgraph.checkpoint.memory import MemorySaver
                memory = MemorySaver()
                return graph.compile(checkpointer=memory)
            except ImportError:
                self.logger.warning("⚠️  MemorySaver를 사용할 수 없어 checkpoint 없이 컴파일합니다.")
                return graph.compile()
        else:
            return graph.compile()
    
    async def _reask_node(self, state: SupervisorState) -> Dict[str, Any]:
        """ReAsk 노드: 정보 충분성 확인"""
        self.logger.debug("🔍 ReAsk 노드 실행")
        
        user_query = state["user_query"]
        agent_state = state["agent_state"]
        conversation_history = state["conversation_history"]
        
        reask_result = await self.agents["reask"].check_and_ask(
            user_query,
            context={"state": agent_state.to_dict()} if agent_state else None,
            conversation_history=conversation_history
        )
        
        if reask_result.get("needs_clarification", False):
            # 재질문 필요 - conditional edge가 "clarify": END로 처리
            return {
                "needs_clarification": True,
                "clarification_question": reask_result.get("clarification_question", ""),
                "status": "needs_clarification"
            }
        else:
            # 계속 진행 - conditional edge가 "continue": "routing"로 처리
            return {
                "needs_clarification": False
            }
    
    async def _routing_node(self, state: SupervisorState) -> Dict[str, Any]:
        """라우팅 노드: 적절한 에이전트 선택"""
        self.logger.debug("🤔 라우팅 노드 실행")
        
        user_query = state["user_query"]
        conversation_history = state["conversation_history"]
        
        self.logger.info(
            f"🔄 [ROUTING] 에이전트 라우팅 시작",
            {
                "user_query": user_query,
                "conversation_history_length": len(conversation_history)
            }
        )
        
        routing_decision = await self._select_agent(user_query, conversation_history)
        
        routing = routing_decision.get("routing", [])
        self.logger.info(
            f"✅ [ROUTING] 라우팅 결정 완료",
            {
                "routing_decision": routing_decision,
                "selected_agents": routing,
                "agents_count": len(routing),
                "task_breakdown": routing_decision.get("task_breakdown", {})
            }
        )
        
        return {
            "routing_decision": routing_decision
        }
    
    async def _agent_execution_node(self, state: SupervisorState) -> Dict[str, Any]:
        """에이전트 실행 노드"""
        self.logger.debug("🚀 에이전트 실행 노드")
        
        routing_decision = state.get("routing_decision", {})
        routing = routing_decision.get("routing", [])
        user_query = state["user_query"]
        agent_state = state["agent_state"]
        collected_data = state.get("collected_data", {})
        user_id = state["user_id"]
        date = state["date"]
        
        if not routing or not isinstance(routing, list):
            # 에러 발생 시 final_answer를 설정하여 conditional edge가 "continue" -> "final_answer"로 라우팅
            return {
                "status": "error",
                "final_answer": "라우팅 정보가 없습니다."
            }
        
        # Sequential 실행
        agent_results = []
        updated_collected_data = collected_data.copy()
        new_references = []  # 이번 노드에서 추가할 references
        new_actions = []  # 이번 노드에서 추가할 actions
        
        for agent_name in routing:
            if agent_name not in self.agents:
                self.logger.warning(f"⚠️ 알 수 없는 에이전트: {agent_name}")
                continue
            
            task = routing_decision.get("task_breakdown", {}).get(agent_name, user_query)
            
            self.logger.info(
                f"🚀 [AGENT EXECUTION] 에이전트 실행 시작",
                {
                    "agent_name": agent_name,
                    "task": task,
                    "user_query": user_query,
                    "user_id": user_id,
                    "date": date
                }
            )
            
            context = {
                "state": agent_state.to_dict(),
                "collected_data": updated_collected_data,
                "user_id": user_id,
                "date": date
            }
            
            result = await self.agents[agent_name].execute(task, context)
            agent_results.append(result)
            
            self.logger.info(
                f"✅ [AGENT EXECUTION] 에이전트 실행 완료",
                {
                    "agent_name": agent_name,
                    "status": result.get("status", "unknown"),
                    "has_answer": "answer" in result or "message" in result,
                    "has_reference": "reference" in result,
                    "has_action": "action" in result
                }
            )
            
            # Reference와 Action 추출 (dict 형태로 변환)
            if "reference" in result:
                if isinstance(result["reference"], list):
                    # Reference 객체를 dict로 변환
                    ref_dicts = [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in result["reference"]]
                    new_references.extend(ref_dicts)
            
            if "action" in result:
                if isinstance(result["action"], list):
                    # Action 객체를 dict로 변환
                    act_dicts = [act.__dict__ if hasattr(act, '__dict__') else act for act in result["action"]]
                    new_actions.extend(act_dicts)
                elif isinstance(result["action"], dict):
                    new_actions.append(result["action"])
            
            # Collected data 업데이트
            if result.get("status") == "success":
                updated_collected_data[agent_name] = result
            
            # Hands-off 체크
            if agent_name != "handsoff" and agent_name != "reask":
                handsoff_result = await self.agents["handsoff"].decide(
                    user_query,
                    collected_data=updated_collected_data,
                    context=context
                )
                
                if handsoff_result.get("decision") == "forward":
                    # 직접 답변 - final_answer를 설정하여 conditional edge가 "continue" -> "final_answer"로 라우팅
                    return {
                        "collected_data": updated_collected_data,
                        "references": new_references,  # reducer가 자동으로 append
                        "actions": new_actions,  # reducer가 자동으로 append
                        "agent_results": agent_results,
                        "current_agent": agent_name,
                        "final_answer": handsoff_result.get("answer", ""),
                        "status": "success"
                    }
        
        # 계속 진행 (Supervisor가 최종 답변 생성)
        # final_answer가 없으므로 conditional edge가 "handsoff" -> "handsoff" 노드로 라우팅
        return {
            "collected_data": updated_collected_data,
            "references": new_references,  # reducer가 자동으로 append
            "actions": new_actions,  # reducer가 자동으로 append
            "agent_results": agent_results
        }
    
    async def _handsoff_node(self, state: SupervisorState) -> Dict[str, Any]:
        """Hands-off 노드: 최종 답변 생성 여부 결정"""
        self.logger.debug("🤔 Hands-off 노드 실행")
        
        user_query = state["user_query"]
        collected_data = state.get("collected_data", {})
        agent_state = state["agent_state"]
        user_id = state["user_id"]
        date = state["date"]
        
        context = {
            "state": agent_state.to_dict(),
            "collected_data": collected_data,
            "user_id": user_id,
            "date": date
        }
        
        handsoff_result = await self.agents["handsoff"].decide(
            user_query,
            collected_data=collected_data,
            context=context
        )
        
        return {
            "handsoff_decision": handsoff_result
        }
    
    async def _final_answer_node(self, state: SupervisorState) -> Dict[str, Any]:
        """최종 답변 생성 노드"""
        self.logger.debug("📝 최종 답변 노드 실행")
        
        agent_results = state.get("agent_results", [])
        references = state.get("references", [])
        actions = state.get("actions", [])
        final_answer = state.get("final_answer")
        handsoff_decision = state.get("handsoff_decision", {})
        
        # 답변 생성
        if final_answer:
            answer = final_answer
        elif handsoff_decision.get("decision") == "forward":
            answer = handsoff_decision.get("answer", "")
        elif len(agent_results) == 1:
            answer = agent_results[0].get("answer", agent_results[0].get("message", ""))
        else:
            answer = "여러 에이전트의 결과를 종합하여 답변을 생성했습니다."
        
        # Reference와 Action을 객체로 변환
        reference_objects = [Reference(**ref) if isinstance(ref, dict) else ref for ref in references]
        action_objects = [Action(**act) if isinstance(act, dict) else act for act in actions]
        
        additional_info = AdditionalInfo(
            answer=answer,
            reference=reference_objects,
            action=action_objects
        )
        
        return {
            "final_answer": answer,
            "additional_info": additional_info.to_dict() if hasattr(additional_info, 'to_dict') else {
                "answer": additional_info.answer,
                "reference": [ref.__dict__ if hasattr(ref, '__dict__') else ref for ref in additional_info.reference],
                "action": [act.__dict__ if hasattr(act, '__dict__') else act for act in additional_info.action]
            },
            "status": "success"
        }
    
    def _should_clarify(self, state: SupervisorState) -> Literal["clarify", "continue"]:
        """재질문이 필요한지 확인"""
        return "clarify" if state.get("needs_clarification", False) else "continue"
    
    def _should_handsoff(self, state: SupervisorState) -> Literal["handsoff", "continue"]:
        """Hands-off가 필요한지 확인"""
        final_answer = state.get("final_answer")
        if final_answer:
            return "continue"
        return "handsoff"
    
    def _handsoff_decision(self, state: SupervisorState) -> Literal["forward", "continue"]:
        """Hands-off 결정에 따른 라우팅"""
        handsoff_decision = state.get("handsoff_decision", {})
        decision = handsoff_decision.get("decision", "handsoff")
        return "forward" if decision == "forward" else "continue"
    
    async def route_task(
        self,
        user_query: str,
        user_id: str,
        date: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        사용자 질문을 분석하여 적절한 에이전트에게 라우팅 (LangGraph 사용)
        
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
        
        # LangGraph 사용 가능하면 그래프 실행
        if LANGGRAPH_AVAILABLE and self.graph:
            try:
                # 초기 State 구성
                initial_state: SupervisorState = {
                    "user_query": user_query,
                    "user_id": user_id,
                    "date": date,
                    "agent_state": state,
                    "conversation_history": state.get_recent_history(limit=10),
                    "routing_decision": None,
                    "collected_data": {},
                    "references": [],
                    "actions": [],
                    "current_agent": None,
                    "agent_results": [],
                    "needs_clarification": False,
                    "clarification_question": None,
                    "final_answer": None,
                    "handsoff_decision": None,
                    "additional_info": None,
                    "status": "processing"
                }
                
                # 그래프 실행 (Checkpointer를 사용하므로 thread_id 필요)
                # thread_id는 사용자별로 고유하게 생성 (user_id 기반)
                import hashlib
                thread_id = hashlib.md5(f"{user_id}_{date}".encode()).hexdigest()
                config = {"configurable": {"thread_id": thread_id}}
                final_state = await self.graph.ainvoke(initial_state, config=config)
                
                # 결과 추출
                if final_state.get("needs_clarification"):
                    # references와 actions를 객체로 변환
                    refs = final_state.get("references", [])
                    acts = final_state.get("actions", [])
                    reference_objects = [Reference(**ref) if isinstance(ref, dict) else ref for ref in refs]
                    action_objects = [Action(**act) if isinstance(act, dict) else act for act in acts]
                    
                    additional_info = AdditionalInfo(
                        answer=f"[재질문] {final_state.get('clarification_question', '')}",
                        reference=reference_objects,
                        action=action_objects
                    )
                    return {
                        "status": "needs_clarification",
                        "answer": final_state.get("clarification_question", ""),
                        "additional_info": additional_info,
                        "agent": "reask"
                    }
                
                additional_info_dict = final_state.get("additional_info")
                if additional_info_dict:
                    # dict에서 AdditionalInfo 객체로 변환
                    refs = additional_info_dict.get("reference", [])
                    acts = additional_info_dict.get("action", [])
                    reference_objects = [Reference(**ref) if isinstance(ref, dict) else ref for ref in refs]
                    action_objects = [Action(**act) if isinstance(act, dict) else act for act in acts]
                    additional_info = AdditionalInfo(
                        answer=additional_info_dict.get("answer", ""),
                        reference=reference_objects,
                        action=action_objects
                    )
                else:
                    # additional_info가 없으면 생성
                    refs = final_state.get("references", [])
                    acts = final_state.get("actions", [])
                    reference_objects = [Reference(**ref) if isinstance(ref, dict) else ref for ref in refs]
                    action_objects = [Action(**act) if isinstance(act, dict) else act for act in acts]
                    additional_info = AdditionalInfo(
                        answer=final_state.get("final_answer", ""),
                        reference=reference_objects,
                        action=action_objects
                    )
                
                return {
                    "status": final_state.get("status", "success"),
                    "answer": final_state.get("final_answer", ""),
                    "additional_info": additional_info,
                    "agent": final_state.get("current_agent", "multiple"),
                    "supervisor_decision": final_state.get("routing_decision")
                }
            except Exception as e:
                import traceback
                self.logger.error(
                    f"❌ LangGraph 실행 실패: {str(e)}",
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    },
                    exc_info=True
                )
                # Fallback to legacy mode
                return await self._route_task_legacy(user_query, user_id, date, state)
        else:
            # Legacy mode (기존 방식)
            return await self._route_task_legacy(user_query, user_id, date, state)
    
    async def _route_task_legacy(
        self,
        user_query: str,
        user_id: str,
        date: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """기존 방식의 라우팅 (LangGraph 미사용)"""
        self.logger.warning("⚠️ Legacy 모드로 실행 (LangGraph 미사용)")
        
        conversation_history = state.get_recent_history(limit=10)
        references: List[Reference] = []
        actions: List[Action] = []
        
        try:
            # 1. ReAsk
            reask_result = await self.agents["reask"].check_and_ask(
                user_query,
                context={"state": state.to_dict()} if state else None,
                conversation_history=conversation_history
            )
            
            if reask_result.get("needs_clarification", False):
                additional_info = AdditionalInfo(
                    answer=f"[재질문] {reask_result.get('clarification_question', '')}",
                    reference=references,
                    action=actions
                )
                return {
                    "status": "needs_clarification",
                    "answer": reask_result.get("clarification_question", ""),
                    "additional_info": additional_info,
                    "agent": "reask"
                }
            
            # 2. Routing
            routing_decision = await self._select_agent(user_query, conversation_history)
            routing = routing_decision.get("routing", [])
            
            # 3. Agent Execution
            collected_data = {}
            results = []
            for agent_name in routing:
                if agent_name in self.agents:
                    task = routing_decision.get("task_breakdown", {}).get(agent_name, user_query)
                    context = {
                        "state": state.to_dict(),
                        "collected_data": collected_data,
                        "user_id": user_id,
                        "date": date
                    }
                    result = await self.agents[agent_name].execute(task, context)
                    results.append(result)
                    
                    if "reference" in result:
                        if isinstance(result["reference"], list):
                            references.extend([Reference(**ref) for ref in result["reference"]])
                    
                    if "action" in result:
                        if isinstance(result["action"], list):
                            actions.extend([Action(**act) for act in result["action"]])
                    
                    if result.get("status") == "success":
                        collected_data[agent_name] = result
            
            # 4. Final Answer
            if len(results) == 1:
                answer = results[0].get("answer", results[0].get("message", ""))
            else:
                answer = "여러 에이전트의 결과를 종합하여 답변을 생성했습니다."
            
            additional_info = AdditionalInfo(
                answer=answer,
                reference=references,
                action=actions
            )
            
            return {
                "status": "success",
                "answer": answer,
                "additional_info": additional_info,
                "agent": results[0].get("agent", "multiple") if results else "unknown",
                "supervisor_decision": routing_decision
            }
        except Exception as e:
            self.logger.error(f"❌ Legacy 라우팅 실패: {str(e)}", exc_info=True)
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
        
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        messages.append({
            "role": "user",
            "content": SUPERVISOR_ROUTING_USER_PROMPT_TEMPLATE.format(user_query=user_query)
        })
        
        # LangChain OpenAI 사용 시
        if LANGCHAIN_OPENAI_AVAILABLE and hasattr(self.client, 'invoke'):
            # LangChain 내장 함수로 메시지 변환
            langchain_messages = convert_dict_messages_to_langchain(messages)
            
            # 모델 및 온도 설정
            if self.model != settings.openai_model:
                self.client.model_name = self.model
            self.client.temperature = 0.3
            
            # LangChain 호출 (LangSmith 자동 추적)
            # JSON 형식 응답을 위한 구조화된 출력 사용
            try:
                # LangChain의 with_structured_output 사용 (내장 함수)
                if hasattr(self.client, 'with_structured_output'):
                    # 구조화된 출력을 dict로 받기
                    structured_llm = self.client.with_structured_output(dict)
                    response = structured_llm.invoke(langchain_messages)
                    content = json.dumps(response) if isinstance(response, dict) else str(response)
                else:
                    # Fallback: 일반 호출 후 JSON 파싱
                    response = self.client.invoke(langchain_messages)
                    content = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                self.logger.warning(
                    f"⚠️  구조화된 출력 실패, 일반 호출 사용",
                    {"error": str(e)}
                )
                # 최종 Fallback: 일반 호출
                response = self.client.invoke(langchain_messages)
                content = response.content if hasattr(response, 'content') else str(response)
        else:
            # OpenAI SDK 직접 사용
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
        
        result = json.loads(content)
        return result
    
    def get_available_agents(self) -> Dict[str, Any]:
        """사용 가능한 에이전트 목록 반환"""
        return {
            name: agent.get_capabilities()
            for name, agent in self.agents.items()
        }
