"""멀티 에이전트 시스템 State 관리"""
from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from src.utils.logger import get_logger

logger = get_logger("state-manager")


@dataclass
class Reference:
    """참고 자료 정보"""
    source: str  # "rdb", "vdb", "web_search" 등
    query: Optional[str] = None  # 실행한 쿼리 또는 검색어
    results_count: int = 0  # 결과 개수
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class Action:
    """에이전트가 수행한 액션 정보"""
    type: str  # "modify", "calculate", "query" 등
    tool: str  # 사용한 툴 이름
    description: str  # 액션 설명
    input: Dict[str, Any] = field(default_factory=dict)  # 입력 파라미터
    output: Dict[str, Any] = field(default_factory=dict)  # 출력 결과
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def create_reference_and_action_from_tool_result(
    tool_name: str,
    tool_result: Any,
    source: str,
    query: Optional[str] = None,
    input_params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> tuple[Reference, Action]:
    """
    툴 실행 결과로부터 Reference와 Action 생성 (헬퍼 함수)
    
    Args:
        tool_name: 툴 이름 (예: "web_search", "rdb_hard")
        tool_result: 툴 실행 결과
        source: 참고 자료 소스 ("rdb", "vdb", "web_search" 등)
        query: 실행한 쿼리 또는 검색어
        input_params: 입력 파라미터
        metadata: 추가 메타데이터
        
    Returns:
        (Reference, Action) 튜플
    """
    # 결과 개수 계산
    if isinstance(tool_result, list):
        results_count = len(tool_result)
    elif isinstance(tool_result, dict):
        # results_count 추출
        if "results_count" in tool_result:
            results_count = tool_result["results_count"]
        elif "results" in tool_result and isinstance(tool_result["results"], list):
            results_count = len(tool_result["results"])
        elif "data" in tool_result and isinstance(tool_result["data"], list):
            results_count = len(tool_result["data"])
        else:
            results_count = 0
    else:
        results_count = 1 if tool_result else 0
    
    # Reference 생성
    reference = Reference(
        source=source,
        query=query,
        results_count=results_count,
        metadata=metadata or {}
    )
    
    # Action 생성
    action = Action(
        type="query" if source in ["rdb", "vdb"] else "search" if source == "web_search" else "execute",
        tool=tool_name,
        description=f"{tool_name} 실행: {query or tool_name}",
        input=input_params or {},
        output={"results_count": results_count}
    )
    
    return reference, action

@dataclass
class AdditionalInfo:
    """추가 정보 (API 출력물 형태)"""
    answer: str  # 결과 답변
    reference: List[Reference] = field(default_factory=list)  # 참고 자료
    action: List[Action] = field(default_factory=list)  # 수행한 액션들


@dataclass
class ConversationTurn:
    """대화 턴 (유저 질의 - 에이전트 답변 쌍)"""
    user_message: str  # 사용자 질의
    date: str  # 질문하는 날짜 (YYYY-MM-DD)
    user_id: str  # 사용자 아이디
    agent_answer: str  # 에이전트 답변
    additional_info: AdditionalInfo  # 추가 정보
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentState:
    """멀티 에이전트 시스템의 State"""
    user_id: str
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(
        self,
        user_message: str,
        date: str,
        agent_answer: str,
        additional_info: AdditionalInfo
    ):
        """대화 턴 추가"""
        turn = ConversationTurn(
            user_message=user_message,
            date=date,
            user_id=self.user_id,
            agent_answer=agent_answer,
            additional_info=additional_info
        )
        self.conversation_history.append(turn)
        logger.debug(
            f"대화 턴 추가",
            {"user_id": self.user_id, "turn_count": len(self.conversation_history)}
        )
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """최근 대화 이력 반환 (LLM용 형식)"""
        recent_turns = self.conversation_history[-limit:]
        history = []
        for turn in recent_turns:
            history.append({
                "role": "user",
                "content": turn.user_message
            })
            history.append({
                "role": "assistant",
                "content": turn.agent_answer
            })
        return history
    
    def to_dict(self) -> Dict[str, Any]:
        """State를 딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """딕셔너리에서 State 생성"""
        state = cls(user_id=data["user_id"])
        
        for turn_data in data.get("conversation_history", []):
            # Reference 복원
            references = [
                Reference(**ref_data)
                for ref_data in turn_data["additional_info"]["reference"]
            ]
            
            # Action 복원
            actions = [
                Action(**action_data)
                for action_data in turn_data["additional_info"]["action"]
            ]
            
            # AdditionalInfo 복원
            additional_info = AdditionalInfo(
                answer=turn_data["additional_info"]["answer"],
                reference=references,
                action=actions
            )
            
            # ConversationTurn 복원
            turn = ConversationTurn(
                user_message=turn_data["user_message"],
                date=turn_data["date"],
                user_id=turn_data["user_id"],
                agent_answer=turn_data["agent_answer"],
                additional_info=additional_info,
                timestamp=turn_data.get("timestamp", "")
            )
            
            state.conversation_history.append(turn)
        
        state.current_context = data.get("current_context", {})
        return state


class StateManager:
    """State 관리자 (세션별 State 저장)"""
    
    def __init__(self):
        self.states: Dict[str, AgentState] = {}
        logger.info("StateManager 초기화")
    
    def get_state(self, user_id: str) -> AgentState:
        """사용자의 State 조회 (없으면 생성)"""
        if user_id not in self.states:
            self.states[user_id] = AgentState(user_id=user_id)
            logger.debug(f"새 State 생성", {"user_id": user_id})
        return self.states[user_id]
    
    def save_state(self, state: AgentState):
        """State 저장"""
        self.states[state.user_id] = state
        logger.debug(f"State 저장", {"user_id": state.user_id})
    
    def clear_state(self, user_id: str):
        """State 삭제"""
        if user_id in self.states:
            del self.states[user_id]
            logger.debug(f"State 삭제", {"user_id": user_id})
    
    def get_all_states(self) -> Dict[str, AgentState]:
        """모든 State 조회"""
        return self.states

