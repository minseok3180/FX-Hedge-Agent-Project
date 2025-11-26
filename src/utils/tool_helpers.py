"""Tool 헬퍼 함수 - Tool 결과를 Command로 변환"""
from typing import Dict, Any, Optional, List, Literal
from src.utils.logger import get_logger
from src.utils.state import create_reference_and_action_from_tool_result, Reference, Action

logger = get_logger("tool-helpers")

try:
    from langgraph.types import Command
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    Command = None


def create_command_from_tool_result(
    tool_name: str,
    tool_result: Any,
    source: str,
    update_state: Optional[Dict[str, Any]] = None,
    goto: Optional[str] = None,
    query: Optional[str] = None,
    input_params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Tool 실행 결과를 Command로 변환하여 state 업데이트 및 노드 이동
    
    Args:
        tool_name: Tool 이름
        tool_result: Tool 실행 결과
        source: 참고 자료 소스 ("rdb", "vdb", "web_search" 등)
        update_state: 추가로 업데이트할 state (선택사항)
        goto: 이동할 노드 이름 (선택사항, None이면 현재 노드 유지)
        query: 실행한 쿼리 또는 검색어
        input_params: 입력 파라미터
        metadata: 추가 메타데이터
        
    Returns:
        Command 객체 (LangGraph 사용 시) 또는 일반 dict (Fallback)
    """
    if not LANGGRAPH_AVAILABLE or Command is None:
        # Fallback: 일반 dict 반환
        logger.warning("⚠️  LangGraph Command를 사용할 수 없어 일반 dict를 반환합니다.")
        reference, action = create_reference_and_action_from_tool_result(
            tool_name, tool_result, source, query, input_params, metadata
        )
        result = {
            "tool_result": tool_result,
            "reference": reference.__dict__,
            "action": action.__dict__
        }
        if update_state:
            result.update(update_state)
        return result
    
    # Reference와 Action 생성
    reference, action = create_reference_and_action_from_tool_result(
        tool_name, tool_result, source, query, input_params, metadata
    )
    
    # State 업데이트 딕셔너리 구성
    state_update = {
        "references": [reference.__dict__],  # reducer가 자동으로 append
        "actions": [action.__dict__]  # reducer가 자동으로 append
    }
    
    # Tool 결과를 collected_data에 추가
    if update_state:
        state_update.update(update_state)
    else:
        # 기본적으로 tool 결과를 collected_data에 저장
        if "collected_data" not in state_update:
            state_update["collected_data"] = {}
        state_update["collected_data"][tool_name] = tool_result
    
    # Command 생성
    if goto:
        return Command(
            goto=goto,
            update=state_update
        )
    else:
        # goto가 없으면 update만 수행 (현재 노드 유지)
        return Command(
            update=state_update
        )

