"""LangChain Runnable 및 LangSmith 추적 헬퍼"""
from typing import List, Dict, Any, Optional, Callable, Awaitable, Union
from src.utils.logger import get_logger

logger = get_logger("runnable-helpers")

# LangChain Runnable 사용 가능 여부 확인
try:
    from langchain_core.runnables import (
        Runnable,
        RunnableLambda,
        RunnablePassthrough,
        RunnableSequence
    )
    from langchain_core.runnables.utils import Input, Output
    LANGCHAIN_RUNNABLE_AVAILABLE = True
except ImportError:
    LANGCHAIN_RUNNABLE_AVAILABLE = False
    Runnable = None
    RunnableLambda = None
    RunnablePassthrough = None
    RunnableSequence = None
    Input = Any
    Output = Any
    logger.warning("⚠️  langchain_core.runnables가 설치되지 않아 Runnable 기능이 제한됩니다")

# LangSmith 추적 강화
try:
    from langsmith import traceable, run_helpers
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    traceable = None
    run_helpers = None
    logger.warning("⚠️  langsmith가 설치되지 않아 고급 추적 기능이 제한됩니다")


def create_tool_call_chain(
    tool_name: str,
    tool_func: Union[Callable, Awaitable],
    input_transformer: Optional[Callable] = None,
    output_transformer: Optional[Callable] = None
) -> Optional[Runnable]:
    """
    Tool 호출을 위한 LangChain Runnable 체인 생성 (LangSmith 추적 포함)
    
    Args:
        tool_name: Tool 이름
        tool_func: Tool 함수 (동기 또는 비동기)
        input_transformer: 입력 변환 함수 (선택사항)
        output_transformer: 출력 변환 함수 (선택사항)
        
    Returns:
        Runnable 체인 또는 None (Runnable 사용 불가 시)
    """
    if not LANGCHAIN_RUNNABLE_AVAILABLE:
        logger.warning(f"⚠️  Runnable을 사용할 수 없어 {tool_name}는 일반 함수로 실행됩니다")
        return None
    
    def _log_tool_call(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tool 호출 로깅"""
        logger.info(
            f"🔧 [RUNNABLE CHAIN] {tool_name} 호출",
            {
                "tool_name": tool_name,
                "input": input_data
            }
        )
        return input_data
    
    def _log_tool_result(output_data: Any) -> Any:
        """Tool 결과 로깅"""
        logger.info(
            f"✅ [RUNNABLE CHAIN] {tool_name} 완료",
            {
                "tool_name": tool_name,
                "output_type": type(output_data).__name__,
                "output_preview": str(output_data)[:200] if output_data else None
            }
        )
        return output_data
    
    # Runnable 체인 구성
    chain = RunnablePassthrough()
    
    # 입력 로깅
    chain = chain | RunnableLambda(_log_tool_call)
    
    # 입력 변환 (있는 경우)
    if input_transformer:
        chain = chain | RunnableLambda(input_transformer)
    
    # Tool 실행 (LangSmith 추적 포함)
    # 비동기 함수는 LangChain Runnable이 자동으로 처리
    def _tool_call(input_data: Dict[str, Any]) -> Any:
        import asyncio
        if asyncio.iscoroutinefunction(tool_func):
            # 비동기 함수는 코루틴 반환 (Runnable이 await 처리)
            return tool_func(**input_data)
        else:
            return tool_func(**input_data)
    
    # LangSmith traceable 적용 (가능한 경우)
    if LANGSMITH_AVAILABLE and traceable:
        _tool_call = traceable(name=f"tool_{tool_name}", run_type="tool")(_tool_call)
    
    chain = chain | RunnableLambda(_tool_call)
    
    # 출력 변환 (있는 경우)
    if output_transformer:
        chain = chain | RunnableLambda(output_transformer)
    
    # 출력 로깅
    chain = chain | RunnableLambda(_log_tool_result)
    
    return chain


def create_tool_sequence(tools: List[Dict[str, Any]]) -> Optional[Runnable]:
    """
    여러 Tool을 순차적으로 실행하는 Runnable Sequence 생성
    
    Args:
        tools: Tool 정보 리스트 [{"name": "tool_name", "func": tool_func, "input": {...}}]
        
    Returns:
        Runnable Sequence 또는 None
    """
    if not LANGCHAIN_RUNNABLE_AVAILABLE:
        logger.warning("⚠️  Runnable을 사용할 수 없어 Tool Sequence를 생성할 수 없습니다")
        return None
    
    if not tools:
        return None
    
    # 첫 번째 tool로 시작
    first_tool = tools[0]
    chain = create_tool_call_chain(
        first_tool["name"],
        first_tool["func"]
    )
    
    if chain is None:
        return None
    
    # 나머지 tool들을 체인에 추가
    for tool in tools[1:]:
        next_chain = create_tool_call_chain(
            tool["name"],
            tool["func"]
        )
        if next_chain:
            chain = chain | next_chain
    
    return chain

