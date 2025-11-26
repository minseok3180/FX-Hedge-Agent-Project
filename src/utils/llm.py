"""LLM 관련 유틸리티 (LangChain, OpenAI, LangSmith 통합)"""
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from src.utils.settings import settings
from src.utils.logger import get_logger

logger = get_logger("llm-utils")

# ============================================================================
# LangSmith 추적 설정
# ============================================================================

# LangSmith 추적을 위한 환경 변수 설정
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"
    # LangChain 환경 변수 (langchain-openai가 사용)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logger.info(f"✅ LangSmith 추적 활성화됨 (프로젝트: {settings.langsmith_project})")
else:
    logger.warning("⚠️  LangSmith 추적 비활성화됨 (설정 확인 필요)")

# LangChain OpenAI 사용 가능 여부 확인
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    logger.warning("⚠️  langchain-openai가 설치되지 않아 LangSmith 추적이 제한될 수 있습니다")

# ============================================================================
# LangChain 메시지 변환 헬퍼
# ============================================================================

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

try:
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        SystemMessage,
        AIMessage,
        ChatMessage,
        convert_to_messages
    )
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False
    BaseMessage = Any
    convert_to_messages = None
    # Fallback 메시지 클래스
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content
    class AIMessage:
        def __init__(self, content: str):
            self.content = content
    class ChatMessage:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content


def _convert_message_manual(msg: Dict[str, str]) -> Any:
    """메시지 수동 변환 (중복 제거를 위한 헬퍼 함수)"""
    role = msg.get("role")
    content = msg.get("content", "")
    
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        return ChatMessage(role=role, content=content)


def convert_dict_messages_to_langchain(
    messages: List[Dict[str, str]]
) -> List[BaseMessage]:
    """
    딕셔너리 형식의 메시지를 LangChain 메시지로 변환 (내장 함수 사용)
    
    Args:
        messages: 딕셔너리 형식의 메시지 리스트 [{"role": "user", "content": "..."}]
        
    Returns:
        LangChain BaseMessage 리스트
    """
    if not LANGCHAIN_CORE_AVAILABLE:
        # Fallback: 수동 변환
        return [_convert_message_manual(msg) for msg in messages]
    
    try:
        # LangChain의 convert_to_messages 사용 (더 안전하고 유연함)
        # OpenAI 형식의 메시지를 자동으로 변환
        return convert_to_messages(messages)
    except Exception as e:
        logger.warning(
            f"⚠️  convert_to_messages 실패, 수동 변환 사용",
            {"error": str(e)}
        )
        # Fallback: 수동 변환
        return [_convert_message_manual(msg) for msg in messages]

# ============================================================================
# GPT API 호출
# ============================================================================

# LangChain OpenAI 클라이언트 (LangSmith 자동 추적)
_client_instance: Optional[Any] = None


def _get_client():
    """ChatOpenAI 인스턴스 반환 (싱글톤 패턴, LangSmith 자동 추적)"""
    global _client_instance
    if _client_instance is None:
        if LANGCHAIN_OPENAI_AVAILABLE:
            from langchain_openai import ChatOpenAI
            _client_instance = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.7
            )
        else:
            # Fallback: OpenAI SDK 직접 사용
            from openai import OpenAI
            _client_instance = OpenAI(api_key=settings.openai_api_key)
    return _client_instance


async def call_gpt(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    response_format: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    GPT API에 요청을 보내 답변을 받는 함수
    
    Args:
        messages: 대화 메시지 리스트 (예: [{"role": "user", "content": "안녕하세요"}])
        model: 사용할 모델명 (None이면 settings의 기본 모델 사용)
        temperature: 생성 온도 (0.0 ~ 2.0, 기본값: 0.7)
        response_format: 응답 형식 (예: {"type": "json_object"})
        **kwargs: 기타 OpenAI API 파라미터
        
    Returns:
        GPT 응답 텍스트
        
    Raises:
        Exception: API 호출 실패 시
    """
    if model is None:
        model = settings.openai_model
    
    logger.debug(
        f"📤 GPT API 호출 시작",
        {
            "model": model,
            "messages_count": len(messages),
            "temperature": temperature,
            "has_response_format": response_format is not None
        }
    )
    
    try:
        client = _get_client()
        
        # LangChain OpenAI 사용 시
        if LANGCHAIN_OPENAI_AVAILABLE and hasattr(client, 'invoke'):
            # LangChain 내장 함수로 메시지 변환
            langchain_messages = convert_dict_messages_to_langchain(messages)
            
            # 모델 및 온도 설정
            if model != settings.openai_model:
                client.model_name = model
            client.temperature = temperature
            
            # LangChain 호출 (LangSmith 자동 추적)
            response = client.invoke(langchain_messages)
            content = response.content if hasattr(response, 'content') else str(response)
        else:
            # OpenAI SDK 직접 사용
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }
            
            if response_format:
                api_params["response_format"] = response_format
            
            response = client.chat.completions.create(**api_params)
            content = response.choices[0].message.content
        
        logger.debug(
            f"📥 GPT API 응답 수신",
            {
                "model": model,
                "response_length": len(content) if content else 0
            }
        )
        
        return content if content else ""
        
    except Exception as e:
        logger.error(
            f"❌ GPT API 호출 실패",
            {
                "model": model,
                "error": str(e)
            },
            exc_info=True
        )
        raise

