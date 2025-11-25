"""GPT API 호출 유틸리티 함수"""
from typing import List, Dict, Any, Optional
from src.utils.settings import settings
from src.utils.openai_tracer import TracedOpenAIClient
from src.utils.logger import get_logger

logger = get_logger("gpt-client")

# 전역 클라이언트 인스턴스 (재사용)
_client_instance: Optional[TracedOpenAIClient] = None


def _get_client() -> TracedOpenAIClient:
    """TracedOpenAIClient 인스턴스 반환 (싱글톤 패턴)"""
    global _client_instance
    if _client_instance is None:
        _client_instance = TracedOpenAIClient(api_key=settings.openai_api_key)
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
        
        # API 호출 파라미터 구성
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if response_format:
            api_params["response_format"] = response_format
        
        # API 호출
        response = client.chat_completions_create(**api_params)
        
        # 응답 추출
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

