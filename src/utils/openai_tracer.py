"""OpenAI SDK 호출을 LangSmith로 추적하는 래퍼"""
import os
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger("openai-tracer")

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

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    # Fallback: OpenAI SDK 직접 사용
    from openai import OpenAI


class TracedOpenAIClient:
    """LangSmith 추적이 포함된 OpenAI 클라이언트 래퍼"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logger
        
        # LangChain OpenAI 사용 (LangSmith 자동 추적)
        if LANGCHAIN_OPENAI_AVAILABLE and settings.langsmith_tracing and settings.langsmith_api_key:
            self.use_langchain = True
            self.langchain_client = ChatOpenAI(
                model=settings.openai_model,
                api_key=api_key,
                temperature=0.7
            )
            logger.info(f"✅ LangSmith 추적 활성화됨 (langchain-openai 사용, 프로젝트: {settings.langsmith_project})")
        else:
            # Fallback: OpenAI SDK 직접 사용
            self.use_langchain = False
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            if not LANGCHAIN_OPENAI_AVAILABLE:
                logger.warning("⚠️  langchain-openai가 설치되지 않아 LangSmith 추적이 제한될 수 있습니다")
            else:
                logger.warning("⚠️  LangSmith 추적 비활성화됨 (OpenAI SDK 직접 사용)")
    
    def chat_completions_create(self, model: str, messages: List[Dict[str, str]], **kwargs):
        """Chat completions 생성 (LangSmith 추적 포함)"""
        call_info = self.logger.trace_openai_call(model, messages, **kwargs)
        
        try:
            if self.use_langchain:
                # LangChain OpenAI 사용 (LangSmith 자동 추적)
                # LangChain 메시지 형식으로 변환
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
                
                langchain_messages = []
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        langchain_messages.append(SystemMessage(content=content))
                    elif role == "user":
                        langchain_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        langchain_messages.append(AIMessage(content=content))
                
                # 모델 설정
                if model != settings.openai_model:
                    self.langchain_client.model_name = model
                
                # 온도 설정
                temperature = kwargs.get("temperature", 0.7)
                self.langchain_client.temperature = temperature
                
                # 응답 생성
                response = self.langchain_client.invoke(langchain_messages)
                
                # OpenAI SDK 형식으로 변환
                class MockResponse:
                    def __init__(self, content):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {'content': content})()
                        })()]
                
                return MockResponse(response.content)
            else:
                # OpenAI SDK 직접 사용 (LangSmith 추적 없음)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
            
            self.logger.log_openai_response(call_info, response)
            return response
        except Exception as e:
            self.logger.log_openai_response(call_info, None, error=e)
            raise
    
    def __getattr__(self, name):
        """다른 메서드는 원본 클라이언트로 위임"""
        if self.use_langchain:
            return getattr(self.langchain_client, name)
        else:
            return getattr(self.client, name)

