"""기본 에이전트 클래스"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from openai import OpenAI
from src.utils.settings import settings
from src.utils.logger import get_logger
from src.utils.openai_tracer import TracedOpenAIClient

# LangSmith tracing 설정
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"


class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = get_logger(f"agent-{name}")
        # LangSmith 추적이 포함된 OpenAI 클라이언트 사용
        self.client = TracedOpenAIClient(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.logger.info(f"🤖 에이전트 초기화: {name}", {"description": description})
    
    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        에이전트의 메인 실행 메서드
        
        Args:
            task: 수행할 작업 설명
            context: 추가 컨텍스트 정보
            
        Returns:
            실행 결과 딕셔너리
        """
        pass
    
    async def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """LLM 호출 헬퍼 메서드"""
        self.logger.debug(
            f"📤 LLM 호출 시작",
            {
                "model": self.model,
                "messages_count": len(messages),
                "temperature": temperature
            }
        )
        
        try:
            response = self.client.chat_completions_create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content
            self.logger.debug(
                f"📥 LLM 응답 수신",
                {
                    "model": self.model,
                    "response_length": len(content) if content else 0
                }
            )
            return content
        except Exception as e:
            self.logger.error(
                f"❌ LLM 호출 실패",
                {
                    "model": self.model,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
    
    def get_capabilities(self) -> Dict[str, Any]:
        """에이전트의 능력 설명 반환"""
        return {
            "name": self.name,
            "description": self.description
        }

