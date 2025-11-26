"""에이전트 기본 클래스"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.utils.settings import settings
from src.utils.logger import get_logger
from src.utils.llm import LANGCHAIN_OPENAI_AVAILABLE, convert_dict_messages_to_langchain

# LangSmith tracing은 llm 모듈에서 환경 변수로 설정됨


class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = get_logger(f"agent-{name}")
        self.model = settings.openai_model
        
        # LangChain OpenAI 클라이언트 사용 (LangSmith 자동 추적)
        if LANGCHAIN_OPENAI_AVAILABLE:
            from langchain_openai import ChatOpenAI
            self.client = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.7
            )
        else:
            # Fallback: OpenAI SDK 직접 사용
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
        
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
            # LangChain OpenAI 사용 시
            if LANGCHAIN_OPENAI_AVAILABLE and hasattr(self.client, 'invoke'):
                # LangChain 내장 함수로 메시지 변환
                langchain_messages = convert_dict_messages_to_langchain(messages)
                
                # 모델 및 온도 설정
                if self.model != settings.openai_model:
                    self.client.model_name = self.model
                self.client.temperature = temperature
                
                # LangChain 호출 (LangSmith 자동 추적)
                response = self.client.invoke(langchain_messages)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                # OpenAI SDK 직접 사용
                response = self.client.chat.completions.create(
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
    
    def bind_tools(self, tools: Optional[List[Any]] = None) -> Any:
        """
        LangChain LLM에 tool을 바인딩 (function calling 지원)
        
        Args:
            tools: 바인딩할 tool 리스트 (None이면 모든 tool 사용)
            
        Returns:
            Tool이 바인딩된 LLM 인스턴스
        """
        if LANGCHAIN_OPENAI_AVAILABLE and hasattr(self.client, 'bind_tools'):
            from src.utils.tools import get_all_tools, bind_tools_to_llm
            if tools is None:
                tools = get_all_tools()
            return bind_tools_to_llm(self.client, tools)
        else:
            self.logger.warning("⚠️  Tool 바인딩을 지원하지 않습니다. (LangChain OpenAI 필요)")
            return self.client

