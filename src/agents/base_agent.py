"""기본 에이전트 클래스"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from openai import OpenAI
from src.config.settings import settings


class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def get_capabilities(self) -> Dict[str, Any]:
        """에이전트의 능력 설명 반환"""
        return {
            "name": self.name,
            "description": self.description
        }

