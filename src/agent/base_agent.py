"""
Agent 기본 클래스 및 공통 기능
모든 에이전트의 기본 구조와 공통 기능을 정의
Mi:dm 2.0 LLM 클라이언트 통합
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import asyncio

from config import Config
from supervisor import Task
from local_llm_client import LocalMidm2Client, ChatMessage


@dataclass
class ToolResult:
    """도구 실행 결과 클래스"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스 (Mi:dm 2.0 통합)"""
    
    def __init__(self, config: Config, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        self.tools: Dict[str, callable] = {}
        self.llm_client: Optional[Midm2Client] = None
        
        # TODO: 에이전트 초기화 로직
        # - Mi:dm 2.0 LLM 클라이언트 초기화
        # - 데이터베이스 연결 설정
        # - 도구 등록
        self._initialize_agent()
    
    def _initialize_agent(self):
        """에이전트 초기화"""
        # TODO: 에이전트별 초기화 로직 구현
        # - Mi:dm 2.0 클라이언트 설정
        # - 도구 등록
        # - 리소스 할당
        # - 상태 초기화
        pass
    
    async def _get_llm_client(self) -> LocalMidm2Client:
        """LLM 클라이언트 인스턴스 반환"""
        if self.llm_client is None:
            self.llm_client = LocalMidm2Client(self.config)
        return self.llm_client
    
    @abstractmethod
    async def execute_task(self, task: Task) -> ToolResult:
        """
        작업 실행 (추상 메서드)
        
        Args:
            task: 실행할 작업
            
        Returns:
            실행 결과
        """
        pass
    
    def register_tool(self, tool_name: str, tool_function: callable):
        """
        도구 등록
        
        Args:
            tool_name: 도구 이름
            tool_function: 도구 함수
        """
        self.tools[tool_name] = tool_function
        self.logger.info(f"도구 '{tool_name}' 등록 완료")
    
    def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록 반환"""
        return list(self.tools.keys())
    
    async def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        도구 호출
        
        Args:
            tool_name: 호출할 도구 이름
            **kwargs: 도구에 전달할 인자
            
        Returns:
            도구 실행 결과
        """
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                error=f"도구 '{tool_name}'를 찾을 수 없습니다"
            )
        
        try:
            tool_function = self.tools[tool_name]
            result = await tool_function(**kwargs)
            return ToolResult(success=True, data=result)
        except Exception as e:
            self.logger.error(f"도구 '{tool_name}' 실행 중 오류: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    async def analyze_with_llm(self, 
                              system_prompt: str,
                              user_prompt: str,
                              temperature: float = 0.7) -> str:
        """
        Mi:dm 2.0을 사용한 텍스트 분석
        
        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            temperature: 창의성 조절
            
        Returns:
            LLM 분석 결과
        """
        try:
            async with await self._get_llm_client() as client:
                messages = [
                    client.create_system_message(system_prompt),
                    client.create_user_message(user_prompt)
                ]
                
                response = await client.chat_completion(messages, temperature=temperature)
                return response.content
                
        except Exception as e:
            self.logger.error(f"LLM 분석 실패: {str(e)}")
            raise e
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        텍스트 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        try:
            async with await self._get_llm_client() as client:
                return await client.create_embedding(text)
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {str(e)}")
            raise e
    
    async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        배치 임베딩 생성
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        try:
            async with await self._get_llm_client() as client:
                return await client.batch_create_embeddings(texts)
        except Exception as e:
            self.logger.error(f"배치 임베딩 생성 실패: {str(e)}")
            raise e
    
    def validate_task_parameters(self, task: Task) -> bool:
        """
        작업 파라미터 검증
        
        Args:
            task: 검증할 작업
            
        Returns:
            검증 성공 여부
        """
        # TODO: 작업 파라미터 검증 로직
        # - 필수 파라미터 존재 확인
        # - 파라미터 타입 검증
        # - 값 범위 검증
        return True
    
    def get_agent_status(self) -> Dict[str, Any]:
        """에이전트 상태 반환"""
        return {
            "name": self.agent_name,
            "status": "active",
            "available_tools": self.get_available_tools(),
            "llm_model": self.config.llm.model_name,
            "last_activity": "now"
        }
