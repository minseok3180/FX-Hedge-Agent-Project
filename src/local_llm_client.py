"""
로컬 LLM 클라이언트 모듈
로컬에 다운로드된 Mi:dm 2.0 모델과 상호작용하는 클라이언트 구현
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch

from config import Config, LLMConfig


@dataclass
class ChatMessage:
    """채팅 메시지 클래스"""
    role: str  # system, user, assistant
    content: str
    timestamp: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM 응답 클래스"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float
    timestamp: str


class LocalMidm2Client:
    """로컬 Mi:dm 2.0 클라이언트"""
    
    def __init__(self, config: Config):
        self.config = config.llm
        self.logger = logging.getLogger(__name__)
        
        # 모델 및 토크나이저 초기화
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.text_generator = None
        
        # TODO: 로컬 모델 초기화
        self._initialize_models()
    
    def _initialize_models(self):
        """로컬 모델들 초기화"""
        try:
            self.logger.info("로컬 모델 초기화 시작...")
            
            # 디바이스 설정
            device = self._get_device()
            self.logger.info(f"사용 디바이스: {device}")
            
            # 메인 LLM 모델 로드
            if self.config.use_local:
                self._load_main_model(device)
            
            # 임베딩 모델 로드
            self._load_embedding_model(device)
            
            self.logger.info("로컬 모델 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"로컬 모델 초기화 실패: {str(e)}")
            raise e
    
    def _get_device(self):
        """사용할 디바이스 결정"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def _load_main_model(self, device):
        """메인 LLM 모델 로드 (transformers 라이브러리 사용)"""
        try:
            model_name = self.config.model_name
            
            self.logger.info(f"transformers를 통해 모델 로드 중: {model_name}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True
            )
            
            if device != "cuda":
                self.model = self.model.to(device)
            
            # 텍스트 생성 파이프라인 생성
            self.text_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                max_length=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.logger.info(f"메인 모델 로드 완료: {model_name}")
            
        except Exception as e:
            self.logger.error(f"메인 모델 로드 실패: {str(e)}")
            raise e
    
    def _load_embedding_model(self, device):
        """임베딩 모델 로드 (transformers 라이브러리 사용)"""
        try:
            embedding_model_name = self.config.embedding_model
            
            self.logger.info(f"transformers를 통해 임베딩 모델 로드 중: {embedding_model_name}")
            
            self.embedding_model = SentenceTransformer(
                embedding_model_name,
                device=device
            )
            
            self.logger.info(f"임베딩 모델 로드 완료: {embedding_model_name}")
            
        except Exception as e:
            self.logger.error(f"임베딩 모델 로드 실패: {str(e)}")
            raise e
    
    async def chat_completion(self,
                            messages: List[ChatMessage],
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None,
                            top_p: Optional[float] = None,
                            stream: bool = False) -> LLMResponse:
        """
        채팅 완성 요청 (로컬 모델)
        
        Args:
            messages: 대화 메시지 리스트
            temperature: 창의성 조절
            max_tokens: 최대 토큰 수
            top_p: 핵샘플링 파라미터
            stream: 스트리밍 여부 (로컬에서는 미지원)
            
        Returns:
            LLM 응답
        """
        if not self.text_generator:
            raise Exception("로컬 모델이 초기화되지 않았습니다")
        
        start_time = time.time()
        
        try:
            # TODO: 메시지를 프롬프트로 변환
            prompt = self._messages_to_prompt(messages)
            
            # TODO: 텍스트 생성
            response = self.text_generator(
                prompt,
                max_length=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                top_p=top_p or self.config.top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # TODO: 응답 파싱
            generated_text = response[0]['generated_text']
            content = generated_text[len(prompt):].strip()
            
            # TODO: 토큰 사용량 계산
            input_tokens = len(self.tokenizer.encode(prompt))
            output_tokens = len(self.tokenizer.encode(content))
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                finish_reason="stop",
                response_time=response_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"채팅 완성 요청 실패: {str(e)}")
            raise e
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """메시지 리스트를 프롬프트로 변환"""
        # TODO: Mi:dm 2.0 모델에 맞는 프롬프트 형식으로 변환
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"<system>\n{message.content}\n</system>")
            elif message.role == "user":
                prompt_parts.append(f"<user>\n{message.content}\n</user>")
            elif message.role == "assistant":
                prompt_parts.append(f"<assistant>\n{message.content}\n</assistant>")
        
        prompt_parts.append("<assistant>\n")
        return "\n".join(prompt_parts)
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        텍스트 임베딩 생성 (로컬 모델)
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        if not self.embedding_model:
            raise Exception("임베딩 모델이 초기화되지 않았습니다")
        
        try:
            # TODO: 로컬 임베딩 모델로 임베딩 생성
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {str(e)}")
            raise e
    
    async def batch_create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        배치 임베딩 생성 (로컬 모델)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        if not self.embedding_model:
            raise Exception("임베딩 모델이 초기화되지 않았습니다")
        
        try:
            # TODO: 배치 임베딩 생성
            embeddings = self.embedding_model.encode(texts)
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            self.logger.error(f"배치 임베딩 생성 실패: {str(e)}")
            raise e
    
    def create_system_message(self, content: str) -> ChatMessage:
        """시스템 메시지 생성"""
        return ChatMessage(
            role="system",
            content=content,
            timestamp=datetime.now().isoformat()
        )
    
    def create_user_message(self, content: str) -> ChatMessage:
        """사용자 메시지 생성"""
        return ChatMessage(
            role="user",
            content=content,
            timestamp=datetime.now().isoformat()
        )
    
    def create_assistant_message(self, content: str) -> ChatMessage:
        """어시스턴트 메시지 생성"""
        return ChatMessage(
            role="assistant",
            content=content,
            timestamp=datetime.now().isoformat()
        )
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        질문 의도 분석 (로컬 Mi:dm 2.0 활용)
        
        Args:
            query: 분석할 질문
            
        Returns:
            의도 분석 결과
        """
        try:
            # TODO: 의도 분석을 위한 시스템 프롬프트
            system_prompt = """당신은 외환 헷지전략 전문가입니다. 사용자의 질문을 분석하여 다음 카테고리로 분류해주세요:

1. 검색 요청 (search): 뉴스, 시계열 데이터, 시장 지표 조회
2. 거래 요청 (trading): 헷지 전략 분석, 거래 실행, 포트폴리오 관리
3. 분석 요청 (analysis): 시장 분석, 리스크 평가, 전략 비교
4. 조회 요청 (inquiry): 거래 히스토리, 계좌 정보, 설정 조회

응답은 JSON 형식으로 제공해주세요:
{
    "intent": "카테고리",
    "confidence": 0.0-1.0,
    "required_agents": ["agent1", "agent2"],
    "complexity": "low/medium/high",
    "parameters": {"key": "value"}
}"""
            
            messages = [
                self.create_system_message(system_prompt),
                self.create_user_message(query)
            ]
            
            response = await self.chat_completion(messages, temperature=0.3)
            
            # TODO: JSON 응답 파싱
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본값 반환
                return {
                    "intent": "unknown",
                    "confidence": 0.5,
                    "required_agents": [],
                    "complexity": "medium",
                    "parameters": {}
                }
                
        except Exception as e:
            self.logger.error(f"질문 의도 분석 실패: {str(e)}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "required_agents": [],
                "complexity": "low",
                "parameters": {}
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.config.model_name,
            "use_local": self.config.use_local,
            "device": self.config.device,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "embedding_model": self.config.embedding_model,
            "embedding_dimension": self.config.embedding_dimension,
            "model_loaded": self.model is not None,
            "embedding_loaded": self.embedding_model is not None
        }
