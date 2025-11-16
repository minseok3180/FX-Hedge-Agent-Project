"""OpenAI SDK 호출을 LangSmith로 추적하는 래퍼"""
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.utils.logger import get_logger

logger = get_logger("openai-tracer")


class TracedOpenAIClient:
    """LangSmith 추적이 포함된 OpenAI 클라이언트 래퍼"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.logger = logger
    
    def chat_completions_create(self, model: str, messages: List[Dict[str, str]], **kwargs):
        """Chat completions 생성 (LangSmith 추적 포함)"""
        call_info = self.logger.trace_openai_call(model, messages, **kwargs)
        
        try:
            # LangSmith가 자동으로 추적하도록 환경 변수 설정
            # OpenAI SDK는 환경 변수를 통해 자동으로 LangSmith에 추적됨
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
        """다른 OpenAI 메서드는 원본 클라이언트로 위임"""
        return getattr(self.client, name)

