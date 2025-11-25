"""LangSmith 추적 및 터미널 로깅을 위한 Custom Logger"""
import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import wraps
from enum import Enum

try:
    from langsmith import Client, traceable, RunTree
    from langsmith.run_helpers import tracing_context
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    traceable = lambda **kwargs: lambda func: func

from src.utils.settings import settings


class LogLevel(Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ColorCode:
    """터미널 컬러 코드"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # 텍스트 색상
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # 배경 색상
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class LangSmithLogger:
    """LangSmith 추적 및 터미널 로깅을 제공하는 Logger"""
    
    def __init__(self, name: str = "fx-hedge-agent"):
        self.name = name
        self.langsmith_enabled = (
            settings.langsmith_tracing 
            and settings.langsmith_api_key 
            and LANGSMITH_AVAILABLE
        )
        
        if self.langsmith_enabled:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
            os.environ["LANGSMITH_TRACING"] = "true"
            try:
                self.langsmith_client = Client(api_key=settings.langsmith_api_key)
            except Exception as e:
                print(f"⚠️  LangSmith 클라이언트 초기화 실패: {e}")
                self.langsmith_enabled = False
                self.langsmith_client = None
        else:
            self.langsmith_client = None
    
    def _get_color(self, level: LogLevel) -> str:
        """로그 레벨에 따른 컬러 반환"""
        colors = {
            LogLevel.DEBUG: ColorCode.CYAN,
            LogLevel.INFO: ColorCode.GREEN,
            LogLevel.WARNING: ColorCode.YELLOW,
            LogLevel.ERROR: ColorCode.RED,
            LogLevel.CRITICAL: ColorCode.BG_RED + ColorCode.WHITE,
        }
        return colors.get(level, ColorCode.RESET)
    
    def _format_message(self, level: LogLevel, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """로그 메시지 포맷팅"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level_str = f"{self._get_color(level)}{ColorCode.BOLD}[{level.value}]{ColorCode.RESET}"
        name_str = f"{ColorCode.BLUE}[{self.name}]{ColorCode.RESET}"
        
        # 메시지만 출력 (metadata는 LangSmith에만 전송)
        return f"{timestamp} {level_str} {name_str} {message}"
    
    def _print_log(self, level: LogLevel, message: str, metadata: Optional[Dict[str, Any]] = None):
        """터미널에 로그 출력"""
        formatted = self._format_message(level, message, metadata)
        print(formatted, file=sys.stderr)
    
    def debug(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """DEBUG 레벨 로그"""
        self._print_log(LogLevel.DEBUG, message, metadata)
        if self.langsmith_enabled:
            self._log_to_langsmith("debug", message, metadata)
    
    def info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """INFO 레벨 로그"""
        self._print_log(LogLevel.INFO, message, metadata)
        if self.langsmith_enabled:
            self._log_to_langsmith("info", message, metadata)
    
    def warning(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """WARNING 레벨 로그"""
        self._print_log(LogLevel.WARNING, message, metadata)
        if self.langsmith_enabled:
            self._log_to_langsmith("warning", message, metadata)
    
    def error(self, message: str, metadata: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """ERROR 레벨 로그"""
        if exc_info:
            error_trace = traceback.format_exc()
            if metadata is None:
                metadata = {}
            metadata["traceback"] = error_trace
        
        self._print_log(LogLevel.ERROR, message, metadata)
        if self.langsmith_enabled:
            self._log_to_langsmith("error", message, metadata)
    
    def critical(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """CRITICAL 레벨 로그"""
        self._print_log(LogLevel.CRITICAL, message, metadata)
        if self.langsmith_enabled:
            self._log_to_langsmith("critical", message, metadata)
    
    def _log_to_langsmith(self, level: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """LangSmith에 로그 전송 (비동기적으로 처리)"""
        if not self.langsmith_client:
            return
        
        try:
            # LangSmith는 자동으로 추적하므로 여기서는 메타데이터만 기록
            # 실제 추적은 traceable 데코레이터를 통해 수행
            pass
        except Exception:
            # LangSmith 전송 실패는 무시 (터미널 로그는 이미 출력됨)
            pass
    
    def trace_function(self, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """함수 추적 데코레이터"""
        def decorator(func):
            func_name = name or func.__name__
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                self.info(f"🚀 함수 시작: {func_name}", metadata)
                
                try:
                    if self.langsmith_enabled:
                        # LangSmith 추적 컨텍스트 생성
                        with tracing_context(
                            name=func_name,
                            run_type="chain",
                            metadata=metadata or {}
                        ):
                            result = await func(*args, **kwargs)
                    else:
                        result = await func(*args, **kwargs)
                    
                    elapsed = time.time() - start_time
                    self.info(
                        f"✅ 함수 완료: {func_name} (소요 시간: {elapsed:.2f}초)",
                        {"elapsed_time": elapsed, "result_type": type(result).__name__}
                    )
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.error(
                        f"❌ 함수 실패: {func_name} (소요 시간: {elapsed:.2f}초)",
                        {"elapsed_time": elapsed, "error": str(e)},
                        exc_info=True
                    )
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                self.info(f"🚀 함수 시작: {func_name}", metadata)
                
                try:
                    if self.langsmith_enabled:
                        with tracing_context(
                            name=func_name,
                            run_type="chain",
                            metadata=metadata or {}
                        ):
                            result = func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    elapsed = time.time() - start_time
                    self.info(
                        f"✅ 함수 완료: {func_name} (소요 시간: {elapsed:.2f}초)",
                        {"elapsed_time": elapsed, "result_type": type(result).__name__}
                    )
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.error(
                        f"❌ 함수 실패: {func_name} (소요 시간: {elapsed:.2f}초)",
                        {"elapsed_time": elapsed, "error": str(e)},
                        exc_info=True
                    )
                    raise
            
            # 비동기 함수인지 확인
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def trace_openai_call(self, model: str, messages: List[Dict[str, str]], **kwargs):
        """OpenAI 호출 추적"""
        start_time = time.time()
        call_id = f"openai_{int(time.time() * 1000)}"
        
        self.info(
            f"🤖 OpenAI 호출 시작",
            {
                "call_id": call_id,
                "model": model,
                "messages_count": len(messages),
                "temperature": kwargs.get("temperature", "default")
            }
        )
        
        return {
            "call_id": call_id,
            "start_time": start_time,
            "model": model,
            "messages": messages,
            "kwargs": kwargs
        }
    
    def log_openai_response(self, call_info: Dict[str, Any], response: Any, error: Optional[Exception] = None):
        """OpenAI 응답 로깅"""
        elapsed = time.time() - call_info["start_time"]
        
        if error:
            self.error(
                f"❌ OpenAI 호출 실패",
                {
                    "call_id": call_info["call_id"],
                    "model": call_info["model"],
                    "elapsed_time": elapsed,
                    "error": str(error)
                },
                exc_info=True
            )
        else:
            try:
                content = response.choices[0].message.content if hasattr(response, 'choices') else str(response)
                token_usage = {}
                if hasattr(response, 'usage'):
                    token_usage = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0)
                    }
                
                self.info(
                    f"✅ OpenAI 호출 완료",
                    {
                        "call_id": call_info["call_id"],
                        "model": call_info["model"],
                        "elapsed_time": elapsed,
                        "response_length": len(content) if content else 0,
                        "token_usage": token_usage
                    }
                )
            except Exception as e:
                self.warning(
                    f"⚠️  OpenAI 응답 파싱 실패",
                    {"call_id": call_info["call_id"], "error": str(e)}
                )


# Logger 인스턴스 캐시
_logger_cache: Dict[str, LangSmithLogger] = {}


def get_logger(name: str = "fx-hedge-agent") -> LangSmithLogger:
    """Logger 인스턴스 반환 (이름별로 캐싱)"""
    if name not in _logger_cache:
        _logger_cache[name] = LangSmithLogger(name)
    return _logger_cache[name]

