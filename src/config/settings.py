"""환경 설정 및 구성 관리"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel


class OpenAIConfig(BaseModel):
    """OpenAI 설정"""
    api_key: str
    model: str = "gpt-4.1"


class DatabaseConfig(BaseModel):
    """데이터베이스 설정"""
    host: str
    port: int = 3306
    user: str
    password: str
    name: str


class QdrantConfig(BaseModel):
    """Qdrant 설정"""
    url: Optional[str] = None  # GCP Qdrant 전체 URL (예: https://xxx.qdrant.io)
    host: Optional[str] = None  # 로컬 Qdrant 호스트
    port: int = 6333  # 로컬 Qdrant 포트
    api_key: Optional[str] = None  # API 키 (GCP Qdrant 필수)


class APIConfig(BaseModel):
    """API 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "FX Hedge Agent API"
    version: str = "1.0.0"


class WebSearchConfig(BaseModel):
    """웹서치 설정"""
    api_key: Optional[str] = None
    engine_id: Optional[str] = None


class LangSmithConfig(BaseModel):
    """LangSmith 설정"""
    api_key: Optional[str] = None
    project: str = "fx-hedge-agent"
    tracing: str = "true"


class Settings(BaseModel):
    """애플리케이션 설정"""
    openai: OpenAIConfig
    database: DatabaseConfig
    qdrant: QdrantConfig
    api: APIConfig
    web_search: WebSearchConfig
    langsmith: LangSmithConfig


def _resolve_env_vars(value: Any) -> Any:
    """환경 변수 해석"""
    if isinstance(value, str):
        # ${VAR} 또는 ${VAR:-default} 형식 처리
        pattern = r'\$\{([^:}]+)(?::-([^}]*))?\}'
        
        def replace(match):
            var_name = match.group(1)
            default = match.group(2) if match.lastindex >= 2 else None
            env_value = os.getenv(var_name)
            
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                return ""
        
        result = re.sub(pattern, replace, value)
        
        # 빈 문자열이면 None 반환 (Optional 필드용)
        if result == "":
            return None
        return result
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_settings() -> Settings:
    """설정 파일 로드"""
    # 프로젝트 루트 디렉토리 찾기
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # config.json 경로
    config_path = project_root / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.json 파일을 찾을 수 없습니다: {config_path}")
    
    # config.json 읽기
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    # 환경 변수 해석
    resolved_config = _resolve_env_vars(config_data)
    
    # 숫자 필드 변환 (port 등)
    if "database" in resolved_config and "port" in resolved_config["database"]:
        resolved_config["database"]["port"] = int(resolved_config["database"]["port"])
    if "qdrant" in resolved_config and "port" in resolved_config["qdrant"]:
        resolved_config["qdrant"]["port"] = int(resolved_config["qdrant"]["port"])
    if "api" in resolved_config and "port" in resolved_config["api"]:
        resolved_config["api"]["port"] = int(resolved_config["api"]["port"])
    
    # Settings 객체 생성
    return Settings(**resolved_config)


# 하위 호환성을 위한 속성 접근
class SettingsWrapper:
    """하위 호환성을 위한 래퍼"""
    def __init__(self, settings: Settings):
        self._settings = settings
    
    @property
    def openai_api_key(self) -> str:
        return self._settings.openai.api_key
    
    @property
    def openai_model(self) -> str:
        return self._settings.openai.model
    
    @property
    def db_host(self) -> str:
        return self._settings.database.host
    
    @property
    def db_port(self) -> int:
        return self._settings.database.port
    
    @property
    def db_user(self) -> str:
        return self._settings.database.user
    
    @property
    def db_password(self) -> str:
        return self._settings.database.password
    
    @property
    def db_name(self) -> str:
        return self._settings.database.name
    
    @property
    def qdrant_url(self) -> Optional[str]:
        return self._settings.qdrant.url
    
    @property
    def qdrant_host(self) -> Optional[str]:
        return self._settings.qdrant.host
    
    @property
    def qdrant_port(self) -> int:
        return self._settings.qdrant.port
    
    @property
    def qdrant_api_key(self) -> Optional[str]:
        return self._settings.qdrant.api_key
    
    @property
    def api_host(self) -> str:
        return self._settings.api.host
    
    @property
    def api_port(self) -> int:
        return self._settings.api.port
    
    @property
    def api_title(self) -> str:
        return self._settings.api.title
    
    @property
    def api_version(self) -> str:
        return self._settings.api.version
    
    @property
    def web_search_api_key(self) -> Optional[str]:
        return self._settings.web_search.api_key
    
    @property
    def web_search_engine_id(self) -> Optional[str]:
        return self._settings.web_search.engine_id
    
    @property
    def langsmith_api_key(self) -> Optional[str]:
        return self._settings.langsmith.api_key
    
    @property
    def langsmith_project(self) -> str:
        return self._settings.langsmith.project
    
    @property
    def langsmith_tracing(self) -> bool:
        return self._settings.langsmith.tracing.lower() == "true"


# 전역 설정 인스턴스 (하위 호환성을 위해 래퍼 사용)
_settings_instance = load_settings()
settings = SettingsWrapper(_settings_instance)
