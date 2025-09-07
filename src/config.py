"""
외환 헷지전략 에이전트 설정 파일
MVP 목적의 최소한의 기능을 포함한 설정 관리
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DatabaseConfig:
    """데이터베이스 관련 설정 (로컬 환경용)"""
    # TSDB (Time Series Database) - 환율 및 시계열 데이터 (추후 개발)
    tsdb_host: str = "localhost"
    tsdb_port: int = 8086
    tsdb_database: str = "fx_timeseries"
    tsdb_username: str = "admin"
    tsdb_password: str = "password"
    
    # VDB (Vector Database) - 로컬 ChromaDB 사용
    vdb_type: str = "chromadb"  # chromadb, weaviate 등
    vdb_path: str = "./data/chromadb"  # 로컬 ChromaDB 경로
    vdb_collection: str = "hedge_strategies"
    
    # RDB (Relational Database) - CSV 파일로 대체
    rdb_type: str = "csv"  # csv, postgresql 등
    csv_data_dir: str = "./data/csv"  # CSV 데이터 디렉토리
    user_data_file: str = "users.csv"
    trading_history_file: str = "trading_history.csv"
    portfolio_file: str = "portfolios.csv"


@dataclass
class LLMConfig:
    """LLM 모델 관련 설정 (로컬 환경용)"""
    # 로컬 Mi:dm 2.0 모델 설정 (transformers 라이브러리 사용)
    model_name: str = "K-intelligence/Midm-2.0-Base-Instruct"
    use_local: bool = True  # 로컬 모델 사용 여부
    device: str = "auto"  # auto, cpu, cuda
    
    # 모델 파라미터
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # 임베딩 모델 (transformers 라이브러리 사용)
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    embedding_dimension: int = 768
    
    # 로컬 모델 특화 설정
    context_length: int = 4096
    batch_size: int = 1
    timeout: int = 30


@dataclass
class AgentConfig:
    """에이전트 관련 설정"""
    # Search Agent 설정
    search_timeout: int = 30  # 웹 API 호출 타임아웃 (초)
    max_news_count: int = 100  # 최대 뉴스 검색 개수
    tsdb_query_limit: int = 1000  # TSDB 쿼리 제한
    
    # Trading Agent 설정
    max_strategy_count: int = 50  # 최대 헷지 전략 개수
    trading_timeout: int = 60  # 거래 실행 타임아웃 (초)
    
    # Supervisor 설정
    max_iterations: int = 10  # 최대 반복 횟수
    conversation_timeout: int = 300  # 대화 타임아웃 (초)


@dataclass
class Config:
    """전체 설정 관리 클래스"""
    
    def __init__(self):
        # 환경변수에서 설정 로드 (우선순위: 환경변수 > 기본값)
        self.db = DatabaseConfig(
            tsdb_host=os.getenv("TSDB_HOST", "localhost"),
            tsdb_port=int(os.getenv("TSDB_PORT", "8086")),
            tsdb_database=os.getenv("TSDB_DATABASE", "fx_timeseries"),
            tsdb_username=os.getenv("TSDB_USERNAME", "admin"),
            tsdb_password=os.getenv("TSDB_PASSWORD", "password"),
            
            vdb_type=os.getenv("VDB_TYPE", "chromadb"),
            vdb_path=os.getenv("VDB_PATH", "./data/chromadb"),
            vdb_collection=os.getenv("VDB_COLLECTION", "hedge_strategies"),
            
            rdb_type=os.getenv("RDB_TYPE", "csv"),
            csv_data_dir=os.getenv("CSV_DATA_DIR", "./data/csv"),
            user_data_file=os.getenv("USER_DATA_FILE", "users.csv"),
            trading_history_file=os.getenv("TRADING_HISTORY_FILE", "trading_history.csv"),
            portfolio_file=os.getenv("PORTFOLIO_FILE", "portfolios.csv")
        )
        
        self.llm = LLMConfig(
            model_name=os.getenv("LLM_MODEL_NAME", "K-intelligence/Midm-2.0-Base-Instruct"),
            use_local=os.getenv("LLM_USE_LOCAL", "true").lower() == "true",
            device=os.getenv("LLM_DEVICE", "auto"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
            frequency_penalty=float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0")),
            presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", "0.0")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            context_length=int(os.getenv("LLM_CONTEXT_LENGTH", "4096")),
            batch_size=int(os.getenv("LLM_BATCH_SIZE", "1")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30"))
        )
        
        self.agent = AgentConfig(
            search_timeout=int(os.getenv("SEARCH_TIMEOUT", "30")),
            max_news_count=int(os.getenv("MAX_NEWS_COUNT", "100")),
            tsdb_query_limit=int(os.getenv("TSDB_QUERY_LIMIT", "1000")),
            max_strategy_count=int(os.getenv("MAX_STRATEGY_COUNT", "50")),
            trading_timeout=int(os.getenv("TRADING_TIMEOUT", "60")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            conversation_timeout=int(os.getenv("CONVERSATION_TIMEOUT", "300"))
        )
    
    def get_db_connection_strings(self) -> Dict[str, str]:
        """데이터베이스 연결 문자열 반환"""
        return {
            "tsdb": f"http://{self.db.tsdb_username}:{self.db.tsdb_password}@{self.db.tsdb_host}:{self.db.tsdb_port}/{self.db.tsdb_database}",
            "vdb": f"http://{self.db.vdb_host}:{self.db.vdb_port}/collections/{self.db.vdb_collection}",
            "rdb": f"postgresql://{self.db.rdb_username}:{self.db.rdb_password}@{self.db.rdb_host}:{self.db.rdb_port}/{self.db.rdb_database}"
        }
    
    def validate_config(self) -> bool:
        """설정 유효성 검증"""
        # TODO: 각 설정값의 유효성 검증 로직 구현
        # - API 키 형식 검증
        # - 포트 번호 범위 검증
        # - 데이터베이스 연결 테스트
        # - LLM API 연결 테스트
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "database": {
                "tsdb": {
                    "host": self.db.tsdb_host,
                    "port": self.db.tsdb_port,
                    "database": self.db.tsdb_database,
                    "username": self.db.tsdb_username
                },
                "vdb": {
                    "host": self.db.vdb_host,
                    "port": self.db.vdb_port,
                    "collection": self.db.vdb_collection
                },
                "rdb": {
                    "host": self.db.rdb_host,
                    "port": self.db.rdb_port,
                    "database": self.db.rdb_database,
                    "username": self.db.rdb_username
                }
            },
            "llm": {
                "model_name": self.llm.model_name,
                "api_base_url": self.llm.api_base_url,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
                "embedding_model": self.llm.embedding_model
            },
            "agent": {
                "search_timeout": self.agent.search_timeout,
                "max_news_count": self.agent.max_news_count,
                "tsdb_query_limit": self.agent.tsdb_query_limit,
                "max_strategy_count": self.agent.max_strategy_count,
                "trading_timeout": self.agent.trading_timeout,
                "max_iterations": self.agent.max_iterations,
                "conversation_timeout": self.agent.conversation_timeout
            }
        }