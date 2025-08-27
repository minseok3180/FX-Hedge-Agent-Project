from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # 데이터
    base_data_dir: str = "data"
    fx_symbol: str = "USDKRW=X"
    start_date: str = "2022-01-01"
    end_date: Optional[str] = None  # None이면 오늘
    news_count: int = 20

    # 예측
    lstm_epochs: int = 3
    lstm_hidden: int = 32
    lstm_lookback: int = 20
    lstm_lr: float = 1e-3

    # RAG/임베딩
    vector_db_dir: str = "data/vector_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    hf_model_name: str = "K-intelligence/Midm-2.0-Mini-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.2

    # 기타
    verbose: bool = True


