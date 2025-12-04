"""
도구 모듈 - LangChain @tool decorator를 사용한 함수형 도구들

## 에이전트용 툴 (읽기/검색 전용)
- rdb_query_hard: RDB 조회 (하드코딩 쿼리)
- rdb_query_llm: RDB 조회 (LLM 생성 쿼리)
- rdb_get_latest_ecos_date: ECOS 최신 날짜 조회
- vdb_search: 벡터 검색
- web_search: 웹 검색

## 독립 스크립트용 함수 (CLI/배치 작업)
- rdb_modify: DB 수정 (INSERT/UPDATE/DELETE)
- vdb_create_collection: Qdrant 컬렉션 생성
- vdb_upsert_points: 벡터 포인트 업서트
- run_ecos_pipeline: ECOS ETL 파이프라인
- ingest_documents: 문서 인제스트
"""

# ============================================================================
# 에이전트용 툴 (읽기/검색 전용)
# ============================================================================
from .rdb import (
    rdb_query_hard,
    rdb_query_llm,
    rdb_get_latest_ecos_date,
)
from .vdb import vdb_search
from .web_search import web_search

# ============================================================================
# 독립 스크립트용 함수 (CLI/배치 작업)
# ============================================================================
from .rdb import (
    rdb_modify,
    _db_connection,  # ecos.py 등에서 재사용
)
from .vdb import (
    vdb_create_collection,
    vdb_upsert_points,
    _get_qdrant_client,  # ingest_docs.py 등에서 재사용
)
from .ecos import (
    run_ecos_pipeline,
    upload_ecos_to_db,
)
from .ingest_docs import ingest_documents

# ============================================================================
# 유틸리티 및 스키마
# ============================================================================
from src.utils.tools import (
    tool,
    get_all_tools,
    get_all_tools_with_write,
    bind_tools_to_llm,
    ToolError,
    handle_tool_error,
    # 스키마 (에이전트용 툴)
    RDBQueryHardInput,
    RDBQueryLLMInput,
    RDBGetLatestEcosDateInput,
    VDBSearchInput,
    WebSearchInput,
    # 스키마 (독립 스크립트용)
    RDBModifyInput,
    VDBCreateCollectionInput,
    VDBUpsertPointsInput,
)

__all__ = [
    # ========== 에이전트용 툴 (읽기/검색 전용) ==========
    "rdb_query_hard",
    "rdb_query_llm",
    "rdb_get_latest_ecos_date",
    "vdb_search",
    "web_search",
    
    # ========== 독립 스크립트용 함수 ==========
    # RDB 수정
    "rdb_modify",
    "_db_connection",
    # VDB 수정
    "vdb_create_collection",
    "vdb_upsert_points",
    "_get_qdrant_client",
    # ECOS ETL
    "run_ecos_pipeline",
    "upload_ecos_to_db",
    # 문서 인제스트
    "ingest_documents",
    
    # ========== 유틸리티 ==========
    "tool",
    "get_all_tools",
    "get_all_tools_with_write",
    "bind_tools_to_llm",
    "ToolError",
    "handle_tool_error",
    
    # ========== 스키마 ==========
    "RDBQueryHardInput",
    "RDBQueryLLMInput",
    "RDBGetLatestEcosDateInput",
    "RDBModifyInput",
    "VDBSearchInput",
    "VDBCreateCollectionInput",
    "VDBUpsertPointsInput",
    "WebSearchInput",
]

