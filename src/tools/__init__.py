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
# vdb는 optional dependency이므로 lazy import
try:
    from .vdb import vdb_search
except ImportError:
    vdb_search = None  # qdrant_client가 없을 경우 None
from .web_search import web_search
from .calculator import (
    compute_expected_fx_return,
    compute_optimal_hedge_weight,
    compute_log_returns_from_prices,
    compute_sigma_and_rho_from_returns,
    compute_fx_vol_and_rho_from_csv,
    compute_fx_vol_and_rho_from_rdb,
    compute_all,
    CalculatorTool,  # 기존 코드 호환성을 위해 유지
)

# ============================================================================
# 독립 스크립트용 함수 (CLI/배치 작업)
# ============================================================================
from .rdb import (
    rdb_modify,
    _db_connection,  # ecos.py 등에서 재사용
)
# vdb는 optional dependency이므로 lazy import
try:
    from .vdb import (
        vdb_create_collection,
        vdb_upsert_points,
        _get_qdrant_client,  # ingest_docs.py 등에서 재사용
    )
except ImportError:
    # qdrant_client가 없을 경우 None으로 설정
    vdb_create_collection = None
    vdb_upsert_points = None
    _get_qdrant_client = None
# ecos는 optional dependency
try:
    from .ecos import (
        run_ecos_pipeline,
        upload_ecos_to_db,
    )
except ImportError:
    run_ecos_pipeline = None
    upload_ecos_to_db = None

# ingest_docs는 optional dependency
try:
    from .ingest_docs import ingest_documents
except ImportError:
    ingest_documents = None

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
    UserInfoGetInput,
    UserInfoUpsertInput,
    VDBSearchInput,
    WebSearchInput,
    # 스키마 (독립 스크립트용)
    RDBModifyInput,
    VDBCreateCollectionInput,
    VDBUpsertPointsInput,
    # Calculator 스키마
    ComputeExpectedFxReturnInput,
    ComputeOptimalHedgeWeightInput,
    ComputeLogReturnsFromPricesInput,
    ComputeSigmaAndRhoFromReturnsInput,
    ComputeFxVolAndRhoFromCsvInput,
    ComputeFxVolAndRhoFromRdbInput,
    ComputeAllInput,
)

__all__ = [
    # ========== 에이전트용 툴 (읽기/검색 전용) ==========
    "rdb_query_hard",
    "rdb_query_llm",
    "rdb_get_latest_ecos_date",
    "vdb_search",  # None일 수 있음 (qdrant_client가 없을 경우)
    "web_search",
    # Calculator tools
    "compute_expected_fx_return",
    "compute_optimal_hedge_weight",
    "compute_log_returns_from_prices",
    "compute_sigma_and_rho_from_returns",
    "compute_fx_vol_and_rho_from_csv",
    "compute_fx_vol_and_rho_from_rdb",
    "compute_all",
    "CalculatorTool",
    
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
    "UserInfoGetInput",
    "UserInfoUpsertInput",
    "RDBModifyInput",
    "VDBSearchInput",
    "VDBCreateCollectionInput",
    "VDBUpsertPointsInput",
    "WebSearchInput",
    # Calculator 스키마
    "ComputeExpectedFxReturnInput",
    "ComputeOptimalHedgeWeightInput",
    "ComputeLogReturnsFromPricesInput",
    "ComputeSigmaAndRhoFromReturnsInput",
    "ComputeFxVolAndRhoFromCsvInput",
    "ComputeFxVolAndRhoFromRdbInput",
    "ComputeAllInput",
]
