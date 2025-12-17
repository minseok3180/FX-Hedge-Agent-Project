"""Tool 관련 유틸리티 (에러 처리, 스키마, decorator 통합)"""
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from functools import wraps
from src.utils.logger import get_logger

logger = get_logger("tools-utils")

# LangGraph Command import
try:
    from langgraph.types import Command
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    Command = None

# ============================================================================
# LangChain Tool Decorator
# ============================================================================

try:
    from langchain_core.tools import tool
    LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError:
    LANGCHAIN_TOOLS_AVAILABLE = False
    # Fallback decorator
    def tool(*args, **kwargs) -> Callable:
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

# ============================================================================
# Tool 스키마 정의 (Pydantic 모델)
# ============================================================================

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback: 간단한 BaseModel
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(*args, **kwargs):
        return None


class RDBQueryHardInput(BaseModel):
    """RDB 하드 쿼리 입력 스키마"""
    query_key: str = Field(
        description="실행할 쿼리의 키 (예: 'get_by_date', 'get_by_range', 'get_latest')",
        examples=["get_by_date", "get_by_range", "get_latest"]
    )
    params: Optional[Tuple[Any, ...]] = Field(
        default=None,
        description="쿼리 파라미터 (튜플) - placeholder 사용 시 None"
    )
    state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="AgentState 딕셔너리 (placeholder 치환용)"
    )


class RDBQueryLLMInput(BaseModel):
    """RDB LLM 쿼리 입력 스키마"""
    user_request: str = Field(
        description="사용자 요청 (자연어)",
        examples=["2024-01-15의 USD/KRW 환율을 조회해줘"]
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="추가 컨텍스트 정보"
    )


class RDBModifyInput(BaseModel):
    """RDB 수정 쿼리 입력 스키마"""
    query: str = Field(
        description="실행할 SQL 쿼리 (INSERT, UPDATE, DELETE)",
        examples=["UPDATE eiExchangeRate SET usdkrw = 1300.0 WHERE date = '2024-01-15'"]
    )
    params: Optional[Tuple[Any, ...]] = Field(
        default=None,
        description="쿼리 파라미터 (튜플)"
    )


class VDBSearchInput(BaseModel):
    """VDB 검색 입력 스키마"""
    query_vector: List[float] = Field(
        description="검색할 벡터",
        examples=[[0.1, 0.2, 0.3, 0.4, 0.5]]
    )
    collection_name: str = Field(
        default="fx_hedge_data",
        description="컬렉션 이름"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=100,
        description="반환할 결과 수"
    )


class VDBCreateCollectionInput(BaseModel):
    """VDB 컬렉션 생성 입력 스키마"""
    collection_name: str = Field(
        default="fx_hedge_data",
        description="컬렉션 이름"
    )
    vector_size: int = Field(
        default=384,
        ge=1,
        description="벡터 크기"
    )


class VDBUpsertPointsInput(BaseModel):
    """VDB 포인트 업서트 입력 스키마"""
    points: List[Dict[str, Any]] = Field(
        description="업서트할 포인트 리스트"
    )
    collection_name: str = Field(
        default="fx_hedge_data",
        description="컬렉션 이름"
    )


class WebSearchInput(BaseModel):
    """웹 검색 입력 스키마"""
    query: str = Field(
        description="검색 쿼리",
        examples=["USD/KRW 환율 최신 뉴스"]
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="반환할 결과 수 (최대 10)"
    )


class RDBGetLatestEcosDateInput(BaseModel):
    """ECOS 데이터 최신 날짜 조회 입력 스키마"""
    table_name: str = Field(
        default="eiExchangeRate",
        description="조회할 테이블명"
    )
    date_column: str = Field(
        default="date",
        description="날짜 컬럼명"
    )


class UserInfoGetInput(BaseModel):
    """사용자 정보 조회 입력 스키마"""
    user_id: str = Field(
        description="조회할 사용자 ID",
        examples=["user_001"],
    )


class UserInfoUpsertInput(BaseModel):
    """사용자 정보 입력/수정(Upsert) 입력 스키마"""
    user_id: str = Field(
        description="사용자 ID (PK, 중복 시 업데이트)",
        examples=["user_001"],
    )
    name: str = Field(
        description="사용자 이름",
        examples=["홍길동"],
    )
    age: int = Field(
        description="사용자 나이",
        ge=0,
        le=150,
        examples=[35],
    )
    gender: str = Field(
        description="성별 (예: 'male', 'female', 'other')",
        examples=["male"],
    )
    total_assets: float = Field(
        description="총 재산 (KRW 기준, 원 단위)",
        examples=[100_000_000.0],
    )
    overseas_assets: float = Field(
        description="해외 재산 (환산 KRW 기준, 원 단위)",
        examples=[30_000_000.0],
    )
    risk_profile: str = Field(
        description="투자 성향 (예: 'conservative', 'moderate', 'aggressive')",
        examples=["moderate"],
    )


# ============================================================================
# Calculator Tool 스키마
# ============================================================================

class ComputeExpectedFxReturnInput(BaseModel):
    """환율 기대수익률 계산 입력 스키마"""
    signal_score: float = Field(
        description="환율 시그널 점수 S (범위: -1 ~ +1, -1 = 강한 달러 약세, +1 = 강한 달러 강세)",
        ge=-1.0,
        le=1.0,
        examples=[0.5]
    )
    alpha: float = Field(
        default=0.05,
        description="스케일링 파라미터 (기본값: 0.05)",
        examples=[0.05]
    )


class ComputeOptimalHedgeWeightInput(BaseModel):
    """최적 환헷지 비중 계산 입력 스키마"""
    sigma_asset: float = Field(
        description="해외자산 수익률 변동성 σ_Asset (예: S&P500 일간 로그수익률 표준편차)",
        gt=0.0,
        examples=[0.15]
    )
    sigma_fx: float = Field(
        description="환율 수익률 변동성 σ_FX (예: USD/KRW 일간 로그수익률 표준편차)",
        gt=0.0,
        examples=[0.02]
    )
    rho: float = Field(
        description="자산 수익률과 환율 수익률의 상관계수 (Corr(R_asset, R_FX))",
        ge=-1.0,
        le=1.0,
        examples=[0.3]
    )
    expected_fx_return: float = Field(
        description="기대 환율 수익률 E[R_FX]",
        examples=[0.0001]
    )
    risk_aversion: float = Field(
        description="위험회피도 λ (값이 클수록 보수적, 일반적으로 1 ~ 10 정도)",
        gt=0.0,
        examples=[4.0]
    )
    clip: bool = Field(
        default=True,
        description="True면 결과를 [0, 1] 범위로 클리핑",
        examples=[True]
    )


class ComputeLogReturnsFromPricesInput(BaseModel):
    """로그수익률 계산 입력 스키마"""
    prices: list = Field(
        description="가격 시계열 (리스트 또는 pandas Series를 리스트로 변환)",
        examples=[[100.0, 101.0, 102.0, 101.5]]
    )


class ComputeSigmaAndRhoFromReturnsInput(BaseModel):
    """자산 수익률과 환율 수익률로부터 변동성 및 상관계수 계산 입력 스키마"""
    asset_returns: list = Field(
        description="자산(예: S&P500) 일간 수익률 시계열",
        examples=[[0.01, -0.02, 0.015, 0.005]]
    )
    fx_returns: list = Field(
        description="환율(예: USD/KRW) 일간 수익률 시계열",
        examples=[[0.001, -0.001, 0.002, -0.0005]]
    )


class ComputeFxVolAndRhoFromCsvInput(BaseModel):
    """CSV 파일에서 환율 변동성 및 상관계수 계산 입력 스키마"""
    csv_path: Optional[str] = Field(
        default=None,
        description="CSV 파일 경로 (예: 'notebook/yonju/df.csv'). df가 제공되면 무시됨.",
        examples=["notebook/yonju/df.csv"]
    )
    fx_col: str = Field(
        default="usdkrw(target)",
        description="환율 컬럼명 (예: 'usdkrw(target)')",
        examples=["usdkrw(target)"]
    )
    asset_col: Optional[str] = Field(
        default=None,
        description="자산(예: S&P500) 가격 또는 수익률 컬럼명. None이면 ρ는 0.0으로 반환.",
        examples=["SPY_close"]
    )


class ComputeFxVolAndRhoFromRdbInput(BaseModel):
    """RDB에서 환율 변동성 및 상관계수 계산 입력 스키마"""
    days: int = Field(
        default=252,
        description="조회할 최근 일수 (기본값: 252일, 약 1년)",
        ge=1,
        examples=[252]
    )
    asset_col: Optional[str] = Field(
        default=None,
        description="자산(예: S&P500) 가격 또는 수익률 컬럼명. None이면 ρ는 0.0으로 반환.",
        examples=["SPY_close"]
    )


class ComputeAllInput(BaseModel):
    """모든 계산을 한 번에 수행하는 통합 tool 입력 스키마"""
    signal_score: float = Field(
        description="환율 시그널 점수",
        ge=-1.0,
        le=1.0,
        examples=[0.5]
    )
    csv_path: str = Field(
        description="CSV 파일 경로",
        examples=["notebook/yonju/df.csv"]
    )
    sigma_asset: float = Field(
        default=0.15,
        description="해외자산 수익률 변동성",
        gt=0.0,
        examples=[0.15]
    )
    risk_aversion: float = Field(
        default=4.0,
        description="위험회피도",
        gt=0.0,
        examples=[4.0]
    )
    alpha: float = Field(
        default=0.05,
        description="스케일링 파라미터",
        examples=[0.05]
    )
    fx_col: str = Field(
        default="usdkrw(target)",
        description="환율 컬럼명",
        examples=["usdkrw(target)"]
    )
    asset_col: Optional[str] = Field(
        default=None,
        description="자산 가격 컬럼명",
        examples=["SPY_close"]
    )
    clip: bool = Field(
        default=True,
        description="True면 결과를 [0, 1] 범위로 클리핑",
        examples=[True]
    )

# ============================================================================
# Tool 에러 처리
# ============================================================================

class ToolError(Exception):
    """Tool 실행 중 발생하는 커스텀 에러"""
    def __init__(self, tool_name: str, message: str, error: Optional[Exception] = None):
        self.tool_name = tool_name
        self.message = message
        self.original_error = error
        super().__init__(f"[{tool_name}] {message}")


def handle_tool_error(tool_name: str):
    """
    Tool 함수의 에러 처리를 통일하는 데코레이터 (LangSmith traceable 포함)
    
    Args:
        tool_name: Tool 이름
        
    Usage:
        @handle_tool_error("rdb_query_hard")
        async def rdb_query_hard(...):
            ...
    """
    def decorator(func):
        # LangSmith traceable 데코레이터 적용 (가능한 경우)
        try:
            from langsmith import traceable
            if traceable:
                func = traceable(name=f"tool_{tool_name}", run_type="tool")(func)
        except ImportError:
            pass
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ToolError:
                # ToolError는 그대로 전파
                raise
            except ValueError as e:
                # ValueError는 ToolError로 변환
                logger.error(
                    f"❌ {tool_name} 실행 실패 (값 오류)",
                    {"error": str(e)},
                    exc_info=True
                )
                raise ToolError(tool_name, f"값 오류: {str(e)}", e)
            except KeyError as e:
                # KeyError는 ToolError로 변환
                logger.error(
                    f"❌ {tool_name} 실행 실패 (키 오류)",
                    {"error": str(e)},
                    exc_info=True
                )
                raise ToolError(tool_name, f"키 오류: {str(e)}", e)
            except Exception as e:
                # 기타 예외는 ToolError로 변환
                logger.error(
                    f"❌ {tool_name} 실행 실패",
                    {"error": str(e)},
                    exc_info=True
                )
                raise ToolError(tool_name, f"예상치 못한 오류: {str(e)}", e)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ToolError:
                raise
            except ValueError as e:
                logger.error(
                    f"❌ {tool_name} 실행 실패 (값 오류)",
                    {"error": str(e)},
                    exc_info=True
                )
                raise ToolError(tool_name, f"값 오류: {str(e)}", e)
            except KeyError as e:
                logger.error(
                    f"❌ {tool_name} 실행 실패 (키 오류)",
                    {"error": str(e)},
                    exc_info=True
                )
                raise ToolError(tool_name, f"키 오류: {str(e)}", e)
            except Exception as e:
                logger.error(
                    f"❌ {tool_name} 실행 실패",
                    {"error": str(e)},
                    exc_info=True
                )
                raise ToolError(tool_name, f"예상치 못한 오류: {str(e)}", e)
        
        # 비동기 함수인지 확인
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# ============================================================================
# Tool 관리 함수
# ============================================================================

def get_all_tools() -> List[Any]:
    """
    에이전트에서 사용할 tool 함수를 반환 (LangChain bind_tools용)
    
    Note:
        읽기/검색 전용 툴만 포함.
        DB 수정(rdb_modify), 컬렉션 생성(vdb_create_collection), 
        포인트 업서트(vdb_upsert_points), 문서 인제스트(ingest_documents_tool),
        ECOS 파이프라인(ecos_run_pipeline)은 독립 스크립트로 사용.
    
    Returns:
        Tool 함수 리스트
    """
    from src.tools.rdb import (
        rdb_query_hard,
        rdb_query_llm,
        rdb_get_latest_ecos_date,
        user_info_get,
    )
    from src.tools.vdb import vdb_search
    from src.tools.web_search import web_search
    from src.tools.calculator_kcw import (
        compute_expected_fx_return,
        compute_optimal_hedge_weight,
        compute_log_returns_from_prices,
        compute_sigma_and_rho_from_returns,
        compute_fx_vol_and_rho_from_csv,
        compute_fx_vol_and_rho_from_rdb,
        compute_all,
    )
    
    return [
        # RDB 조회 도구
        rdb_query_hard,
        rdb_query_llm,
        rdb_get_latest_ecos_date,
        user_info_get,
        # VDB 검색 도구
        vdb_search,
        # 웹 검색 도구
        web_search,
        # Calculator 도구
        compute_expected_fx_return,
        compute_optimal_hedge_weight,
        compute_log_returns_from_prices,
        compute_sigma_and_rho_from_returns,
        compute_fx_vol_and_rho_from_csv,
        compute_fx_vol_and_rho_from_rdb,
        compute_all,
    ]


def get_all_tools_with_write() -> List[Any]:
    """
    모든 tool 함수를 반환 (쓰기 권한 포함, 특수 용도)
    
    Returns:
        Tool 함수 리스트 (읽기 + 쓰기)
    """
    from src.tools.rdb import (
        rdb_query_hard,
        rdb_query_llm,
        rdb_modify,
        rdb_get_latest_ecos_date,
        user_info_get,
        user_info_upsert,
    )
    from src.tools.vdb import (
        vdb_search,
        vdb_create_collection,
        vdb_upsert_points
    )
    from src.tools.web_search import web_search
    
    return [
        # RDB 도구
        rdb_query_hard,
        rdb_query_llm,
        rdb_modify,
        rdb_get_latest_ecos_date,
        user_info_get,
        user_info_upsert,
        # VDB 도구
        vdb_search,
        vdb_create_collection,
        vdb_upsert_points,
        # 웹 검색 도구
        web_search,
    ]


def bind_tools_to_llm(llm: Any, tools: Optional[List[Any]] = None) -> Any:
    """
    LangChain LLM에 tool을 바인딩 (LangChain 내장 함수 사용)
    
    Args:
        llm: LangChain LLM 인스턴스 (ChatOpenAI 등)
        tools: 바인딩할 tool 리스트 (None이면 모든 tool 사용)
        
    Returns:
        Tool이 바인딩된 LLM 인스턴스
    """
    if tools is None:
        tools = get_all_tools()
    
    try:
        # LangChain의 bind_tools 내장 함수 사용
        if hasattr(llm, 'bind_tools'):
            # bind_tools는 LangChain의 표준 메서드
            return llm.bind_tools(tools)
        else:
            logger.warning("⚠️  LLM이 bind_tools를 지원하지 않습니다.")
            return llm
    except Exception as e:
        logger.error(
            f"❌ Tool 바인딩 실패",
            {"error": str(e)},
            exc_info=True
        )
        return llm

