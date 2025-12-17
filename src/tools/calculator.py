"""환헷지 계산 도구"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import os
from dotenv import load_dotenv
from src.utils.tools import (
    tool,
    handle_tool_error,
    ComputeExpectedFxReturnInput,
    ComputeOptimalHedgeWeightInput,
    ComputeLogReturnsFromPricesInput,
    ComputeSigmaAndRhoFromReturnsInput,
    ComputeFxVolAndRhoFromCsvInput,
    ComputeFxVolAndRhoFromRdbInput,
    ComputeAllInput,
)
from src.utils.logger import get_logger

logger = get_logger("calculator-tool")


class CalculatorTool:
    """환헷지 계산을 수행하는 도구"""
    
    def __init__(self):
        """CalculatorTool 초기화"""
        pass
    
    def compute_expected_fx_return(
        self, 
        signal_score: float, 
        alpha: float = 0.05
    ) -> float:
        """
        환율 기대수익률 E[R_FX]를 계산하는 tool.
        
        E[R_FX] = S * alpha
        
        Args:
            signal_score: 환율 시그널 점수 S (범위: -1 ~ +1, -1 = 강한 달러 약세, +1 = 강한 달러 강세)
            alpha: 스케일링 파라미터 (기본값: 0.05)
            
        Returns:
            E[R_FX] (기대 환율 수익률)
        """
        # 안전장치: S를 -1 ~ 1 사이로 클리핑
        s_clipped = max(-1.0, min(1.0, signal_score))
        return s_clipped * alpha
    
    def compute_optimal_hedge_weight(
        self,
        sigma_asset: float,
        sigma_fx: float,
        rho: float,
        expected_fx_return: float,
        risk_aversion: float,
        clip: bool = True,
        debug: bool = False
    ) -> Optional[float]:
        """
        최적 환헷지 비중 w_H^*을 계산하는 tool.
        
        w_H^* = (1 + rho * sigma_asset / sigma_fx) - E[R_FX] / (lambda * sigma_fx^2)
        
        Args:
            sigma_asset: 해외자산 수익률 변동성 σ_Asset (예: S&P500 일간 로그수익률 표준편차)
            sigma_fx: 환율 수익률 변동성 σ_FX (예: USD/KRW 일간 로그수익률 표준편차)
            rho: 자산 수익률과 환율 수익률의 상관계수 (Corr(R_asset, R_FX))
            expected_fx_return: 기대 환율 수익률 E[R_FX]
            risk_aversion: 위험회피도 λ (값이 클수록 보수적, 일반적으로 1 ~ 10 정도)
            clip: True면 결과를 [0, 1] 범위로 클리핑
            debug: True면 중간 계산값들을 출력
            
        Returns:
            최적 환헷지 비중 w_H^* (0 ~ 1 사이), 
            sigma_fx 또는 risk_aversion이 0 이하인 경우 None 반환
        """
        # 방어 코드: 위험한 입력값 처리
        if sigma_fx <= 0:
            if debug:
                print(f"DEBUG: sigma_fx <= 0: {sigma_fx}")
            return None
        
        if risk_aversion <= 0:
            if debug:
                print(f"DEBUG: risk_aversion <= 0: {risk_aversion}")
            return None
        
        # Minimum Variance Term
        rho_ratio = rho * (sigma_asset / sigma_fx)
        mv_term = 1.0 + rho_ratio
        
        # Speculative Term
        # sigma_fx가 너무 작으면 spec_term이 폭발할 수 있으므로 안전장치 추가
        denominator = risk_aversion * (sigma_fx ** 2)
        spec_term_raw = expected_fx_return / denominator  # 실제 계산값
        
        if abs(expected_fx_return) > abs(mv_term * denominator):
            # spec_term이 mv_term보다 크면 w_H*가 음수가 됨
            # 이 경우 spec_term을 제한하거나 경고
            if debug:
                print(f"  ⚠️ 안전장치 적용: spec_term이 mv_term보다 큼")
                print(f"    spec_term (실제) = {spec_term_raw}")
                print(f"    spec_term (제한) = {mv_term} * 0.5 = {mv_term * 0.5}")
            # spec_term을 mv_term의 일정 비율로 제한 (예: 0.5배)
            spec_term = mv_term * 0.5
            spec_term_limited = True
        else:
            spec_term = spec_term_raw
            spec_term_limited = False
        
        # 최종 w_H*
        w_h_star = mv_term - spec_term
        w_h_star_raw = w_h_star  # 클리핑 전 값 저장
        
        if debug:
            print("\n=== w_H* 계산 디버깅 ===")
            print(f"입력값:")
            print(f"  sigma_asset = {sigma_asset}")
            print(f"  sigma_fx = {sigma_fx}")
            print(f"  rho = {rho}")
            print(f"  expected_fx_return = {expected_fx_return}")
            print(f"  risk_aversion = {risk_aversion}")
            print(f"\n중간 계산:")
            print(f"  rho * (sigma_asset / sigma_fx) = {rho} * ({sigma_asset} / {sigma_fx}) = {rho_ratio}")
            print(f"  mv_term = 1.0 + {rho_ratio} = {mv_term}")
            print(f"  denominator = {risk_aversion} * ({sigma_fx} ** 2) = {denominator}")
            if spec_term_limited:
                print(f"  spec_term (실제 계산) = {expected_fx_return} / {denominator} = {spec_term_raw}")
                print(f"  spec_term (안전장치 적용) = {mv_term} * 0.5 = {spec_term}")
            else:
                print(f"  spec_term = {expected_fx_return} / {denominator} = {spec_term}")
            print(f"\n결과:")
            print(f"  w_H* (raw) = {mv_term} - {spec_term} = {w_h_star_raw}")
        
        # 0 ~ 1 사이로 클리핑
        if clip:
            w_h_star = max(0.0, min(1.0, w_h_star))
            if debug:
                print(f"  w_H* (clipped) = {w_h_star}")
                if w_h_star_raw != w_h_star:
                    print(f"  ⚠️ 클리핑됨: {w_h_star_raw} → {w_h_star}")
        
        return w_h_star
    
    def compute_log_returns_from_prices(
        self,
        prices: pd.Series
    ) -> np.ndarray:
        """
        가격 시계열로부터 로그수익률(log return)을 계산하는 tool.
        
        R_t = ln(P_t / P_{t-1})
        
        Args:
            prices: 가격 시계열 (pandas Series)
            
        Returns:
            로그수익률 배열
        """
        # 결측치 제거 및 float 변환
        clean = pd.to_numeric(prices, errors="coerce").dropna().values
        if len(clean) < 2:
            raise ValueError("가격 데이터가 너무 적어서 수익률을 계산할 수 없습니다 (len < 2).")
        
        returns = np.log(clean[1:] / clean[:-1])
        return returns
    
    def compute_sigma_and_rho_from_returns(
        self,
        asset_returns: np.ndarray,
        fx_returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        자산 수익률과 환율 수익률로부터 
        σ_Asset, σ_FX, ρ를 계산하는 tool.
        
        Args:
            asset_returns: 자산(예: S&P500) 일간 수익률 시계열
            fx_returns: 환율(예: USD/KRW) 일간 수익률 시계열
            
        Returns:
            (sigma_asset, sigma_fx, rho) 튜플
        """
        # 길이 맞추기 (둘 중 더 짧은 길이에 맞춤)
        n = min(len(asset_returns), len(fx_returns))
        if n < 2:
            raise ValueError("수익률 데이터 길이가 너무 짧습니다 (len < 2).")
        
        a = np.asarray(asset_returns[:n], dtype=float)
        f = np.asarray(fx_returns[:n], dtype=float)
        
        sigma_asset = float(np.std(a, ddof=1))  # 표본 표준편차
        sigma_fx = float(np.std(f, ddof=1))
        
        # 상관계수 ρ
        if np.allclose(a, a[0]) or np.allclose(f, f[0]):
            # 둘 중 하나라도 상수면 상관계수 정의가 애매하므로 0으로 처리
            rho = 0.0
        else:
            rho = float(np.corrcoef(a, f)[0, 1])
        
        return sigma_asset, sigma_fx, rho
    
    def compute_fx_vol_and_rho_from_csv(
        self,
        csv_path: Optional[str] = None,
        fx_col: str = "usdkrw(target)",
        asset_col: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, float]:
        """
        CSV 파일 또는 DataFrame에서 환율 변동성 σ_FX와 상관계수 ρ를 계산하는 tool.
        
        Args:
            csv_path: CSV 파일 경로 (예: 'notebook/yonju/df.csv'). df가 제공되면 무시됨.
            fx_col: 환율 컬럼명 (예: 'usdkrw(target)')
            asset_col: 자산(예: S&P500) 가격 또는 수익률 컬럼명.
                      None이면 ρ는 0.0으로 반환.
            df: 직접 제공하는 DataFrame. 제공되면 csv_path는 무시됨.
            
        Returns:
            (sigma_fx, rho) 튜플
            sigma_fx: 환율 수익률 변동성
            rho: 자산-환율 상관계수 (asset_col이 없으면 0.0)
        """
        if df is None:
            if csv_path is None:
                raise ValueError("csv_path 또는 df 중 하나는 제공되어야 합니다.")
            df = pd.read_csv(csv_path)
        
        if fx_col not in df.columns:
            raise KeyError(f"CSV에 '{fx_col}' 컬럼이 없습니다. 실제 컬럼명을 확인해주세요.")
        
        fx_prices = df[fx_col]
        fx_returns = self.compute_log_returns_from_prices(fx_prices)
        
        sigma_fx = float(np.std(fx_returns, ddof=1))
        
        # asset_col이 없으면 rho는 0으로 반환 (중립 가정)
        if asset_col is None or asset_col not in df.columns:
            rho = 0.0
            return sigma_fx, rho
        
        asset_prices = df[asset_col]
        asset_returns = self.compute_log_returns_from_prices(asset_prices)
        
        _, _, rho = self.compute_sigma_and_rho_from_returns(asset_returns, fx_returns)
        
        return sigma_fx, rho
    
    def compute_fx_vol_and_rho_from_rdb(
        self,
        days: int = 252,
        asset_col: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        RDB에서 환율 데이터를 가져와서 변동성과 상관계수를 계산하는 tool.
        
        Args:
            days: 조회할 최근 일수 (기본값: 252일, 약 1년)
            asset_col: 자산(예: S&P500) 가격 또는 수익률 컬럼명.
                      None이면 ρ는 0.0으로 반환.
            
        Returns:
            (sigma_fx, rho) 튜플
            sigma_fx: 환율 수익률 변동성
            rho: 자산-환율 상관계수 (asset_col이 없으면 0.0)
        """
        # .env 로드
        load_dotenv()
        
        # DB 연결 정보 가져오기
        try:
            import pymysql
            
            db_host = os.getenv("DATABASE_HOST") or os.getenv("DB_HOST")
            db_port = int(os.getenv("DATABASE_PORT") or os.getenv("DB_PORT", 3306))
            db_user = os.getenv("DATABASE_USER") or os.getenv("DB_USER")
            db_password = os.getenv("DATABASE_PASSWORD") or os.getenv("DB_PASSWORD")
            db_name = os.getenv("DATABASE_NAME") or os.getenv("DB_NAME")
            
            if not all([db_host, db_user, db_password, db_name]):
                raise ValueError("데이터베이스 연결 정보가 .env에 없습니다.")
            
            # DB 연결
            conn = pymysql.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor
            )
            
            # 최근 N일 환율 데이터 조회
            query = """
                SELECT date, usdkrw
                FROM eiExchangeRate
                ORDER BY date DESC
                LIMIT %s
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, (days,))
                results = cursor.fetchall()
            
            conn.close()
            
            if not results:
                raise ValueError(f"RDB에서 환율 데이터를 찾을 수 없습니다. (최근 {days}일)")
            
            # DataFrame으로 변환 (날짜 역순이므로 역순으로 정렬)
            df = pd.DataFrame(results)
            df = df.sort_values('date')  # 날짜 오름차순으로 정렬
            df = df.reset_index(drop=True)
            
            # usdkrw 컬럼명을 'usdkrw(target)'로 변경 (기존 CSV와 호환)
            if 'usdkrw' in df.columns:
                df['usdkrw(target)'] = df['usdkrw']
            
            # 기존 CSV 메서드 재사용
            return self.compute_fx_vol_and_rho_from_csv(
                csv_path=None,  # 사용하지 않음
                fx_col="usdkrw(target)",
                asset_col=asset_col,
                df=df  # DataFrame 직접 전달
            )
            
        except ImportError:
            raise ImportError("pymysql이 설치되지 않았습니다. pip install pymysql")
        except Exception as e:
            raise RuntimeError(f"RDB에서 데이터를 가져오는 중 오류 발생: {str(e)}")
    
    def compute_all(
        self,
        signal_score: float,
        csv_path: str,
        sigma_asset: float = 0.15,
        risk_aversion: float = 4.0,
        alpha: float = 0.05,
        fx_col: str = "usdkrw(target)",
        asset_col: Optional[str] = None,
        clip: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        모든 계산을 한 번에 수행하는 통합 tool.
        
        Args:
            signal_score: 환율 시그널 점수
            csv_path: CSV 파일 경로
            sigma_asset: 해외자산 수익률 변동성
            risk_aversion: 위험회피도
            alpha: 스케일링 파라미터
            fx_col: 환율 컬럼명
            asset_col: 자산 가격 컬럼명
            clip: True면 결과를 [0, 1] 범위로 클리핑
            debug: True면 중간 계산값들을 출력
            
        Returns:
            계산 결과를 담은 딕셔너리
        """
        # 1. expected_fx_return 계산
        expected_fx_return = self.compute_expected_fx_return(signal_score, alpha)
        
        # 2. sigma_fx, rho 계산
        sigma_fx, rho = self.compute_fx_vol_and_rho_from_csv(
            csv_path, fx_col, asset_col
        )
        
        # 3. w_H* 계산
        w_H_star = self.compute_optimal_hedge_weight(
            sigma_asset, sigma_fx, rho, expected_fx_return,
            risk_aversion, clip, debug
        )
        
        return {
            "signal_score": signal_score,
            "expected_fx_return": expected_fx_return,
            "sigma_asset": sigma_asset,
            "sigma_fx": sigma_fx,
            "rho": rho,
            "risk_aversion": risk_aversion,
            "alpha": alpha,
            "w_H_star": w_H_star,
            "w_UH_star": None if w_H_star is None else (1.0 - w_H_star),
        }


# ============================================================================
# Tool 함수들 (LangChain @tool decorator 사용)
# ============================================================================

@tool(args_schema=ComputeExpectedFxReturnInput)
@handle_tool_error("compute_expected_fx_return")
async def compute_expected_fx_return(
    signal_score: float,
    alpha: float = 0.05
) -> float:
    """
    환율 기대수익률 E[R_FX]를 계산하는 tool.
    
    E[R_FX] = S * alpha
    
    Args:
        signal_score: 환율 시그널 점수 S (범위: -1 ~ +1, -1 = 강한 달러 약세, +1 = 강한 달러 강세)
        alpha: 스케일링 파라미터 (기본값: 0.05)
        
    Returns:
        E[R_FX] (기대 환율 수익률)
    """
    calculator = CalculatorTool()
    result = calculator.compute_expected_fx_return(signal_score, alpha)
    logger.info(
        f"🔧 [TOOL CALL] compute_expected_fx_return 실행",
        {"signal_score": signal_score, "alpha": alpha, "result": result}
    )
    return result


@tool(args_schema=ComputeOptimalHedgeWeightInput)
@handle_tool_error("compute_optimal_hedge_weight")
async def compute_optimal_hedge_weight(
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    expected_fx_return: float,
    risk_aversion: float,
    clip: bool = True
) -> Optional[float]:
    """
    최적 환헷지 비중 w_H^*을 계산하는 tool.
    
    w_H^* = (1 + rho * sigma_asset / sigma_fx) - E[R_FX] / (lambda * sigma_fx^2)
    
    Args:
        sigma_asset: 해외자산 수익률 변동성 σ_Asset (예: S&P500 일간 로그수익률 표준편차)
        sigma_fx: 환율 수익률 변동성 σ_FX (예: USD/KRW 일간 로그수익률 표준편차)
        rho: 자산 수익률과 환율 수익률의 상관계수 (Corr(R_asset, R_FX))
        expected_fx_return: 기대 환율 수익률 E[R_FX]
        risk_aversion: 위험회피도 λ (값이 클수록 보수적, 일반적으로 1 ~ 10 정도)
        clip: True면 결과를 [0, 1] 범위로 클리핑
        
    Returns:
        최적 환헷지 비중 w_H^* (0 ~ 1 사이), 
        sigma_fx 또는 risk_aversion이 0 이하인 경우 None 반환
    """
    calculator = CalculatorTool()
    result = calculator.compute_optimal_hedge_weight(
        sigma_asset, sigma_fx, rho, expected_fx_return,
        risk_aversion, clip, debug=False
    )
    logger.info(
        f"🔧 [TOOL CALL] compute_optimal_hedge_weight 실행",
        {
            "sigma_asset": sigma_asset,
            "sigma_fx": sigma_fx,
            "rho": rho,
            "expected_fx_return": expected_fx_return,
            "risk_aversion": risk_aversion,
            "result": result
        }
    )
    return result


@tool(args_schema=ComputeLogReturnsFromPricesInput)
@handle_tool_error("compute_log_returns_from_prices")
async def compute_log_returns_from_prices(
    prices: list
) -> list:
    """
    가격 시계열로부터 로그수익률(log return)을 계산하는 tool.
    
    R_t = ln(P_t / P_{t-1})
    
    Args:
        prices: 가격 시계열 (리스트)
        
    Returns:
        로그수익률 배열 (리스트로 변환)
    """
    calculator = CalculatorTool()
    prices_series = pd.Series(prices)
    result = calculator.compute_log_returns_from_prices(prices_series)
    logger.info(
        f"🔧 [TOOL CALL] compute_log_returns_from_prices 실행",
        {"prices_count": len(prices), "returns_count": len(result)}
    )
    return result.tolist()


@tool(args_schema=ComputeSigmaAndRhoFromReturnsInput)
@handle_tool_error("compute_sigma_and_rho_from_returns")
async def compute_sigma_and_rho_from_returns(
    asset_returns: list,
    fx_returns: list
) -> Dict[str, float]:
    """
    자산 수익률과 환율 수익률로부터 
    σ_Asset, σ_FX, ρ를 계산하는 tool.
    
    Args:
        asset_returns: 자산(예: S&P500) 일간 수익률 시계열
        fx_returns: 환율(예: USD/KRW) 일간 수익률 시계열
        
    Returns:
        {"sigma_asset": float, "sigma_fx": float, "rho": float} 딕셔너리
    """
    calculator = CalculatorTool()
    asset_arr = np.array(asset_returns)
    fx_arr = np.array(fx_returns)
    sigma_asset, sigma_fx, rho = calculator.compute_sigma_and_rho_from_returns(asset_arr, fx_arr)
    result = {
        "sigma_asset": sigma_asset,
        "sigma_fx": sigma_fx,
        "rho": rho
    }
    logger.info(
        f"🔧 [TOOL CALL] compute_sigma_and_rho_from_returns 실행",
        result
    )
    return result


@tool(args_schema=ComputeFxVolAndRhoFromCsvInput)
@handle_tool_error("compute_fx_vol_and_rho_from_csv")
async def compute_fx_vol_and_rho_from_csv(
    csv_path: Optional[str] = None,
    fx_col: str = "usdkrw(target)",
    asset_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    CSV 파일 또는 DataFrame에서 환율 변동성 σ_FX와 상관계수 ρ를 계산하는 tool.
    
    Args:
        csv_path: CSV 파일 경로 (예: 'notebook/yonju/df.csv'). df가 제공되면 무시됨.
        fx_col: 환율 컬럼명 (예: 'usdkrw(target)')
        asset_col: 자산(예: S&P500) 가격 또는 수익률 컬럼명.
                  None이면 ρ는 0.0으로 반환.
        
    Returns:
        {"sigma_fx": float, "rho": float} 딕셔너리
    """
    calculator = CalculatorTool()
    sigma_fx, rho = calculator.compute_fx_vol_and_rho_from_csv(
        csv_path=csv_path,
        fx_col=fx_col,
        asset_col=asset_col,
        df=None
    )
    result = {
        "sigma_fx": sigma_fx,
        "rho": rho
    }
    logger.info(
        f"🔧 [TOOL CALL] compute_fx_vol_and_rho_from_csv 실행",
        {"csv_path": csv_path, "fx_col": fx_col, "asset_col": asset_col, **result}
    )
    return result


@tool(args_schema=ComputeFxVolAndRhoFromRdbInput)
@handle_tool_error("compute_fx_vol_and_rho_from_rdb")
async def compute_fx_vol_and_rho_from_rdb(
    days: int = 252,
    asset_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    RDB에서 환율 데이터를 가져와서 변동성과 상관계수를 계산하는 tool.
    
    Args:
        days: 조회할 최근 일수 (기본값: 252일, 약 1년)
        asset_col: 자산(예: S&P500) 가격 또는 수익률 컬럼명.
                  None이면 ρ는 0.0으로 반환.
        
    Returns:
        {"sigma_fx": float, "rho": float} 딕셔너리
    """
    calculator = CalculatorTool()
    sigma_fx, rho = calculator.compute_fx_vol_and_rho_from_rdb(
        days=days,
        asset_col=asset_col
    )
    result = {
        "sigma_fx": sigma_fx,
        "rho": rho
    }
    logger.info(
        f"🔧 [TOOL CALL] compute_fx_vol_and_rho_from_rdb 실행",
        {"days": days, "asset_col": asset_col, **result}
    )
    return result


@tool(args_schema=ComputeAllInput)
@handle_tool_error("compute_all")
async def compute_all(
    signal_score: float,
    csv_path: str,
    sigma_asset: float = 0.15,
    risk_aversion: float = 4.0,
    alpha: float = 0.05,
    fx_col: str = "usdkrw(target)",
    asset_col: Optional[str] = None,
    clip: bool = True
) -> Dict[str, Any]:
    """
    모든 계산을 한 번에 수행하는 통합 tool.
    
    Args:
        signal_score: 환율 시그널 점수
        csv_path: CSV 파일 경로
        sigma_asset: 해외자산 수익률 변동성
        risk_aversion: 위험회피도
        alpha: 스케일링 파라미터
        fx_col: 환율 컬럼명
        asset_col: 자산 가격 컬럼명
        clip: True면 결과를 [0, 1] 범위로 클리핑
        
    Returns:
        계산 결과를 담은 딕셔너리
    """
    calculator = CalculatorTool()
    result = calculator.compute_all(
        signal_score=signal_score,
        csv_path=csv_path,
        sigma_asset=sigma_asset,
        risk_aversion=risk_aversion,
        alpha=alpha,
        fx_col=fx_col,
        asset_col=asset_col,
        clip=clip,
        debug=False
    )
    logger.info(
        f"🔧 [TOOL CALL] compute_all 실행",
        {"signal_score": signal_score, "csv_path": csv_path, "w_H_star": result.get("w_H_star")}
    )
    return result
