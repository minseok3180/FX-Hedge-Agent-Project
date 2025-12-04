import numpy as np
import pandas as pd
from typing import Optional, Tuple


def compute_expected_fx_return(signal_score: float, alpha: float = 0.05) -> float:
    """
    환율 기대수익률 E[R_FX]를 계산하는 함수.
    
    E[R_FX] = S * alpha
    
    Parameters
    ----------
    signal_score : float
        환율 시그널 점수 S (범위: -1 ~ +1, -1 = 강한 달러 약세, +1 = 강한 달러 강세)
    alpha : float, optional
        스케일링 파라미터 (예: 0.05 → 최대 ±5% 수준의 환율 기대수익 반영)
    
    Returns
    -------
    float
        E[R_FX] (기대 환율 수익률)
    """
    # 안전장치: S를 -1 ~ 1 사이로 클리핑
    s_clipped = max(-1.0, min(1.0, signal_score))
    return s_clipped * alpha


def compute_optimal_hedge_weight(
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    expected_fx_return: float,
    risk_aversion: float,
    clip: bool = True,
    debug: bool = False
) -> Optional[float]:
    """
    최적 환헷지 비중 w_H^*을 계산하는 함수.
    
    w_H^* = (1 + rho * sigma_asset / sigma_fx) - E[R_FX] / (lambda * sigma_fx^2)
    
    Parameters
    ----------
    sigma_asset : float
        해외자산 수익률 변동성 σ_Asset (예: S&P500 일간 로그수익률 표준편차)
    sigma_fx : float
        환율 수익률 변동성 σ_FX (예: USD/KRW 일간 로그수익률 표준편차)
    rho : float
        자산 수익률과 환율 수익률의 상관계수 (Corr(R_asset, R_FX))
    expected_fx_return : float
        기대 환율 수익률 E[R_FX]
    risk_aversion : float
        위험회피도 λ (값이 클수록 보수적, 일반적으로 1 ~ 10 정도)
    clip : bool, optional
        True면 결과를 [0, 1] 범위로 클리핑
    debug : bool, optional
        True면 중간 계산값들을 출력
    
    Returns
    -------
    float or None
        최적 환헷지 비중 w_H^* (0 ~ 1 사이), 
        sigma_fx 또는 risk_aversion이 0 이하인 경우 None 반환
    """
    # 방어 코드: 위험한 입력값 처리
    if sigma_fx <= 0:
        # 환율 변동성이 0이면 이론적으로는 헷지가 의미 없다고 볼 수 있음
        # 정책적으로 0 또는 1을 반환하도록 바꿀 수도 있음
        if debug:
            print(f"DEBUG: sigma_fx <= 0: {sigma_fx}")
        return None
    
    if risk_aversion <= 0:
        # λ <= 0이면 유틸리티 함수가 성립하지 않으므로 계산 불가
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


# ============================
# 🔹 변동성(σ) / 상관계수(ρ) 계산 함수
# ============================

def compute_log_returns_from_prices(prices: pd.Series) -> np.ndarray:
    """
    가격 시계열로부터 로그수익률(log return)을 계산하는 유틸 함수.
    
    R_t = ln(P_t / P_{t-1})
    """
    # 결측치 제거 및 float 변환
    clean = pd.to_numeric(prices, errors="coerce").dropna().values
    if len(clean) < 2:
        raise ValueError("가격 데이터가 너무 적어서 수익률을 계산할 수 없습니다 (len < 2).")
    
    returns = np.log(clean[1:] / clean[:-1])
    return returns


def compute_sigma_and_rho_from_returns(
    asset_returns: np.ndarray,
    fx_returns: np.ndarray
) -> Tuple[float, float, float]:
    """
    자산 수익률과 환율 수익률로부터 
    σ_Asset, σ_FX, ρ를 계산하는 함수.
    
    Parameters
    ----------
    asset_returns : np.ndarray
        자산(예: S&P500) 일간 수익률 시계열
    fx_returns : np.ndarray
        환율(예: USD/KRW) 일간 수익률 시계열
    
    Returns
    -------
    (sigma_asset, sigma_fx, rho)
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
    csv_path: str,
    fx_col: str = "usdkrw(target)",  # 실제 CSV 컬럼명
    asset_col: Optional[str] = None
) -> Tuple[float, float]:
    """
    notebook/yonju/df.csv 같은 CSV에서 
    환율 변동성 σ_FX와 (옵션) 상관계수 ρ를 계산하는 함수.
    
    Parameters
    ----------
    csv_path : str
        CSV 파일 경로 (예: 'notebook/yonju/df.csv')
    fx_col : str, optional
        환율 컬럼명 (예: 'usdkrw' 또는 'target' 등)
    asset_col : str, optional
        자산(예: S&P500) 가격 또는 수익률 컬럼명.
        None이면 ρ는 0.0으로 반환.
    
    Returns
    -------
    (sigma_fx, rho)
        sigma_fx : 환율 수익률 변동성
        rho : 자산-환율 상관계수 (asset_col이 없으면 0.0)
    """
    df = pd.read_csv(csv_path)
    
    if fx_col not in df.columns:
        raise KeyError(f"CSV에 '{fx_col}' 컬럼이 없습니다. 실제 컬럼명을 확인해주세요.")
    
    fx_prices = df[fx_col]
    fx_returns = compute_log_returns_from_prices(fx_prices)
    
    sigma_fx = float(np.std(fx_returns, ddof=1))
    
    # asset_col이 없으면 rho는 0으로 반환 (중립 가정)
    if asset_col is None or asset_col not in df.columns:
        rho = 0.0
        return sigma_fx, rho
    
    asset_prices = df[asset_col]
    asset_returns = compute_log_returns_from_prices(asset_prices)
    
    _, _, rho = compute_sigma_and_rho_from_returns(asset_returns, fx_returns)
    
    return sigma_fx, rho
