"""
전략 실행 Agent 예시 코드

- 뉴스 센티멘트 분석 결과 → signal_score
- notebook/yonju/df.csv → 환율 변동성(sigma_fx), 상관계수(rho)
- sigma_asset, lambda(위험회피도)는 임의 값
- 위 값들을 이용해서 최적 환헷지 비중 w_H* 계산
"""

from typing import Optional
import sys
import importlib.util
from pathlib import Path

import numpy as np

# calculator_kcw를 직접 로드 (__init__.py를 거치지 않음)
calculator_path = Path(__file__).parent / "calculator_kcw.py"
spec = importlib.util.spec_from_file_location("calculator_kcw", calculator_path)
calculator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calculator_module)

compute_expected_fx_return = calculator_module.compute_expected_fx_return
compute_optimal_hedge_weight = calculator_module.compute_optimal_hedge_weight
compute_fx_vol_and_rho_from_csv = calculator_module.compute_fx_vol_and_rho_from_csv

# news_sentimental_analysis를 직접 로드
news_sent_path = Path(__file__).parent / "news_sentimental_analysis.py"
spec2 = importlib.util.spec_from_file_location("news_sentimental_analysis", news_sent_path)
news_sent_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(news_sent_module)
run_sentiment_analysis = news_sent_module.run_sentiment_analysis


def run_strategy_execution(
    csv_path: str = "notebook/yonju/df.csv",
    fx_col: str = "usdkrw(target)",  # 실제 CSV 컬럼명
    asset_col: Optional[str] = None,
        sigma_asset_manual: float = 0.15,
        risk_aversion: float = 4.0,
        alpha: float = 0.0001,  # 일간 기준으로 적절한 값 (0.01% 수준)
) -> dict:
    """
    환헷지 전략 실행 함수.
    
    1) 뉴스 센티멘트 분석 결과 → signal_score
    2) df.csv → 환율 변동성(sigma_fx), 상관계수(rho)
    3) sigma_asset은 임의값 사용 (또는 나중에 실제 계산으로 대체)
    4) E[R_FX]와 w_H* 계산
    
    Parameters
    ----------
    csv_path : str
        환율 데이터가 들어있는 CSV 경로 (예: 'notebook/yonju/df.csv')
    fx_col : str
        환율 컬럼명 (예: 'usdkrw(target)')
    asset_col : str, optional
        자산 가격 컬럼명. 없으면 rho=0.0 가정.
    sigma_asset_manual : float
        임의로 설정하는 σ_Asset 값 (나중에 실제 계산으로 대체 가능)
    risk_aversion : float
        위험회피도 λ
    alpha : float
        시그널 → E[R_FX] 변환 스케일 파라미터
    
    Returns
    -------
    dict
        계산에 사용된 입력값 및 결과를 담은 딕셔너리
    """
    # 1) 뉴스 센티멘트 분석 → signal_score 가져오기
    # LLM 응답을 파싱해서 signal_score 추출 (간단한 예시)
    sentiment_result = run_sentiment_analysis()
    
    # LLM 응답이 문자열이므로 파싱 필요 (예: JSON 또는 리스트 형태)
    # 여기서는 간단히 USD 관련 impact의 평균을 signal_score로 사용
    # 실제로는 더 정교한 파싱 로직이 필요할 수 있음
    try:
        import ast
        import re
        # 문자열에서 리스트 추출 시도
        if isinstance(sentiment_result, str):
            # Python 리스트 형태인지 확인
            match = re.search(r'\[.*?\]', sentiment_result, re.DOTALL)
            if match:
                data = ast.literal_eval(match.group())
                # USD 관련 항목 찾기
                usd_items = [item for item in data if isinstance(item, dict) and item.get("currency") == "USD"]
                if usd_items:
                    # USD의 impact 평균을 signal_score로 사용
                    # direction이 "up"이면 양수, "down"이면 음수
                    impacts = []
                    for item in usd_items:
                        impact = item.get("impact", 0.0)
                        if isinstance(impact, (int, float)):
                            direction = item.get("direction", "neutral")
                            if direction == "down":
                                impact = -impact
                            impacts.append(impact)
                    signal_score = sum(impacts) / len(impacts) if impacts else 0.0
                else:
                    signal_score = 0.0
            else:
                signal_score = 0.0
        else:
            signal_score = 0.0
    except Exception as e:
        print(f"Warning: Failed to parse sentiment result: {e}")
        signal_score = 0.0
    
    # 2) df.csv에서 sigma_fx, rho 계산
    sigma_fx, rho = compute_fx_vol_and_rho_from_csv(
        csv_path=csv_path,
        fx_col=fx_col,
        asset_col=asset_col,
    )
    
    # 3) sigma_asset은 일단 수동 입력 값 사용 (나중에 실제 자산 데이터로 대체 가능)
    sigma_asset = float(sigma_asset_manual)
    
    # 4) E[R_FX] 계산
    expected_fx_return = compute_expected_fx_return(
        signal_score=signal_score,
        alpha=alpha,
    )
    
    # 5) 최적 환헷지 비중 w_H* 계산
    w_H_star = compute_optimal_hedge_weight(
        sigma_asset=sigma_asset,
        sigma_fx=sigma_fx,
        rho=rho,
        expected_fx_return=expected_fx_return,
        risk_aversion=risk_aversion,
        clip=True,
        debug=True,  # 디버깅 모드 활성화
    )
    
    # 6) 결과 정리
    result = {
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
    
    return result


if __name__ == "__main__":
    res = run_strategy_execution(
        csv_path="notebook/yonju/df.csv",
        fx_col="usdkrw(target)",  # 실제 컬럼명
        asset_col=None,       # df.csv에 자산 가격 컬럼 있으면 이름 넣으면 됨
        sigma_asset_manual=0.15,
        risk_aversion=4.0,
        alpha=0.0001,  # 일간 기준으로 적절한 값 (0.01% 수준)
    )
    
    print("=== 환헷지 전략 실행 결과 ===")
    for k, v in res.items():
        print(f"{k:>20}: {v}")

