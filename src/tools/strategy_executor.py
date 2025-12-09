"""
전략 실행 Agent 예시 코드

- 뉴스 센티멘트 분석 결과 → signal_score
- RDB (eiExchangeRate 테이블) → 환율 변동성(sigma_fx), 상관계수(rho)
- sigma_asset, lambda(위험회피도)는 임의 값
- 위 값들을 이용해서 최적 환헷지 비중 w_H* 계산
"""

from typing import Optional
import sys
import importlib.util
from pathlib import Path

import numpy as np

# CalculatorTool을 직접 로드 (__init__.py를 거치지 않음)
calculator_path = Path(__file__).parent / "calculator_kcw.py"
spec = importlib.util.spec_from_file_location("calculator_kcw", calculator_path)
calculator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calculator_module)

CalculatorTool = calculator_module.CalculatorTool

# news_sentimental_analysis를 직접 로드
news_sent_path = Path(__file__).parent / "news_sentimental_analysis.py"
spec2 = importlib.util.spec_from_file_location("news_sentimental_analysis", news_sent_path)
news_sent_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(news_sent_module)
run_sentiment_analysis = news_sent_module.run_sentiment_analysis


def get_user_risk_aversion(user_id: Optional[str] = None) -> float:
    """
    RDB의 user_info 테이블에서 user_risk_aversion 값을 가져옵니다.
    
    Args:
        user_id: 사용자 ID. None이면 기본값 4.0 반환
        
    Returns:
        user_risk_aversion 값 (없으면 4.0)
    """
    if user_id is None:
        return 4.0
    
    try:
        import pymysql
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        db_host = os.getenv("DATABASE_HOST") or os.getenv("DB_HOST")
        db_port = int(os.getenv("DATABASE_PORT") or os.getenv("DB_PORT", 3306))
        db_user = os.getenv("DATABASE_USER") or os.getenv("DB_USER")
        db_password = os.getenv("DATABASE_PASSWORD") or os.getenv("DB_PASSWORD")
        db_name = os.getenv("DATABASE_NAME") or os.getenv("DB_NAME")
        
        if not all([db_host, db_user, db_password, db_name]):
            print("⚠️  데이터베이스 연결 정보가 없어 기본값 4.0을 사용합니다.")
            return 4.0
        
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
        
        # user_risk_aversion 조회
        query = """
            SELECT user_risk_aversion
            FROM user_info
            WHERE user_id = %s
            LIMIT 1
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
        
        conn.close()
        
        if result and result.get("user_risk_aversion") is not None:
            risk_aversion = float(result["user_risk_aversion"])
            print(f"✅ 사용자 {user_id}의 risk_aversion: {risk_aversion}")
            return risk_aversion
        else:
            print(f"⚠️  사용자 {user_id}의 risk_aversion 정보가 없어 기본값 4.0을 사용합니다.")
            return 4.0
            
    except ImportError:
        print("⚠️  pymysql이 설치되지 않아 기본값 4.0을 사용합니다.")
        return 4.0
    except Exception as e:
        print(f"⚠️  RDB에서 risk_aversion을 가져오는 중 오류 발생: {e}")
        print("   기본값 4.0을 사용합니다.")
        return 4.0


def run_strategy_execution(
    user_id: Optional[str] = None,
    asset_col: Optional[str] = None,
    sigma_asset_manual: float = 0.15,
    risk_aversion: Optional[float] = None,  # None이면 user_id에서 가져오거나 기본값 4.0
    alpha: float = 0.0001,  # 일간 기준으로 적절한 값 (0.01% 수준)
    rdb_days: int = 252,  # RDB에서 가져올 최근 일수
) -> dict:
    """
    환헷지 전략 실행 함수.
    
    1) 뉴스 센티멘트 분석 결과 → signal_score
    2) RDB → 환율 변동성(sigma_fx), 상관계수(rho)
    3) sigma_asset은 임의값 사용 (또는 나중에 실제 계산으로 대체)
    4) E[R_FX]와 w_H* 계산
    
    Parameters
    ----------
    user_id : str, optional
        사용자 ID. 제공되면 RDB에서 user_risk_aversion을 가져옵니다.
    asset_col : str, optional
        자산 가격 컬럼명. 없으면 rho=0.0 가정.
    sigma_asset_manual : float
        임의로 설정하는 σ_Asset 값 (나중에 실제 계산으로 대체 가능)
    risk_aversion : float, optional
        위험회피도 λ. None이면 user_id에서 가져오거나 기본값 4.0 사용
    alpha : float
        시그널 → E[R_FX] 변환 스케일 파라미터
    rdb_days : int
        RDB에서 가져올 최근 일수 (기본값: 252일)
    
    Returns
    -------
    dict
        계산에 사용된 입력값 및 결과를 담은 딕셔너리
    """
    # 1) 뉴스 센티멘트 분석 → signal_score 가져오기
    # LLM 응답을 파싱해서 USD의 signal_score 추출
    sentiment_result = run_sentiment_analysis()
    
    # 기본값: 실패 시 0.5로 고정
    signal_score = 0.5
    
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
                    # USD의 signal_score 가져오기 (있으면 사용, 없으면 impact 사용)
                    for item in usd_items:
                        # signal_score 필드가 있으면 사용
                        if "signal_score" in item and isinstance(item.get("signal_score"), (int, float)):
                            signal_score = float(item["signal_score"])
                            break
                        # signal_score가 없으면 impact를 사용 (direction에 따라 부호 조정)
                        elif "impact" in item and isinstance(item.get("impact"), (int, float)):
                            impact = float(item.get("impact", 0.0))
                            direction = item.get("direction", "neutral")
                            if direction == "down":
                                signal_score = -impact
                            elif direction == "up":
                                signal_score = impact
                            else:
                                signal_score = impact
                            break
    except Exception as e:
        print(f"Warning: Failed to parse sentiment result: {e}")
        signal_score = 0.5  # 실패 시 0.5로 고정
    
    # CalculatorTool 인스턴스 생성
    calculator = CalculatorTool()
    
    # 1.5) risk_aversion 가져오기 (user_id가 있으면 RDB에서, 없으면 파라미터 또는 기본값)
    if risk_aversion is None:
        risk_aversion = get_user_risk_aversion(user_id)
    else:
        print(f"✅ risk_aversion 파라미터 사용: {risk_aversion}")
    
    # 2) RDB에서 sigma_fx, rho 계산
    sigma_fx, rho = calculator.compute_fx_vol_and_rho_from_rdb(
        days=rdb_days,
        asset_col=asset_col,
    )
    print(f"✅ RDB에서 환율 데이터를 가져왔습니다. (최근 {rdb_days}일)")
    
    # 3) sigma_asset은 일단 수동 입력 값 사용 (나중에 실제 자산 데이터로 대체 가능)
    sigma_asset = float(sigma_asset_manual)
    
    # 4) E[R_FX] 계산
    expected_fx_return = calculator.compute_expected_fx_return(
        signal_score=signal_score,
        alpha=alpha,
    )
    
    # 5) 최적 환헷지 비중 w_H* 계산
    w_H_star = calculator.compute_optimal_hedge_weight(
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
        user_id=None,  # 사용자 ID (예: "user123"). None이면 기본값 4.0 사용
        asset_col=None,       # RDB에 자산 가격 컬럼 있으면 이름 넣으면 됨
        sigma_asset_manual=0.15,
        risk_aversion=None,  # None이면 user_id에서 가져오거나 기본값 4.0
        alpha=0.0001,  # 일간 기준으로 적절한 값 (0.01% 수준)
        rdb_days=252,  # 최근 1년 데이터
    )
    
    print("=== 환헷지 전략 실행 결과 ===")
    for k, v in res.items():
        print(f"{k:>20}: {v}")