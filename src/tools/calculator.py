import pandas as pd
import numpy as np
import json
from langchain_core.tools import tool
from tools.rdb import rdb_query_hard 

@tool
async def calculate_expected_fx_return() -> dict:
    """
    (인자 없음)
    DB에 저장된 가장 최신의 거시경제 지표(금리, DXY, VIX 등)를 'get_latest' 쿼리로 조회하여,
    자체 로직에 따라 향후 환율 예상 수익률(View)을 계산합니다.
    """
    try:
        # 1. DB 툴 호출 (비동기 방식)
        # rdb_hard_queries.py에 정의된 "get_latest" 쿼리 실행 (LIMIT 1)
        result_list = await rdb_query_hard.ainvoke({
            "query_key": "get_latest",
            "params": (1,)  # 튜플 형태 필수
        })
        
        # 2. 데이터 유효성 체크
        if not result_list:
            return {"error": "DB에서 데이터를 찾을 수 없습니다."}
            
        current_data = result_list[0] # 가장 최신 데이터 1줄
        
        # 3. Scoring Logic (뇌)
        score = 0.0
        details = []
        
        # (A) 금리차 분석 (DB 컬럼명: us_interest, base)
        # 데이터가 없을 경우(None)를 대비해 안전하게 처리
        try:
            us_rate = float(current_data.get('us_interest') or 0)
            kr_rate = float(current_data.get('base') or 0)
            
            if us_rate > 0 and kr_rate > 0:
                rate_diff = us_rate - kr_rate
                if rate_diff > 2.0:
                    score += 0.03
                    details.append(f"금리차 확대({rate_diff:.2f}%p) -> 달러 강세 요인(+3%)")
                elif rate_diff > 1.0:
                    score += 0.01
                    details.append(f"금리차 유지({rate_diff:.2f}%p) -> 달러 강세 요인(+1%)")
        except:
            details.append("금리 데이터 계산 오류")

        # (B) 달러 인덱스 (DXY)
        dxy = current_data.get('dxy')
        # dxy가 NULL(None)이면 조건문 거짓으로 자동 패스
        if dxy and float(dxy) > 105:
            score += 0.02
            details.append(f"킹달러 지속(DXY {dxy}) -> 달러 강세 요인(+2%)")

        # (C) 공포 지수 (VIX)
        vix = current_data.get('vix')
        if vix and float(vix) > 20:
            score += 0.02
            details.append(f"시장 공포(VIX {vix}) -> 안전자산 선호(+2%)")
            
        # 4. 결과 반환
        return {
            "expected_fx_return": round(score, 4),
            "reasoning": details,
            "raw_data_date": str(current_data.get('date', 'Unknown'))
        }

    except Exception as e:
        # 에러 발생 시 안전하게 0.0(중립) 반환
        return {
            "expected_fx_return": 0.0,
            "error": f"Scoring logic failed: {str(e)}"
        }


@tool
def calculate_optimal_hedge_ratio(
    price_json: str,
    risk_aversion: float,
    expected_fx_return: float
) -> dict:
    """
    자산과 환율의 과거 시계열 데이터(JSON)와 투자자 성향, 환율 전망을 입력받아
    최적의 환헤지 비율(Optimal Hedge Ratio)을 계산합니다.
    
    Args:
        price_json: market_data 툴에서 가져온 JSON 문자열 (usdkrw, us_stock 포함)
        risk_aversion: 사용자의 위험 회피 성향 lambda (보통 2~10)
        expected_fx_return: calculate_expected_fx_return 툴의 결과값 (예: 0.03)
    """
    try:
        # 1. JSON 데이터를 DataFrame으로 변환
        df = pd.read_json(price_json)
        
        # 2. 일간 수익률 계산
        returns = df.pct_change().dropna()
        
        # 3. 컬럼 매핑 (DB 컬럼명 기준)
        # market_data.py에서 us_stock, usdkrw로 변환해서 줌
        asset_col = 'us_stock'
        fx_col = 'usdkrw'
        
        # 만약 해당 컬럼이 없으면 유사한 이름 찾기 (방어 코드)
        if asset_col not in returns.columns:
            asset_col = next((c for c in returns.columns if 'stock' in c or 'SPY' in c), None)
        if fx_col not in returns.columns:
            fx_col = next((c for c in returns.columns if 'usdkrw' in c or 'KRW' in c), None)
            
        if not asset_col or not fx_col:
            return {"error": f"Columns not found. Available: {returns.columns}"}

        # 4. 통계량 산출 (연율화 적용, 252일)
        sigma_asset = returns[asset_col].std() * np.sqrt(252)
        sigma_fx = returns[fx_col].std() * np.sqrt(252)
        rho = returns.corr().loc[asset_col, fx_col]
        
        # 5. 최적화 공식 적용
        # (1) Minimum Variance Hedge Term (위험 최소화)
        mv_hedge = 1 + rho * (sigma_asset / sigma_fx)
        
        # (2) Speculative Term (투기적 수요)
        if sigma_fx == 0:
            speculative = 0
        else:
            # risk_aversion이 0이거나 음수면 기본값 1.0 적용
            safe_lambda = risk_aversion if risk_aversion > 0 else 1.0
            speculative = expected_fx_return / (safe_lambda * (sigma_fx ** 2))
        
        # (3) 최종 비율 계산
        w_h = mv_hedge - speculative
        
        # 6. 결과 Clipping (0 ~ 100%)
        final_ratio = max(0.0, min(1.0, w_h))
        
        return {
            "optimal_hedge_ratio": round(final_ratio, 4),
            "details": {
                "sigma_asset": round(sigma_asset, 4),
                "sigma_fx": round(sigma_fx, 4),
                "correlation": round(rho, 4), # type: ignore
                "mv_base_ratio": round(mv_hedge, 4),
                "view_adjustment": round(speculative, 4)
            }
        }
        
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}