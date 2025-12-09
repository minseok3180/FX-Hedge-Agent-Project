import pandas as pd
import json
from langchain_core.tools import tool
from tools.rdb import rdb_query_hard 

@tool
async def fetch_historical_data_for_quant() -> str:
    """
    RDB에서 최근 1년치 환율(usdkrw) 및 미국 주식(us_stock) 데이터를 조회하여,
    Calculator가 변동성과 상관계수 계산에 사용할 수 있는 JSON 문자열로 변환합니다.

    Returns:
        str: 날짜(date)를 인덱스로 하고 'usdkrw', 'us_stock' 컬럼을 가진 JSON 문자열
    """
    try:
        # 1. DB에서 1년치 데이터 조회 (비동기 호출)
        rows = await rdb_query_hard.ainvoke({
            "query_key": "get_1y_history",
            "params": None
        })
        
        if not rows:
            return json.dumps({"error": "No historical data found in DB"})
            
        # 2. DataFrame 변환
        df = pd.DataFrame(rows)
        
        # 3. 데이터 전처리
        # (A) 날짜 컬럼 인덱스 설정
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        # (B) 숫자형 변환 (Decimal -> float)
        # MariaDB에서 넘어온 데이터는 Decimal 타입일 수 있어 JSON 변환 시 에러가 날 수 있음
        numeric_cols = ['usdkrw', 'us_stock']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # 4. JSON 변환 (Quant Engine 입력용)
        # date_format='iso'를 써야 날짜가 "YYYY-MM-DD" 문자열로 나옴
        return df.to_json(date_format='iso')

    except Exception as e:
        return json.dumps({"error": f"Data fetch failed: {str(e)}"})