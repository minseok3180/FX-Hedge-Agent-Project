"""RDB 하드 쿼리 - 조회용 SQL 쿼리

Placeholder 지원:
- {date}: state에서 가져온 날짜 (YYYY-MM-DD)
- {user_id}: state에서 가져온 사용자 ID

사용 예시:
- Placeholder 사용: WHERE date = '{date}' → state에서 date 값으로 자동 치환
- 파라미터 바인딩: WHERE date = %s → execute() 호출 시 params로 전달
"""

rdb_hard_queries = {
    "get_by_date": """
        SELECT 
            date, 
            usdkrw, 
            us_ex, 
            us_im, 
            reserve, 
            us_reserve, 
            us_export, 
            us_import,
            base, 
            market, 
            consumer, 
            exp_rate, 
            im_rate, 
            us_current, 
            us_growth, 
            us_gdp, 
            us_stock, 
            us_interest
        FROM eiExchangeRate
        WHERE date = '{date}'
        ORDER BY date DESC
        LIMIT 1
    """,
    
    "get_by_range": """
        SELECT 
            date, 
            usdkrw, 
            us_ex, 
            us_im, 
            reserve, 
            us_reserve, 
            us_export, 
            us_import,
            base, 
            market, 
            consumer, 
            exp_rate, 
            im_rate, 
            us_current, 
            us_growth, 
            us_gdp, 
            us_stock, 
            us_interest
        FROM eiExchangeRate
        WHERE date BETWEEN %s AND %s
        ORDER BY date DESC
        LIMIT %s
    """,
    
    "get_latest": """
        SELECT 
            date, 
            usdkrw, 
            us_ex, 
            us_im, 
            reserve, 
            us_reserve, 
            us_export, 
            us_import,
            base, 
            market, 
            consumer, 
            exp_rate, 
            im_rate, 
            us_current, 
            us_growth, 
            us_gdp, 
            us_stock, 
            us_interest
        FROM eiExchangeRate
        ORDER BY date DESC
        LIMIT %s
    """,
    
    "get_exchange_rate_only": """
        SELECT date, usdkrw
        FROM eiExchangeRate
        WHERE date = %s
        LIMIT 1
    """,
    
    # User Info 쿼리
    "get_user_info_by_id": """
        SELECT 
            user_id,
            user_name,
            user_krw,
            user_usd
        FROM user_info
        WHERE user_id = '{user_id}'
        LIMIT 1
    """,
    
    "get_all_users": """
        SELECT 
            user_id,
            user_name,
            user_krw,
            user_usd
        FROM user_info
    """,
}

