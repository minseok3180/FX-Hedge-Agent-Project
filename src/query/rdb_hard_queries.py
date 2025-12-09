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
            미국수출금액, 
            미국수입금액, 
            외환보유액, 
            미국외환보유액, 
            한국은행기준금리, 
            정부대출금금리, 
            시장금리, 
            소비자물가지수, 
            수출물가지수, 
            수입물가지수, 
            경제성장률, 
            미국경제성장률, 
            gdp, 
            us_gdp, 
            주가지수, 
            미국주가지수, 
            한국금리, 
            미국금리
        FROM eiExchangeRate
        WHERE date = '{date}'
        ORDER BY date DESC
        LIMIT 1
    """,
    
    "get_by_range": """
        SELECT 
            date, 
            usdkrw, 
            미국수출금액, 
            미국수입금액, 
            외환보유액, 
            미국외환보유액, 
            한국은행기준금리, 
            정부대출금금리, 
            시장금리, 
            소비자물가지수, 
            수출물가지수, 
            수입물가지수, 
            경제성장률, 
            미국경제성장률, 
            gdp, 
            us_gdp, 
            주가지수, 
            미국주가지수, 
            한국금리, 
            미국금리
        FROM eiExchangeRate
        WHERE date BETWEEN %s AND %s
        ORDER BY date DESC
        LIMIT %s
    """,
    
    "get_latest": """
        SELECT 
            date, 
            usdkrw, 
            미국수출금액, 
            미국수입금액, 
            외환보유액, 
            미국외환보유액, 
            한국은행기준금리, 
            정부대출금금리, 
            시장금리, 
            소비자물가지수, 
            수출물가지수, 
            수입물가지수, 
            경제성장률, 
            미국경제성장률, 
            gdp, 
            us_gdp, 
            주가지수, 
            미국주가지수, 
            한국금리, 
            미국금리
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
    
    # User Info 쿼리 (새 사용자 프로필 스키마 기준)
    "get_user_info_by_id": """
        SELECT 
            user_id,
            name,
            age,
            gender,
            total_assets,
            overseas_assets,
            risk_profile
        FROM user_info
        WHERE user_id = '{user_id}'
        LIMIT 1
    """,
    
    "get_all_users": """
        SELECT 
            user_id,
            name,
            age,
            gender,
            total_assets,
            overseas_assets,
            risk_profile
        FROM user_info
    """,
}

