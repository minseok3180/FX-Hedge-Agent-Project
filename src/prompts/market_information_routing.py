"""Market Information 에이전트 라우팅 설명 (Supervisor 라우팅용)"""

MARKET_INFORMATION_ROUTING = """## 에이전트: market_information

**역할**: 시장 정보를 수집하고 분석하여 헤지 전략 수립에 필요한 인사이트를 제공

**보유 툴**:
- rdb_query_hard: Query 폴더에 정의된 쿼리를 사용하여 RDB에서 데이터 조회
  - get_by_date: 특정 날짜의 환율 및 경제 지표 조회
  - get_by_range: 날짜 범위의 환율 및 경제 지표 조회
  - get_latest: 최신 환율 및 경제 지표 조회
- rdb_query_llm: LLM이 쿼리문을 직접 작성하여 RDB에서 데이터 조회 (복잡한 쿼리)
- web_search: 웹에서 최신 정보 검색 (Google Custom Search API, 최신 뉴스 및 실시간 정보)

**사용 시기**: 
- 시장 정보 수집 및 분석이 필요할 때
- 헤지 전략 수립을 위한 데이터가 필요할 때
- 환율 예측 및 리스크 분석이 필요할 때
- 최신 뉴스나 실시간 시장 동향 정보가 필요할 때
- RDB에 없는 최근 시장 정보나 뉴스가 필요할 때
"""

