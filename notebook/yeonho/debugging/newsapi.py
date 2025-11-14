import requests
import json
from datetime import datetime, timedelta

# --- 1. 변수 설정: 개인 정보 및 검색 조건 정의 ---

# 🚨 여기에 발급받은 API 키를 넣으세요!
API_KEY = "47dfb621bef246c99feba959caa2d3ba"

# 검색 엔드포인트
ENDPOINT = "https://newsapi.org/v2/everything"

# 검색어 (외환 관련 기사를 찾기 위해 'forex' 사용)
QUERY = "forex OR foreign exchange"

# 검색 기간 설정 (예시: 오늘로부터 지난 7일)
# News API는 Free Plan의 경우 30일 이내의 기사만 제공합니다.
today = datetime.now()
seven_days_ago = today - timedelta(days=7)
FROM_DATE = seven_days_ago.strftime('%Y-%m-%d')
TO_DATE = today.strftime('%Y-%m-%d')


# --- 2. 요청 파라미터 구성 ---

# 여러 주요 금융 도메인들을 쉼표로 연결
FINANCE_DOMAINS = (
    "dailyforex.com," +
    "wsj.com," +
    "ft.com," +
    "bloomberg.com," +
    "reuters.com," +
    "cnbc.com," +
    "investing.com," +
    "fxstreet.com"
)


# API 요청에 필요한 모든 조건을 딕셔너리로 만듭니다.
parameters = {
    'q': QUERY,                 # 검색 키워드
    'apiKey': API_KEY,          # 인증 키
    'language': 'en',           # 기사 언어 (예: 영어)
    'sortBy': 'relevancy',      # 정렬 기준 (관련성 순)
    'from': FROM_DATE,          # 검색 시작 날짜
    'to': TO_DATE,              # 검색 종료 날짜
    'domains': FINANCE_DOMAINS,
    'pageSize': 10,             # 가져올 기사 수 (최대 100)
    'page': 1                   # 페이지 번호
}


# --- 3. API 호출 및 응답 처리 ---

try:
    # GET 요청 보내기
    response = requests.get(ENDPOINT, params=parameters)
    
    # HTTP 상태 코드 확인 (200은 성공)
    if response.status_code == 200:
        data = response.json()
        
        # 전체 결과 수 확인
        total_results = data.get('totalResults', 0)
        print(f"✅ API 요청 성공! 총 {total_results}개의 기사가 검색되었습니다.")
        
        # 기사 목록 가져오기
        articles = data.get('articles', [])
        
        # 상위 5개 기사 정보 출력 예시
        print("\n--- 상위 10개 Forex 기사 제목 ---")
        for i, article in enumerate(articles[:10]):
            print(f"{i+1}. {article.get('title')}")
            print(f"   출처: {article.get('source', {}).get('name')}")
            print(f"   URL: {article.get('url')}\n")
            
    else:
        # 요청 실패 시 오류 메시지 출력
        print(f"❌ API 요청 실패. HTTP Status Code: {response.status_code}")
        print("응답 메시지:", response.json().get('message', 'N/A'))

except requests.exceptions.RequestException as e:
    # 네트워크 연결 등 요청 자체의 오류 처리
    print(f"🚨 요청 중 오류 발생: {e}")

