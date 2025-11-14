from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams
# import json (json은 여기서 불필요하므로 삭제하거나 주석 처리)

# 🚨 여기에 발급받은 당신의 API 키를 넣어주세요!
API_KEY = "sk_f5ca63f305d5e0c274a955f19f65785ce9e9ceddcb4ddcacf124f80ef8ffa9b2"

# 외환 검색의 정확도를 높이기 위한 강력한 쿼리
# 'forex'나 'foreign exchange'를 포함하는 기사 중, 반드시 주요 통화 키워드 중 하나 이상이 포함된 기사만 검색
QUERY = "(forex OR \"foreign exchange\" OR currency) AND (USD OR EUR OR JPY OR GBP OR CAD OR AUD OR YEN OR DOLLAR)"


def fetch_forex_news():
    client = FinlightApi(
        config=ApiConfig(
            api_key=API_KEY,
            base_url="https://api.finlight.me"  # type: ignore
        )
    )

    # Forex 검색에 최적화된 파라미터 설정
    params = GetArticlesParams(
        query=QUERY,
        language="en",
    ) # type: ignore

    try:
        # 1. API를 호출하여 응답 객체(ArticleResponse)를 받습니다.
        response = client.articles.fetch_articles(params=params)
        
        # 2. 응답 객체에서 실제 기사 목록을 추출합니다. (변수 이름을 articles로 재정의)
        articles = response.articles 
        
        # 3. 이제 list 형태인 articles에 len()을 호출합니다.
        print(f"✅ finlight 요청 성공! 총 {len(articles)}개의 기사를 가져왔습니다.")
        print("\n--- 상위 Forex 기사 및 센티먼트 ---")
        
        for i, article in enumerate(articles):
            
            print(f"{i+1}. {article.title}")
            
            # 💡 수정된 부분: article.source가 문자열일 경우를 대비하여 .name을 제거합니다.
            source_name = getattr(article.source, 'name', str(article.source))
            print(f"   출처: {source_name}")
            
            # link 또는 url 필드를 사용하여 URL 출력
            article_url = getattr(article, 'url', getattr(article, 'link', 'URL Not Found'))
            print(f"   URL: {article_url}\n") 
            
    except Exception as e:
        # 오류 메시지를 좀 더 명확하게 출력합니다.
        print(f"🚨 API 요청 중 오류 발생: {e}")

if __name__ == "__main__":
    fetch_forex_news()