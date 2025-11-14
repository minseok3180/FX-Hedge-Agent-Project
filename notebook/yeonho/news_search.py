from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("FINLIGHT_API_KEY")


def news_search(keywords):

    if not keywords:
        print("🚨 키워드가 없습니다.")
        return []
    
    QUERY = f"({' OR '.join(keywords)})" # (forex OR \"foreign exchange\" OR currency) AND 

    client = FinlightApi(
        config=ApiConfig(
            api_key=API_KEY, # type: ignore
            base_url="https://api.finlight.me" # type: ignore
        )
    )

    params = GetArticlesParams(
        query=QUERY,
        language="en",
        from_="2025-01-01", # type: ignore
        to="2025-11-13",
    ) # type: ignore

    try:
        response = client.articles.fetch_articles(params=params)
        articles = response.articles
        
        articles_list = []
        for article in articles[:30]:
            articles_list.append({
                "title": article.title,
                "source": getattr(article.source, 'name', str(article.source)),
                "url": getattr(article, 'url', getattr(article, 'link', 'URL Not Found')), 
                "date": article.publishDate.strftime("%Y-%m-%d") if article.publishDate else "Unknown"
            })
        return articles_list

    except Exception as e:
        print(f"🚨 API 요청 중 오류 발생: {e}")
        return []

# 테스트용
if __name__ == "__main__":
    test_keywords = ["USD/KRW", "currency rate", "forecast"]
    results = news_search(test_keywords)
    for i, article in enumerate(results):
        print(f"\n--- {i+1}. {article['title']} ---")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}")
        print(f"PublishDate : {article['date']}")