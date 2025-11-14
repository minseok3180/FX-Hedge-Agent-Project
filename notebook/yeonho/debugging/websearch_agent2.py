from keyword_extract import extract_keywords
from news_search import news_search
from notebook.yeonho.debugging.open_url import get_article_text

def web_search_agent(question):
    
    # 1. 키워드 추출
    keywords = extract_keywords(question)
    if not keywords:
        print("🚨 키워드가 추출되지 않았습니다.")
        return []

    print(f"🔑 추출된 키워드: {keywords}")

    # 2. 뉴스 검색
    articles = news_search(keywords)
    if not articles:
        print("🚨 뉴스 검색 결과가 없습니다.")
        return []

    print(f"📰 {len(articles)}개의 기사 검색됨")

    return results

    # 3. 뉴스 원문 열람 + 요약
    # results = []
    # for article in articles:
    #     text = get_article_text(article['url'])
    #     if not text:
    #         continue
    #     summary = summarize_text(text)
    #     results.append({
    #         "title": article['title'],
    #         "source": article['source'],
    #         "url": article['url'],
    #         "summary": summary
    #     })

    # return results

# 테스트용
if __name__ == "__main__":
    question = "오늘 USD/KRW 환율 전망 알려줘"
    results = web_search_agent(question)
    for i, article in enumerate(results):
        print(f"\n{i+1}. {article['title']} ({article['source']})")
        print(f"URL: {article['url']}")
        print(f"Summary: {article['summary']}")
