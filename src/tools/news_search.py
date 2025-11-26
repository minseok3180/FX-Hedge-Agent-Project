import requests

def fetch_news():
    url = "https://forexnewsapi.com/api/v1/category"
    params = {
        "section": "general",
        "items": 50,
        "page": 5,
        "token": "yyngfmqhyup8kqfze4rqg5sbcuiedbfzvsjwskpw"
    }

    response = requests.get(url, params=params)
    articles_list = []

    if response.status_code == 200:
        data = response.json()
        articles = data.get("data", []) or data.get("items", [])

        for article in articles:
            combined = (
                f"Title: {article.get('title')}\n"
                f"Date: {article.get('date')}\n"
                f"Topics: {', '.join(article.get('topics', []))}\n"
                f"Sentiment: {article.get('sentiment')}\n"
                f"Text: {article.get('text')}"
            )
            articles_list.append(combined)

    else:
        print(f"API Error: {response.status_code}")

    return articles_list

import json

# 뉴스 저장
news_list = fetch_news()
with open("news_cache.json", "w", encoding="utf-8") as f:
    json.dump(news_list, f, ensure_ascii=False, indent=2)

# 뉴스 불러오기
with open("news_cache.json", "r", encoding="utf-8") as f:
    news_list = json.load(f)