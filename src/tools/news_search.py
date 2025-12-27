import requests

def fetch_news():
    url = "https://forexnewsapi.com/api/v1/category"
    params = {
        "section": "general",
        "items": 3,
        "page": 1,
        "token": "5bzy7pu46uy9pascstk9sdbgkohmfzpxxwy5n1ia"
    }

    response = requests.get(url, params=params)
    articles_list = []   # 문자열 리스트

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


