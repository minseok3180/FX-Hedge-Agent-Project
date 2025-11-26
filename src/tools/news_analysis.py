from openai import OpenAI
from news_search import fetch_news
from dotenv import load_dotenv
import os
import json
import time
import ast

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

CACHE_FILE = "news_cache.json"
CACHE_EXPIRY = 24 * 60 * 60  # 하루마다 갱신

def get_news():
    # 캐시 존재 여부 확인
    if os.path.exists(CACHE_FILE):
        last_mod = os.path.getmtime(CACHE_FILE)
        if time.time() - last_mod < CACHE_EXPIRY:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                news_list = json.load(f)
                return news_list

    # 캐시 없거나 만료 → API 호출
    news_list = fetch_news()
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(news_list, f, ensure_ascii=False, indent=2)
    return news_list

def analyze_news(news_list):
    news_list_str = "\n".join(news_list)

    prompt = f"""
    You are an expert FX analyst focusing ONLY on USD.

    Analyze the following news articles and provide the USD impact in Python dictionary format.

    Requirements:
    1. Only analyze USD, ignore other currencies.
    2. For USD, provide:
    - "currency": "USD"
    - "impact": a number between 0 and 1 representing strength of the news' effect.
    - "direction": "up", "down", or "neutral".
    - "risk": "low", "medium", or "high".
    - "confidence": a number between 0 and 1 representing how confident the model is.
    - "volatility_impact": a number between 0 and 1 representing expected short-term FX volatility effect.
    3. Output ONLY a single Python dictionary, no extra text.
    -  Do NOT include any Markdown code blocks like ```python or ```; just the raw dictionary.

    News Articles:
    {news_list_str}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a forex market analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    # 문자열을 Python dict로 변환
    return ast.literal_eval(response.choices[0].message.content) # type: ignore

def run_sentiment_analysis():
    news_list = get_news()
    result = analyze_news(news_list)
    return result

if __name__ == "__main__":
    result = run_sentiment_analysis()
    print(result)

    # JSON 파일로 저장
    with open("usd_news.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)