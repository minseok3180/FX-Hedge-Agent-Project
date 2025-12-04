from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import importlib.util
from pathlib import Path

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# news_search를 직접 로드 (__init__.py를 거치지 않음)
news_search_path = Path(__file__).parent / "news_search.py"
spec = importlib.util.spec_from_file_location("news_search", news_search_path)
news_search_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(news_search_module)
fetch_news = news_search_module.fetch_news

def analyze_news(news_list):
    news_list_str = "\n".join(news_list)

    prompt = f"""
    You are an expert FX analyst. 

    Analyze the following news articles and provide the FX impact in list of dictionaries format.

    Requirements:
    1. Only include currencies mentioned or affected in the news (USD, EUR, JPY, etc.).
    2. For each currency, provide:
    - "currency": currency code as string
    - "impact": a number between 0 and 1 representing the strength of the news' effect. If unknown, use "unknown".
    - "direction": "up", "down", or "neutral". If unknown, use "unknown".
    - "risk": "low", "medium", or "high". If unknown, use "unknown".
    3. Output ONLY python list of dictionaries, no extra text.

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

    return response.choices[0].message.content


def run_sentiment_analysis():
    news_list = fetch_news()
    result = analyze_news(news_list)
    return result


if __name__ == "__main__":
    print(run_sentiment_analysis())
