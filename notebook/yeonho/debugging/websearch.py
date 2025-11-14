from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams
from datetime import datetime, timedelta
import re
from openai import OpenAI

API_KEY = "sk_f5ca63f305d5e0c274a955f19f65785ce9e9ceddcb4ddcacf124f80ef8ffa9b2"

BASE_QUERY = "(forex OR \"foreign exchange\" OR currency)"

# --- 1. 사용자 질문 분석 함수 ---
def analyze_question(question: str):
    """
    사용자 질문에서 핵심 통화, 검색어, 기간 정보를 추출합니다.
    """
    # 1. 핵심 통화 및 키워드 추출
    currency_map = {
        "달러": ["USD", "DOLLAR"],
        "엔화": ["JPY", "YEN"],
        "유로": ["EUR", "EURO"],
        "파운드": ["GBP", "POUND"]
    }
    
    # 질문에 포함된 모든 통화 관련 키워드를 찾습니다.
    keywords = []
    for kor, eng_list in currency_map.items():
        if kor in question or any(eng in question.upper() for eng in eng_list):
            keywords.extend(eng_list)
            
    # 핵심 검색어(finlight query) 생성
    if not keywords:
        # 특정 통화가 없으면 BASE_QUERY에 질문의 키워드를 추가합니다.
        final_query = f"{BASE_QUERY} AND ({question})" 
    else:
        # 특정 통화가 있으면 해당 통화 키워드를 AND로 묶습니다.
        keyword_part = " OR ".join(keywords)
        final_query = f"{BASE_QUERY} AND ({keyword_part})"

    # 2. 기간 설정 (간단하게 '오늘'만 처리)
    # 실제 에이전트는 더 복잡한 날짜 처리가 필요합니다.
    if "오늘" in question or "최근" in question:
        from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
    else:
        # 기간 명시가 없으면 최근 3일 치를 기본값으로 설정
        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
    return final_query, from_date, to_date

# --- 2. API 호출 함수 (확장) ---
def fetch_forex_news(query: str, from_date: str, to_date: str):
    client = FinlightApi(
        config=ApiConfig(
            api_key=API_KEY,
            base_url="https://api.finlight.me" # type: ignore
        )
    )
    
    params = GetArticlesParams(
        query=query,
        language="en",
    ) # type: ignore

    try:
        response = client.articles.fetch_articles(params=params)
        return response.articles
    except Exception as e:
        print(f"🚨 API 요청 중 오류 발생: {e}")
        return []

# --- 3. 메인 챗봇 로직 ---
def run_forex_agent(user_question: str):
    """
    사용자의 질문을 받고 finlight API를 호출하여 응답을 생성합니다.
    """
    print(f"**받은 질문:** {user_question}")
    
    # 1. 질문 분석 및 검색어 생성
    final_query, from_date, to_date = analyze_question(user_question)
    print(f"**생성된 검색 쿼리:** {final_query}")

    # 2. API 호출
    articles = fetch_forex_news(final_query, from_date, to_date)

    if not articles:
        return "죄송합니다. 해당 주제에 대한 최신 금융 뉴스를 찾을 수 없습니다."

    # 3. 결과 요약 및 응답 생성
    response_lines = [f"✅ **'{user_question}'**에 대한 최근 금융 뉴스 요약입니다:"]
    
    # 상위 3~5개 기사 제목과 요약을 기반으로 응답 생성
    for i, article in enumerate(articles[:5]):
        summary_text = getattr(article, 'summary', '상세 요약 없음') 
        
        response_lines.append(f"\n--- {i+1}. {article.title} ---")
        response_lines.append(f"   출처: {getattr(article.source, 'name', str(article.source))}")
        response_lines.append(f"   주요 내용: {summary_text}.")

    return "\n".join(response_lines)

# --- 실행 예시 ---
if __name__ == "__main__":
    
    # 1. 달러 뉴스 요청
    question_dollar = "오늘 달러 뉴스 요약해줘!"
    response_dollar = run_forex_agent(question_dollar)
    print("\n" + "="*50)
    print(f"**[에이전트 응답]**\n{response_dollar}")
    print("="*50 + "\n")

    # 2. 엔화 급등 이유 요청
    question_yen = "최근 엔화가 급등한 이유가 뭐야?"
    response_yen = run_forex_agent(question_yen)
    print("\n" + "="*50)
    print(f"**[에이전트 응답]**\n{response_yen}")
    print("="*50)