from openai import OpenAI
import os
from dotenv import load_dotenv
from keyword_extract import extract_keywords
from news_search import news_search

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

def web_search_agent(question):
    # 1. 키워드 추출
    keywords = extract_keywords(question)
    if not keywords:
        print("🚨 키워드가 추출되지 않았습니다.")
        return None

    print(f"🔑 추출된 키워드: {keywords}")

    # 2. 뉴스 검색 (제목만 사용)
    articles = news_search(keywords)
    if not articles:
        print("🚨 뉴스 검색 결과가 없습니다.")
        return None

    print(f"📰 {len(articles)}개의 기사 검색됨")

    # 3. 기사 제목 리스트 만들기 
    for i, article in enumerate(articles):
        print(f"{i+1}. {article['title']}")

    titles = [article['title'] for article in articles]

    # 4. Prompt 구성
    titles_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
    prompt = f"""
    당신은 외환 전문가이자 분석가입니다.

    사용자가 질문을 했습니다: "{question}"

    아래는 사용자 질문 관련 최근 외환 뉴스 기사 제목입니다.
    기사 제목은 참고용으로 활용하세요.

    기사 제목 :
    {titles_text}

    요청:
    1) 질문에 대한 간단하고 명확한 답변을 **한국어로** 작성하세요.
    2) 기사 제목과 관련이 있으면 참고하여 답변에 활용할 수 있지만, 불필요하거나 부정확한 내용은 생략하세요.
    3) 결론과 현재 시장 추세를 포함하세요.
    """
    
    # 5. LLM 호출
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "당신은 금융 뉴스 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    answer = response.choices[0].message.content.strip()  # type: ignore
    return answer

# --- 테스트 ---
if __name__ == "__main__":
    question = "오늘 USD/KRW 환율 전망 알려줘"
    answer = web_search_agent(question)
    print("\n💡 에이전트 답변:\n", answer)
