from openai import OpenAI
import ast
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

def extract_keywords(question):
    prompt = f"""
    다음 문장에서 핵심 키워드를 **영어로만** 뽑아주세요.
    문장: "{question}"
    출력 형식: ["키워드1", "키워드2", ...]
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "당신은 키워드 추출 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    keywords_str = response.choices[0].message.content.strip() # type: ignore
    try:
        keywords = ast.literal_eval(keywords_str)
    except:
        keywords = [keywords_str]
    return keywords


# # 테스트용 
# if __name__ == "__main__":
#     question = "오늘 USD/KRW 환율 전망 알려줘"
#     print("추출된 키워드:", extract_keywords(question))
