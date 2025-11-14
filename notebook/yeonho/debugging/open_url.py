import requests
from bs4 import BeautifulSoup

def get_article_text(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 처리

        soup = BeautifulSoup(response.text, "html.parser")

        # 일반적인 기사 본문 태그
        # <p> 태그 안 텍스트를 합쳐서 반환
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs])

        return article_text.strip()

    except Exception as e:
        print(f"🚨 URL 열람 오류 ({url}): {e}")
        return ""
    

# 테스트용
if __name__ == "__main__":
    test_url = "https://www.investing.com/news/forex-news/asia-fx-muted-yuan-dips-on-dismal-trade-dollar-weakens-on-soft-jobs-data-4341072"
    text = get_article_text(test_url)
    print(text[:500])  # 앞 500자만 확인
