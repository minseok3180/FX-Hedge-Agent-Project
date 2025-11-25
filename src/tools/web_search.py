"""웹 검색 도구"""
import requests
from typing import List, Dict, Any
from src.utils.settings import settings


class WebSearchTool:
    """웹 검색을 수행하는 도구"""
    
    def __init__(self):
        self.api_key = settings.web_search_api_key
        self.engine_id = settings.web_search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        웹 검색 수행
        
        Args:
            query: 검색 쿼리
            num_results: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self.api_key or not self.engine_id:
            # API 키가 없는 경우 더미 데이터 반환 (개발용)
            return [
                {
                    "title": f"환율 정보: {query}",
                    "snippet": f"{query}에 대한 환율 정보입니다.",
                    "link": "https://example.com"
                }
            ]
        
        try:
            params = {
                "key": self.api_key,
                "cx": self.engine_id,
                "q": query,
                "num": num_results
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
            
            return results
        except Exception as e:
            # 에러 발생 시 빈 리스트 반환
            print(f"웹 검색 에러: {e}")
            return []

