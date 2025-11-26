"""웹 검색 도구"""
import requests
from typing import List, Dict, Any
from src.utils.settings import settings
from src.utils.logger import get_logger
from src.utils.tools import (
    tool,
    handle_tool_error,
    WebSearchInput
)

logger = get_logger("web-search-tool")

_base_url = "https://www.googleapis.com/customsearch/v1"


@tool(args_schema=WebSearchInput)
@handle_tool_error("web_search")
async def web_search(
    query: str,
    num_results: int = 5
) -> List[Dict[str, Any]]:
    """
    웹 검색 수행 (Google Custom Search API 사용)
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수 (최대 10, 기본값: 5)
        
    Returns:
        검색 결과 리스트 (title, snippet, link 포함)
    """
    # num_results 제한 (Google API는 최대 10개)
    num_results = min(num_results, 10)
    
    api_key = settings.web_search_api_key
    engine_id = settings.web_search_engine_id
    
    if not api_key or not engine_id:
        logger.warning(
            "⚠️ 웹 검색 API 키 또는 Engine ID가 설정되지 않음",
            {"has_api_key": bool(api_key), "has_engine_id": bool(engine_id)}
        )
        return []
    
    # Tool 호출 및 검색 쿼리 로깅
    logger.info(
        f"🔧 [TOOL CALL] web_search 실행",
        {
            "tool_name": "web_search",
            "query": query,
            "num_results": num_results
        }
    )
    
    logger.info(
        f"📝 [QUERY] 웹 검색 실행",
        {
            "search_query": query,
            "num_results": num_results
        }
    )
    
    try:
        params = {
            "key": api_key,
            "cx": engine_id,
            "q": query,
            "num": num_results
        }
        
        response = requests.get(_base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        
        logger.info(
            f"✅ [TOOL RESULT] web_search 완료",
            {
                "tool_name": "web_search",
                "query": query,
                "results_count": len(results),
                "result_preview": [{"title": r["title"][:50]} for r in results[:3]] if results else []
            }
        )
        
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(
            f"❌ 웹 검색 API 요청 실패",
            {"query": query, "error": str(e)},
            exc_info=True
        )
        return []
    except Exception as e:
        logger.error(
            f"❌ 웹 검색 처리 중 오류 발생",
            {"query": query, "error": str(e)},
            exc_info=True
        )
        return []

