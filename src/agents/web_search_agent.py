"""웹서치 에이전트"""
import json
from typing import Dict, Any, Optional
from src.agents.base_agent import BaseAgent
from src.tools.web_search import WebSearchTool
from src.prompts.web_search_prompt import WEB_SEARCH_SYSTEM_PROMPT, WEB_SEARCH_USER_PROMPT_TEMPLATE


class WebSearchAgent(BaseAgent):
    """환율 정보를 웹에서 검색하는 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="web_search_agent",
            description="환율 정보를 웹에서 검색하고 분석하는 에이전트"
        )
        self.web_search_tool = WebSearchTool()
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        웹 검색을 수행하고 결과를 분석
        
        Args:
            task: 검색할 질문
            context: 추가 컨텍스트
            
        Returns:
            검색 및 분석 결과
        """
        try:
            # 1. 검색 쿼리 생성 (간단하게 task를 그대로 사용)
            search_query = task
            
            # 2. 웹 검색 수행
            search_results = await self.web_search_tool.search(search_query, num_results=5)
            
            # 3. 검색 결과를 텍스트로 변환
            results_text = "\n\n".join([
                f"제목: {r['title']}\n요약: {r['snippet']}\n링크: {r['link']}"
                for r in search_results
            ])
            
            # 4. LLM을 통해 검색 결과 분석 및 답변 생성
            messages = [
                {"role": "system", "content": WEB_SEARCH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": WEB_SEARCH_USER_PROMPT_TEMPLATE.format(
                        user_query=task,
                        search_results=results_text
                    )
                }
            ]
            
            answer = await self._call_llm(messages, temperature=0.7)
            
            return {
                "agent": self.name,
                "task": task,
                "search_query": search_query,
                "search_results": search_results,
                "answer": answer,
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name,
                "task": task,
                "error": str(e),
                "status": "error"
            }

