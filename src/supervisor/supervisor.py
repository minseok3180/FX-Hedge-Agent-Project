"""Supervisor - 멀티 에이전트 오케스트레이터"""
import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from src.config.settings import settings
from src.agents.web_search_agent import WebSearchAgent
from src.agents.rag_agent import RAGAgent
from src.prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_USER_PROMPT_TEMPLATE

# LangSmith tracing 설정
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"


class Supervisor:
    """멀티 에이전트 시스템의 Supervisor"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        
        # 하위 에이전트 초기화
        self.agents = {
            "web_search_agent": WebSearchAgent(),
            "rag_agent": RAGAgent()
        }
    
    async def route_task(self, user_query: str) -> Dict[str, Any]:
        """
        사용자 질문을 분석하여 적절한 에이전트에게 라우팅
        
        Args:
            user_query: 사용자 질문
            
        Returns:
            에이전트 실행 결과
        """
        try:
            # 1. LLM을 통해 적절한 에이전트 선택
            selected_agent = await self._select_agent(user_query)
            
            # 2. 선택된 에이전트 실행
            agent_name = selected_agent.get("agent")
            if agent_name not in self.agents:
                return {
                    "error": f"알 수 없는 에이전트: {agent_name}",
                    "status": "error"
                }
            
            agent = self.agents[agent_name]
            result = await agent.execute(user_query)
            
            # 3. Supervisor 메타데이터 추가
            result["supervisor_decision"] = selected_agent
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    async def _select_agent(self, user_query: str) -> Dict[str, str]:
        """LLM을 통해 적절한 에이전트 선택"""
        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUPERVISOR_USER_PROMPT_TEMPLATE.format(user_query=user_query)
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    def get_available_agents(self) -> Dict[str, Any]:
        """사용 가능한 에이전트 목록 반환"""
        return {
            name: agent.get_capabilities()
            for name, agent in self.agents.items()
        }

