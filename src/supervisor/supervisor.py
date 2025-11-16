"""Supervisor - 멀티 에이전트 오케스트레이터"""
import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from src.config.settings import settings
from src.agents.web_search_agent import WebSearchAgent
from src.agents.rag_agent import RAGAgent
from src.prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_USER_PROMPT_TEMPLATE
from src.utils.logger import get_logger
from src.utils.openai_tracer import TracedOpenAIClient

# LangSmith tracing 설정
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"


class Supervisor:
    """멀티 에이전트 시스템의 Supervisor"""
    
    def __init__(self):
        self.logger = get_logger("supervisor")
        # LangSmith 추적이 포함된 OpenAI 클라이언트 사용
        self.client = TracedOpenAIClient(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        
        self.logger.info("🔧 Supervisor 초기화 시작")
        
        # 하위 에이전트 초기화
        self.agents = {
            "web_search_agent": WebSearchAgent(),
            "rag_agent": RAGAgent()
        }
        
        self.logger.info(
            f"✅ Supervisor 초기화 완료",
            {"available_agents": list(self.agents.keys())}
        )
    
    async def route_task(self, user_query: str) -> Dict[str, Any]:
        """
        사용자 질문을 분석하여 적절한 에이전트에게 라우팅
        
        Args:
            user_query: 사용자 질문
            
        Returns:
            에이전트 실행 결과
        """
        self.logger.info(
            f"🔄 작업 라우팅 시작",
            {"query_length": len(user_query), "query_preview": user_query[:100]}
        )
        
        try:
            # 1. LLM을 통해 적절한 에이전트 선택
            self.logger.debug("🤔 에이전트 선택 중...")
            selected_agent = await self._select_agent(user_query)
            agent_name = selected_agent.get("agent")
            
            self.logger.info(
                f"✅ 에이전트 선택 완료: {agent_name}",
                {
                    "selected_agent": agent_name,
                    "reasoning": selected_agent.get("reasoning", ""),
                    "task": selected_agent.get("task", "")
                }
            )
            
            # 2. 선택된 에이전트 실행
            if agent_name not in self.agents:
                error_msg = f"알 수 없는 에이전트: {agent_name}"
                self.logger.error(f"❌ {error_msg}", {"selected_agent": agent_name})
                return {
                    "error": error_msg,
                    "status": "error"
                }
            
            self.logger.info(f"🚀 에이전트 실행 시작: {agent_name}")
            agent = self.agents[agent_name]
            result = await agent.execute(user_query)
            
            # 3. Supervisor 메타데이터 추가
            result["supervisor_decision"] = selected_agent
            
            self.logger.info(
                f"✅ 작업 라우팅 완료",
                {
                    "agent": agent_name,
                    "status": result.get("status"),
                    "answer_length": len(result.get("answer", ""))
                }
            )
            
            return result
        except Exception as e:
            self.logger.error(
                f"❌ 작업 라우팅 실패",
                {"error": str(e)},
                exc_info=True
            )
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
        
        self.logger.debug(
            "📤 Supervisor LLM 호출",
            {"model": self.model, "query_preview": user_query[:50]}
        )
        
        response = self.client.chat_completions_create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        self.logger.debug(f"📥 Supervisor LLM 응답 수신", {"result": result})
        return result
    
    def get_available_agents(self) -> Dict[str, Any]:
        """사용 가능한 에이전트 목록 반환"""
        return {
            name: agent.get_capabilities()
            for name, agent in self.agents.items()
        }

