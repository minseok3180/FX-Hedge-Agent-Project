# src/agents/expert_information.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger
from src.utils.settings import settings

from src.prompts.docs_prompt import (
    DOCS_SYSTEM_PROMPT,
    DOCS_USER_PROMPT_TEMPLATE,
)
from src.tools.expert_information_tools import (
    expert_search,
    build_expert_context_from_hits,
)


logger = get_logger(__name__)


class ExpertInformationAgent(BaseAgent):
    """
    전문가 문서(hedge fund docs)에서 정보를 검색해 요약해주는 RAG 에이전트.

    구조:
      1) 사용자의 질문을 OpenAI 임베딩으로 변환
      2) expert_search 툴로 Qdrant에서 관련 청크 검색
      3) 검색된 컨텍스트를 LLM 프롬프트에 넣어 요약/분석
      4) 최종 답변과 hits를 함께 반환
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        collection_name: Optional[str] = None,
    ):
        super().__init__(
            name="expert_information",
            description="전문가 리포트/문서(hedge docs)를 기반으로 환헤지 관련 정보를 제공하는 에이전트",
        )
        self.embedding_model = embedding_model
        self.collection_name = collection_name or getattr(
            settings,
            "hedge_docs_collection",
            "hedge_fund_docs",
        )
        self._embed_client = OpenAI(api_key=settings.openai_api_key)

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            task: 사용자의 자연어 질문
            context:
              - top_k (int, optional): 검색할 문서 청크 개수 (기본 5)

        Returns:
            {
              "agent": "expert_information",
              "task": <원본 task>,
              "status": "success" | "error",
              "answer": <LLM이 정리한 최종 답변>,
              "hits": [...],  # Qdrant 검색 결과
            }
        """
        context = context or {}
        top_k = int(context.get("top_k", 5))

        try:
            # 1) 쿼리 임베딩 생성
            query_embedding = await self._embed_text(task)

            # 2) 전문가 문서 검색 (툴 사용)
            hits = await expert_search.ainvoke(
                {
                    "query_vector": query_embedding,
                    "top_k": top_k,
                    "collection_name": self.collection_name,
                }
            )

            # hits가 비었으면 그대로 안내 메시지 반환
            if not hits:
                msg = (
                    "전문가 문서 벡터 DB에서 관련된 내용을 찾지 못했습니다. "
                    "지금은 일반적인 금융 지식 수준에서만 답변이 가능할 것 같습니다."
                )
                return {
                    "agent": self.name,
                    "task": task,
                    "status": "success",
                    "answer": msg,
                    "hits": [],
                }

            # 3) LLM 컨텍스트 문자열 생성
            context_text = build_expert_context_from_hits(hits)

            # 4) LLM 프롬프트 구성
            system_prompt = DOCS_SYSTEM_PROMPT

            user_prompt = DOCS_USER_PROMPT_TEMPLATE.format(
                user_question=task,
                retrieved_context=context_text,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # 5) LLM 호출
            answer = await self._call_llm(messages, temperature=0.2)

            return {
                "agent": self.name,
                "task": task,
                "status": "success",
                "answer": answer,
                "hits": hits,
            }

        except Exception as e:
            logger.error(
                "ExpertInformationAgent.execute failed",
                {"error": str(e)},
                exc_info=True,
            )
            return {
                "agent": self.name,
                "task": task,
                "status": "error",
                "answer": f"전문가 정보 검색 중 오류가 발생했습니다: {e}",
                "hits": [],
            }

    async def _embed_text(self, text: str) -> List[float]:
        """
        OpenAI 임베딩 API를 사용해 쿼리 텍스트를 벡터로 변환.
        (툴이 아니라 에이전트 내부에서만 사용)
        """
        logger.info(
            "Creating embedding for expert query",
            {"model": self.embedding_model},
        )
        res = await self._embed_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return res.data[0].embedding
