# src/agents/expert_information_agent.py
"""
전문가 문서(hedge fund docs)에서 정보를 검색해 요약해주는 RAG 에이전트.

vdb.py의 vdb_search를 사용하여 Qdrant에서 관련 청크를 검색한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

from openai import OpenAI

from src.utils.agents import BaseAgent
from src.utils.logger import get_logger
from src.utils.settings import settings
from src.utils.state import Reference, Action, create_reference_and_action_from_tool_result
from src.tools.vdb import vdb_search

# 프롬프트
from src.prompts.expert_information_instruction import EXPERT_INFORMATION_INSTRUCTION

# RAG 답변 생성용 사용자 프롬프트 템플릿
EXPERT_USER_PROMPT_TEMPLATE = """## 사용자 질문
{user_question}

## 검색된 컨텍스트
{retrieved_context}

위 컨텍스트를 바탕으로 사용자의 질문에 답변해주세요. 출처를 반드시 명시하세요."""


logger = get_logger(__name__)


def build_expert_context_from_hits(hits: List[Dict[str, Any]]) -> str:
    """
    Qdrant 검색 결과에서 LLM 컨텍스트 문자열을 생성한다.
    
    Args:
        hits: vdb_search 결과 리스트 (각 항목에 payload, score 포함)
        
    Returns:
        LLM에 전달할 컨텍스트 문자열
    """
    if not hits:
        return "검색 결과가 없습니다."
    
    context_parts = []
    for i, hit in enumerate(hits, 1):
        payload = hit.get("payload", {})
        text = payload.get("text", "")
        source = payload.get("source", "알 수 없음")
        source_type = payload.get("source_type", "")
        title = payload.get("title", "")
        score = hit.get("score", 0)
        
        context_parts.append(
            f"[문서 {i}] (출처: {source}, 유형: {source_type}, 유사도: {score:.3f})\n"
            f"제목: {title}\n"
            f"내용: {text}\n"
        )
    
    return "\n---\n".join(context_parts)


class ExpertInformationAgent(BaseAgent):
    """
    전문가 문서(hedge fund docs)에서 정보를 검색해 요약해주는 RAG 에이전트.

    구조:
      1) 사용자의 질문을 OpenAI 임베딩으로 변환
      2) vdb_search 툴로 Qdrant에서 관련 청크 검색
      3) 검색된 컨텍스트를 LLM 프롬프트에 넣어 요약/분석
      4) 최종 답변과 hits를 함께 반환
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        super().__init__(
            name="expert_information",
            description="전문가 리포트/문서(hedge docs)를 기반으로 환헤지 관련 정보를 제공하는 에이전트",
        )
        self.embedding_model = embedding_model or getattr(
            settings,
            "openai_embedding_model",
            "text-embedding-3-large",
        )
        self.collection_name = collection_name or getattr(
            settings,
            "qdrant_collection",
            "Hedge_Expert",
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
              "hits": [...],          # Qdrant 검색 결과
              "reference": [...],     # Reference 리스트
              "action": [...],        # Action 리스트
            }
        """
        context = context or {}
        top_k = int(context.get("top_k", 5))

        execution_start_time = time.time()
        execution_id = f"expert_info_{int(execution_start_time * 1000)}"

        logger.info(
            "🚀 [EXECUTION START] 전문가 정보 에이전트 실행 시작",
            {
                "execution_id": execution_id,
                "agent_name": self.name,
                "task": task,
                "task_length": len(task),
                "has_context": bool(context),
                "context_keys": list(context.keys()),
                "top_k": top_k,
            },
        )

        references: List[Reference] = []
        actions: List[Action] = []

        try:
            # 1) 쿼리 임베딩 생성
            embed_start = time.time()
            query_embedding = await self._embed_text(task)
            embed_elapsed = time.time() - embed_start
            logger.info(
                "✅ [EMBEDDING COMPLETE] 임베딩 생성 완료",
                {
                    "execution_id": execution_id,
                    "elapsed_seconds": round(embed_elapsed, 3),
                    "embedding_dim": len(query_embedding) if query_embedding else 0,
                },
            )

            # 2) vdb_search 툴로 Qdrant 검색
            search_start = time.time()
            logger.info(
                "🔍 [VDB SEARCH START] 전문가 문서 검색 시작",
                {
                    "execution_id": execution_id,
                    "collection": self.collection_name,
                    "top_k": top_k,
                },
            )
            hits = await vdb_search.ainvoke(
                {
                    "query_vector": query_embedding,
                    "collection_name": self.collection_name,
                    "limit": top_k,
                }
            )
            search_elapsed = time.time() - search_start

            # Reference / Action 생성 (vdb_search)
            reference, action = create_reference_and_action_from_tool_result(
                tool_name="vdb_search",
                tool_result=hits,
                source="vdb",
                query=f"collection={self.collection_name}, top_k={top_k}",
                input_params={
                    "collection_name": self.collection_name,
                    "limit": top_k,
                },
                metadata={
                    "collection_name": self.collection_name,
                    "top_k": top_k,
                },
            )
            references.append(reference)
            actions.append(action)

            # hits가 비었으면 그대로 안내 메시지 반환
            if not hits:
                msg = (
                    "전문가 문서 벡터 DB에서 관련된 내용을 찾지 못했습니다. "
                    "지금은 일반적인 금융 지식 수준에서만 답변이 가능할 것 같습니다."
                )
                logger.warning(
                    "⚠️ [VDB SEARCH EMPTY] 전문가 문서 검색 결과 없음",
                    {
                        "execution_id": execution_id,
                        "elapsed_seconds": round(search_elapsed, 3),
                    },
                )
                return {
                    "agent": self.name,
                    "task": task,
                    "status": "success",
                    "answer": msg,
                    "hits": [],
                    "reference": [ref.__dict__ for ref in references],
                    "action": [act.__dict__ for act in actions],
                }

            logger.info(
                "✅ [VDB SEARCH COMPLETE] 전문가 문서 검색 완료",
                {
                    "execution_id": execution_id,
                    "elapsed_seconds": round(search_elapsed, 3),
                    "hits_count": len(hits),
                },
            )

            # 3) LLM 컨텍스트 문자열 생성
            context_text = build_expert_context_from_hits(hits)

            # 4) LLM 프롬프트 구성
            system_prompt = EXPERT_INFORMATION_INSTRUCTION

            user_prompt = EXPERT_USER_PROMPT_TEMPLATE.format(
                user_question=task,
                retrieved_context=context_text,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # 5) LLM 호출
            llm_start = time.time()
            answer = await self._call_llm(messages, temperature=0.2)
            llm_elapsed = time.time() - llm_start

            logger.info(
                "✅ [LLM COMPLETE] 전문가 답변 생성 완료",
                {
                    "execution_id": execution_id,
                    "elapsed_seconds": round(llm_elapsed, 3),
                    "answer_length": len(answer) if answer else 0,
                },
            )

            total_elapsed = time.time() - execution_start_time
            logger.info(
                "🎉 [EXECUTION COMPLETE] 전문가 정보 에이전트 실행 완료",
                {
                    "execution_id": execution_id,
                    "total_elapsed_seconds": round(total_elapsed, 3),
                    "status": "success",
                },
            )

            return {
                "agent": self.name,
                "task": task,
                "status": "success",
                "answer": answer,
                "hits": hits,
                "reference": [ref.__dict__ for ref in references],
                "action": [act.__dict__ for act in actions],
            }

        except Exception as e:
            total_elapsed = time.time() - execution_start_time
            logger.error(
                "❌ [EXECUTION ERROR] ExpertInformationAgent.execute failed",
                {
                    "execution_id": execution_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "total_elapsed_seconds": round(total_elapsed, 3),
                },
                exc_info=True,
            )
            return {
                "agent": self.name,
                "task": task,
                "status": "error",
                "answer": f"전문가 정보 검색 중 오류가 발생했습니다: {e}",
                "hits": [],
                "reference": [ref.__dict__ for ref in references],
                "action": [act.__dict__ for act in actions],
            }

    async def _embed_text(self, text: str) -> List[float]:
        """
        OpenAI 임베딩 API를 사용해 쿼리 텍스트를 벡터로 변환.
        """
        logger.info(
            f"📊 임베딩 생성 중",
            {"model": self.embedding_model},
        )
        # OpenAI 클라이언트는 동기식이므로 create 사용
        res = self._embed_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return res.data[0].embedding
