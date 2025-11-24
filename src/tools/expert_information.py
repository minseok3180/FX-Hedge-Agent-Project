"""Hedge 관련 문서 RAG 에이전트"""

from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

# .env 로드 (프로젝트 루트 기준)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

from src.agents.base_agent import BaseAgent
from src.tools.qdrant_client import QdrantTool
from src.prompts.expert_information_prompt import DOCS_SYSTEM_PROMPT, DOCS_USER_PROMPT_TEMPLATE


class ExpertInformationAgent(BaseAgent):
    """
    Qdrant에 적재된 헤지 관련 리포트(PDF/웹 문서)를 기반으로
    질의응답을 수행하는 에이전트.
    """

    def __init__(
        self,
        collection_name: str = "hedge_fund_docs",  # ingest_hedge_docs.py와 동일 컬렉션
        embedding_model: str = "text-embedding-3-large",  # ingest에서 사용한 모델과 동일
        top_k: int = 5,
    ):
        super().__init__(
            name="expert_information",
            description="Qdrant에 저장된 통화 헤지 관련 문서를 기반으로 RAG를 수행하는 에이전트",
        )
        # qdrant_client.QdrantTool은 collection 매개변수를 사용하도록 구현되어 있음
        self.qdrant = QdrantTool(collection=collection_name)
        self.embedding_model = embedding_model
        self.top_k = top_k

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        1) 사용자 질문을 임베딩으로 변환
        2) Qdrant에서 유사 문서 검색
        3) 검색된 문서를 컨텍스트로 LLM 호출
        """
        try:
            # 1. 쿼리 임베딩 생성
            query_embedding = await self._embed_text(task)

            # 2. Qdrant에서 유사 문서 검색 (QdrantTool.search는 동기 함수)
            vector_results = self.qdrant.search(
                query_vector=query_embedding,
                top_k=self.top_k,
            )

            # 3. 컨텍스트 텍스트 구성
            context_text = self._build_context_text(vector_results)

            # 컨텍스트가 비어 있으면 그에 맞게 안내
            if not context_text.strip():
                answer = (
                    "현재 Qdrant에 적재된 헤지 관련 문서에서 유의미한 컨텍스트를 찾지 못했습니다. "
                    "지금은 일반적인 금융 지식을 바탕으로 개념적인 설명만 제공할 수 있습니다.\n\n"
                    f"질문: {task}"
                )
                return {
                    "agent": self.name,
                    "task": task,
                    "answer": answer,
                    "hits": [],
                    "status": "success",
                }

            # 4. LLM 호출
            messages = [
                {"role": "system", "content": DOCS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": DOCS_USER_PROMPT_TEMPLATE.format(
                        user_query=task,
                        context=context_text,
                    ),
                },
            ]

            answer = await self._call_llm(messages, temperature=0.5)

            return {
                "agent": self.name,
                "task": task,
                "answer": answer,
                "hits": vector_results,
                "status": "success",
            }

        except Exception as e:
            return {
                "agent": self.name,
                "task": task,
                "error": str(e),
                "status": "error",
            }

    async def _embed_text(self, text: str) -> List[float]:
        """
        OpenAI 임베딩 API를 사용해 텍스트를 벡터로 변환.
        ingest_hedge_docs.py에서 사용한 것과 동일한 임베딩 모델을 써야 한다.
        BaseAgent에서 self.client(OpenAI 클라이언트)를 갖고 있다고 가정.
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _build_context_text(self, results: List[Dict[str, Any]]) -> str:
        """
        Qdrant 검색 결과를 사람이 읽을 수 있는 컨텍스트 텍스트로 변환.
        ingest_hedge_docs.py에서 payload에 넣은 필드를 그대로 사용한다
        (text, title, source, url, chunk_id 등).
        """
        if not results:
            return ""

        blocks = []
        for idx, item in enumerate(results, start=1):
            payload = item.get("payload", {}) or {}
            score = item.get("score", 0.0)

            title = payload.get("title", "제목 없음")
            text = payload.get("text", "")
            source = payload.get("source", "")
            url = payload.get("url", None)
            chunk_id = payload.get("chunk_id", None)

            meta_parts = []
            if source:
                meta_parts.append(f"출처: {source}")
            if chunk_id is not None:
                meta_parts.append(f"청크 ID: {chunk_id}")
            if url:
                meta_parts.append(f"URL: {url}")
            meta_str = " / ".join(meta_parts) if meta_parts else ""

            block_lines = [
                f"[문서 {idx}] (score={score:.4f})",
                f"제목: {title}",
                meta_str,
                f"본문: {text}",
            ]
            # 공백 라인 제거
            block = "\n".join([line for line in block_lines if line.strip()])
            blocks.append(block)

        return "\n\n".join(blocks)
