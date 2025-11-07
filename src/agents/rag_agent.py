"""RAG 에이전트"""
import json
from typing import Dict, Any, Optional, List
from src.agents.base_agent import BaseAgent
from src.tools.database import DatabaseTool
from src.tools.qdrant_client import QdrantTool
from src.query.query_builder import QueryBuilder
from src.prompts.rag_prompt import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE


class RAGAgent(BaseAgent):
    """데이터베이스 조회 및 RAG를 수행하는 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="rag_agent",
            description="데이터베이스에서 데이터를 조회하고 RAG를 통해 정보를 제공하는 에이전트"
        )
        self.db_tool = DatabaseTool()
        self.qdrant_tool = QdrantTool()
        self.query_builder = QueryBuilder()
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        데이터베이스 쿼리 및 RAG 수행
        
        Args:
            task: 수행할 작업 설명
            context: 추가 컨텍스트 (테이블명, 쿼리 등)
            
        Returns:
            쿼리 및 RAG 결과
        """
        try:
            # 1. LLM을 통해 SQL 쿼리 생성 (간단한 예시)
            # 실제로는 더 정교한 쿼리 생성 로직이 필요
            sql_query = await self._generate_sql_query(task, context)
            
            # 2. 데이터베이스 쿼리 실행
            db_results = await self.db_tool.execute_query(sql_query)
            
            # 3. Qdrant 벡터 검색 (간단한 예시 - 실제로는 임베딩 생성 필요)
            # 여기서는 더미 벡터 사용
            vector_results = await self._perform_vector_search(task)
            
            # 4. 결과를 텍스트로 변환
            db_results_text = json.dumps(db_results, ensure_ascii=False, indent=2)
            vector_results_text = json.dumps(vector_results, ensure_ascii=False, indent=2)
            
            # 5. LLM을 통해 최종 답변 생성
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": RAG_USER_PROMPT_TEMPLATE.format(
                        user_query=task,
                        db_results=db_results_text,
                        vector_results=vector_results_text
                    )
                }
            ]
            
            answer = await self._call_llm(messages, temperature=0.7)
            
            return {
                "agent": self.name,
                "task": task,
                "sql_query": sql_query,
                "db_results": db_results,
                "vector_results": vector_results,
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
    
    async def _generate_sql_query(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """LLM을 통해 SQL 쿼리 생성"""
        # 간단한 예시 - 실제로는 더 정교한 프롬프트 필요
        prompt = f"""사용자 질문: {task}

위 질문에 답하기 위한 SQL 쿼리를 생성해주세요.
테이블명은 'fx_data'라고 가정하고, 가능한 컬럼은 date, currency_pair, rate, volume 등입니다.

SQL 쿼리만 반환해주세요 (SELECT 문)."""
        
        messages = [
            {"role": "system", "content": "당신은 SQL 쿼리 생성 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
        
        query = await self._call_llm(messages, temperature=0.3)
        # SQL 쿼리만 추출 (마크다운 코드 블록 제거)
        query = query.strip().replace("```sql", "").replace("```", "").strip()
        return query
    
    async def _perform_vector_search(self, query: str) -> List[Dict[str, Any]]:
        """벡터 검색 수행"""
        # 실제로는 query를 임베딩으로 변환해야 함
        # 여기서는 더미 벡터 사용
        dummy_vector = [0.1] * 384  # 384차원 벡터
        results = await self.qdrant_tool.search(dummy_vector, limit=5)
        return results

