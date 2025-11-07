"""RAG 에이전트"""
import json
import re
from typing import Dict, Any, Optional, List
from src.agents.base_agent import BaseAgent
from src.tools.database import DatabaseTool
# from src.tools.qdrant_client import QdrantTool  # Qdrant 사용 안 함
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
        # self.qdrant_tool = QdrantTool()  # Qdrant 사용 안 함
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
            # 1. 날짜 정보 추출 및 적절한 데이터베이스 메서드 선택
            db_results = await self._query_exchange_rate_data(task)
            
            # 2. Qdrant 벡터 검색 (사용 안 함 - 주석처리)
            # vector_results = await self._perform_vector_search(task)
            vector_results = []  # Qdrant 사용 안 함
            
            # 3. 결과를 텍스트로 변환
            db_results_text = json.dumps(db_results, ensure_ascii=False, indent=2) if db_results else "조회된 데이터가 없습니다."
            
            # 4. LLM을 통해 최종 답변 생성
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": RAG_USER_PROMPT_TEMPLATE.format(
                        user_query=task,
                        db_results=db_results_text
                    )
                }
            ]
            
            answer = await self._call_llm(messages, temperature=0.7)
            
            return {
                "agent": self.name,
                "task": task,
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
    
    async def _query_exchange_rate_data(self, task: str) -> List[Dict[str, Any]]:
        """사용자 질문을 분석하여 적절한 환율 데이터 조회"""
        # 날짜 패턴 추출 (YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD 등)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}/\d{2}/\d{2}',   # YYYY/MM/DD
            r'\d{4}\.\d{2}\.\d{2}', # YYYY.MM.DD
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',  # YYYY년 MM월 DD일
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, task)
            for match in matches:
                # 날짜 형식 정규화
                normalized_date = self._normalize_date(match)
                if normalized_date:
                    dates.append(normalized_date)
        
        # 날짜 범위 추출 (시작일-종료일, 시작일~종료일 등)
        range_patterns = [
            r'(\d{4}-\d{2}-\d{2})\s*[-~부터]\s*(\d{4}-\d{2}-\d{2})',
            r'(\d{4}/\d{2}/\d{2})\s*[-~부터]\s*(\d{4}/\d{2}/\d{2})',
        ]
        
        date_range = None
        for pattern in range_patterns:
            match = re.search(pattern, task)
            if match:
                start_date = self._normalize_date(match.group(1))
                end_date = self._normalize_date(match.group(2))
                if start_date and end_date:
                    date_range = (start_date, end_date)
                    break
        
        # 질문 유형 분석
        task_lower = task.lower()
        
        # 특정 날짜 조회
        if dates:
            if len(dates) == 1:
                return await self.db_tool.get_exchange_rate_by_date(dates[0])
            elif len(dates) == 2:
                return await self.db_tool.get_exchange_rate_range(dates[0], dates[1])
        
        # 날짜 범위 조회
        if date_range:
            return await self.db_tool.get_exchange_rate_range(date_range[0], date_range[1])
        
        # 최신 데이터 조회 키워드
        if any(keyword in task_lower for keyword in ['최신', '최근', 'latest', 'recent', '현재', '오늘']):
            limit = 10
            # 숫자 추출 (예: "최근 5개")
            limit_match = re.search(r'(\d+)\s*개', task)
            if limit_match:
                limit = int(limit_match.group(1))
            return await self.db_tool.get_latest_exchange_rate(limit)
        
        # 날짜가 명시되지 않은 경우 LLM을 통해 날짜 추출 시도
        extracted_date = await self._extract_date_with_llm(task)
        if extracted_date:
            return await self.db_tool.get_exchange_rate_by_date(extracted_date)
        
        # 기본값: 최신 10개 데이터
        return await self.db_tool.get_latest_exchange_rate(10)
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """다양한 날짜 형식을 YYYY-MM-DD 형식으로 정규화"""
        try:
            # YYYY-MM-DD 형식
            if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                return date_str
            
            # YYYY/MM/DD 형식
            if re.match(r'\d{4}/\d{2}/\d{2}', date_str):
                return date_str.replace('/', '-')
            
            # YYYY.MM.DD 형식
            if re.match(r'\d{4}\.\d{2}\.\d{2}', date_str):
                return date_str.replace('.', '-')
            
            # YYYY년 MM월 DD일 형식
            match = re.match(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', date_str)
            if match:
                year = match.group(1)
                month = match.group(2).zfill(2)
                day = match.group(3).zfill(2)
                return f"{year}-{month}-{day}"
            
            return None
        except Exception:
            return None
    
    async def _extract_date_with_llm(self, task: str) -> Optional[str]:
        """LLM을 통해 질문에서 날짜 정보 추출"""
        prompt = f"""사용자 질문: {task}

위 질문에서 날짜 정보를 추출하여 YYYY-MM-DD 형식으로 반환해주세요.
날짜 정보가 없거나 불명확한 경우 "없음"이라고만 답변해주세요.

예시:
- "2024년 1월 15일" -> 2024-01-15
- "2024-01-15" -> 2024-01-15
- "어제", "오늘" 같은 상대적 표현 -> 없음

날짜만 반환하세요 (설명 없이)."""
        
        messages = [
            {"role": "system", "content": "당신은 날짜 추출 전문가입니다. 날짜만 반환하세요."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            result = await self._call_llm(messages, temperature=0.3)
            result = result.strip()
            
            if result == "없음" or not re.match(r'\d{4}-\d{2}-\d{2}', result):
                return None
            
            return result
        except Exception:
            return None
    
    # Qdrant 사용 안 함 - 주석처리
    # async def _perform_vector_search(self, query: str) -> List[Dict[str, Any]]:
    #     """벡터 검색 수행"""
    #     # 실제로는 query를 임베딩으로 변환해야 함
    #     # 여기서는 더미 벡터 사용
    #     dummy_vector = [0.1] * 384  # 384차원 벡터
    #     results = await self.qdrant_tool.search(dummy_vector, limit=5)
    #     return results

