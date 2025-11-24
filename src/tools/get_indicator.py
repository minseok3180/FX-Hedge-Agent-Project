"""Get Indicator 에이전트"""
import json
import re
from typing import Dict, Any, Optional, List

from src.agents.base_agent import BaseAgent
from src.tools.database import DatabaseTool
from src.query.query_builder import QueryBuilder
from src.prompts.get_indicator_prompt import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE


class GetIndicatorAgent(BaseAgent):
    """
    MariaDB(eiExchangeRate)에서 환율 데이터를 조회하고,
    조회 결과를 컨텍스트로 사용해 LLM RAG를 수행하는 에이전트.
    """

    def __init__(self):
        super().__init__(
            name="get_indicator",
            description="데이터베이스에서 환율 및 지표 데이터를 조회하고 RAG를 통해 정보를 제공하는 에이전트"
        )
        self.db_tool = DatabaseTool()
        # 현재는 직접 SQL 메서드를 사용하지만, 향후 복잡한 쿼리 생성 시 QueryBuilder 활용 가능
        self.query_builder = QueryBuilder()

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        데이터베이스 쿼리 및 RAG 수행

        Args:
            task: 수행할 작업 설명 (사용자 질문)
            context: 추가 컨텍스트 (테이블명, 쿼리 옵션 등 - 현재는 미사용)

        Returns:
            쿼리 및 RAG 결과 딕셔너리
        """
        try:
            # 1. 사용자 질문을 분석하여 적절한 DB 조회 수행
            db_results = await self._query_exchange_rate_data(task)

            # 벡터 DB(Qdrant)는 현재 사용하지 않음
            vector_results: List[Dict[str, Any]] = []

            # 2. DB 결과를 사람이 읽기 쉬운 컨텍스트 텍스트로 변환
            db_context_text = self._build_db_context_text(db_results)

            # 3. LLM 입력 메시지 구성
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": RAG_USER_PROMPT_TEMPLATE.format(
                        user_query=task,
                        db_results=db_context_text
                    )
                }
            ]

            # 4. LLM 호출
            answer = await self._call_llm(messages, temperature=0.7)

            return {
                "agent": self.name,
                "task": task,
                "db_results": db_results,         # 원시 데이터 (JSON 직렬화 가능한 dict 리스트)
                "vector_results": vector_results, # 현재는 항상 빈 리스트
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
        """
        사용자 질문을 분석하여 적절한 환율 데이터 조회.

        우선 정규표현식으로 날짜/범위를 파싱하고,
        없으면 LLM을 통해 날짜를 추출한 뒤 DB를 조회한다.
        마지막까지도 날짜가 없으면 최신 n개(기본 10개)를 조회한다.
        """
        # 1) 다양한 날짜 형식 추출
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",                 # YYYY-MM-DD
            r"\d{4}/\d{2}/\d{2}",                 # YYYY/MM/DD
            r"\d{4}\.\d{2}\.\d{2}",               # YYYY.MM.DD
            r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일",   # YYYY년 MM월 DD일
        ]

        dates: List[str] = []
        for pattern in date_patterns:
            matches = re.findall(pattern, task)
            for match in matches:
                normalized_date = self._normalize_date(match)
                if normalized_date:
                    dates.append(normalized_date)

        # 2) 날짜 범위 패턴 (시작일-종료일, 시작일~종료일 등)
        range_patterns = [
            r"(\d{4}-\d{2}-\d{2})\s*[-~부터]\s*(\d{4}-\d{2}-\d{2})",
            r"(\d{4}/\d{2}/\d{2})\s*[-~부터]\s*(\d{4}/\d{2}/\d{2})",
        ]

        date_range: Optional[tuple] = None
        for pattern in range_patterns:
            match = re.search(pattern, task)
            if match:
                start_date = self._normalize_date(match.group(1))
                end_date = self._normalize_date(match.group(2))
                if start_date and end_date:
                    date_range = (start_date, end_date)
                    break

        task_lower = task.lower()

        # 3) 특정 날짜(1개) 조회
        if dates:
            if len(dates) == 1:
                return await self.db_tool.get_exchange_rate_by_date(dates[0])
            elif len(dates) >= 2:
                # 여러 개가 있는 경우 앞의 두 개를 범위로 사용
                return await self.db_tool.get_exchange_rate_range(dates[0], dates[1])

        # 4) 범위가 명시된 경우
        if date_range:
            return await self.db_tool.get_exchange_rate_range(date_range[0], date_range[1])

        # 5) 최신 데이터 조회 키워드
        if any(keyword in task_lower for keyword in ["최신", "최근", "latest", "recent", "현재", "오늘"]):
            limit = 10
            limit_match = re.search(r"(\d+)\s*개", task)
            if limit_match:
                try:
                    limit = int(limit_match.group(1))
                except ValueError:
                    pass
            return await self.db_tool.get_latest_exchange_rate(limit)

        # 6) 날짜가 명시되지 않은 경우 LLM을 통해 날짜 추출 시도
        extracted_date = await self._extract_date_with_llm(task)
        if extracted_date:
            return await self.db_tool.get_exchange_rate_by_date(extracted_date)

        # 7) 그래도 정보가 없으면 기본값: 최신 10개 데이터
        return await self.db_tool.get_latest_exchange_rate(10)

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        다양한 날짜 형식을 YYYY-MM-DD 형식으로 정규화.
        """
        try:
            # YYYY-MM-DD
            if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                return date_str

            # YYYY/MM/DD
            if re.match(r"\d{4}/\d{2}/\d{2}", date_str):
                return date_str.replace("/", "-")

            # YYYY.MM.DD
            if re.match(r"\d{4}\.\d{2}\.\d{2}", date_str):
                return date_str.replace(".", "-")

            # YYYY년 MM월 DD일
            match = re.match(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", date_str)
            if match:
                year = match.group(1)
                month = match.group(2).zfill(2)
                day = match.group(3).zfill(2)
                return f"{year}-{month}-{day}"

            return None
        except Exception:
            return None

    async def _extract_date_with_llm(self, task: str) -> Optional[str]:
        """
        LLM을 통해 질문에서 날짜 정보 추출.

        상대적인 표현(어제, 오늘 등)은 처리하지 않고,
        명시적인 YYYY-MM-DD 형식만 허용한다.
        """
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

            if result == "없음" or not re.match(r"\d{4}-\d{2}-\d{2}", result):
                return None

            return result
        except Exception:
            return None

    def _build_db_context_text(self, db_results: List[Dict[str, Any]]) -> str:
        """
        DB 조회 결과를 LLM에 넘길 수 있는 컨텍스트 텍스트로 변환.

        DocsAgent의 _build_context_text와 유사하게,
        각 행을 [데이터 n] 블록으로 정리한다.
        """
        if not db_results:
            return "조회된 데이터가 없습니다."

        blocks: List[str] = []
        for idx, row in enumerate(db_results, start=1):
            lines = [f"[데이터 {idx}]"]
            for key, value in row.items():
                lines.append(f"{key}: {value}")
            block = "\n".join(lines)
            blocks.append(block)

        return "\n\n".join(blocks)
