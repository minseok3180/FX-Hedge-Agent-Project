"""RDB 소프트 쿼리 툴 - LLM이 쿼리문을 직접 작성하여 RDB에서 데이터 가져오기"""
import json
import re
from typing import List, Dict, Any, Optional
from src.tools.database import DatabaseTool
from src.utils.settings import settings
from src.utils.logger import get_logger
from src.utils.openai_tracer import TracedOpenAIClient


# 데이터베이스 메타데이터
DB_METADATA = {
    "database_name": "fx_hedge",
    "tables": {
        "eiExchangeRate": {
            "description": "환율 및 경제 지표 데이터 테이블",
            "columns": {
                "date": {"type": "DATE", "description": "날짜 (YYYY-MM-DD)"},
                "usdkrw": {"type": "DECIMAL", "description": "USD/KRW 환율"},
                # 실제 컬럼명은 DB 스키마에 따라 다를 수 있음
                # 이미지 기준으로는 한글 컬럼명이 보이지만, 실제 영어 컬럼명 사용 가능
            }
        },
        "user_info": {
            "description": "사용자 정보 테이블",
            "columns": {
                "user_id": {"type": "VARCHAR", "description": "사용자 ID"},
                "user_name": {"type": "VARCHAR", "description": "사용자 이름"},
                "user_krw": {"type": "DECIMAL", "description": "사용자 보유 KRW 금액"},
                "user_usd": {"type": "DECIMAL", "description": "사용자 보유 USD 금액"},
            }
        }
    }
}

# 퓨샷 예시
FEW_SHOT_EXAMPLES = [
    {
        "user_request": "2024-01-15의 USD/KRW 환율을 조회해줘",
        "sql_query": "SELECT date, usdkrw FROM eiExchangeRate WHERE date = '2024-01-15' LIMIT 1",
        "explanation": "특정 날짜의 환율만 조회하는 간단한 쿼리"
    },
    {
        "user_request": "최근 10일간의 환율과 기준금리 데이터를 가져와줘",
        "sql_query": "SELECT date, usdkrw, base FROM eiExchangeRate ORDER BY date DESC LIMIT 10",
        "explanation": "최신 데이터를 날짜 내림차순으로 정렬하여 조회"
    },
    {
        "user_request": "2024년 1월부터 3월까지의 평균 환율을 계산해줘",
        "sql_query": "SELECT AVG(usdkrw) as avg_rate FROM eiExchangeRate WHERE date BETWEEN '2024-01-01' AND '2024-03-31'",
        "explanation": "날짜 범위를 지정하고 집계 함수를 사용하여 평균 계산"
    },
    {
        "user_request": "미국 금리가 5% 이상인 날짜들의 환율을 조회해줘",
        "sql_query": "SELECT date, usdkrw, us_interest FROM eiExchangeRate WHERE us_interest >= 5.0 ORDER BY date DESC",
        "explanation": "조건문을 사용하여 특정 조건을 만족하는 데이터만 필터링"
    }
]


class RDBSoftTool:
    """LLM이 쿼리문을 직접 작성하여 RDB에서 데이터를 가져오는 툴"""
    
    def __init__(self):
        self.db_tool = DatabaseTool()
        self.logger = get_logger("rdb_soft_tool")
        self.client = TracedOpenAIClient(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    def _build_system_prompt(self) -> str:
        """SQL 쿼리 생성용 시스템 프롬프트 생성"""
        metadata_str = json.dumps(DB_METADATA, ensure_ascii=False, indent=2)
        
        examples_str = "\n\n".join([
            f"예시 {i+1}:\n"
            f"사용자 요청: {ex['user_request']}\n"
            f"생성된 SQL: {ex['sql_query']}\n"
            f"설명: {ex['explanation']}"
            for i, ex in enumerate(FEW_SHOT_EXAMPLES)
        ])
        
        return f"""당신은 MariaDB 데이터베이스에 대한 SQL 쿼리를 생성하는 전문가입니다.

## 데이터베이스 메타데이터
{metadata_str}

## 쿼리 작성 규칙
1. **보안**: SQL Injection을 방지하기 위해 파라미터화된 쿼리를 사용하지 않고, 직접 값을 넣되 문자열은 작은따옴표로 감싸세요.
2. **날짜 형식**: 날짜는 반드시 'YYYY-MM-DD' 형식을 사용하세요.
3. **테이블명**: 대소문자를 구분하므로 정확한 테이블명을 사용하세요 (eiExchangeRate).
4. **컬럼명**: 정확한 컬럼명을 사용하세요.
5. **LIMIT**: 대량의 데이터 조회 시 반드시 LIMIT을 사용하세요.
6. **SELECT**: 필요한 컬럼만 선택하세요.

## 퓨샷 예시
{examples_str}

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:
{{
    "sql_query": "생성된 SQL 쿼리문",
    "explanation": "쿼리 생성 이유 및 설명"
}}

SQL 쿼리만 반환하되, JSON 형식으로 감싸서 반환하세요."""

    def _build_user_prompt(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        """사용자 요청 기반 프롬프트 생성"""
        prompt = f"사용자 요청: {user_request}\n\n"
        
        if context:
            prompt += f"추가 컨텍스트:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        
        prompt += "위 요청에 맞는 SQL 쿼리를 생성해주세요."
        return prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """LLM 응답에서 SQL 쿼리 추출"""
        # JSON 형식으로 응답이 오는 경우
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "sql_query" in parsed:
                return parsed["sql_query"].strip()
        except json.JSONDecodeError:
            pass
        
        # SQL 쿼리 패턴 찾기 (SELECT로 시작하는 부분, 세미콜론까지 또는 끝까지)
        sql_pattern = r"(SELECT\s+.*?)(?:;|\n\n|$)"
        match = re.search(sql_pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            # 세미콜론 제거
            if sql.endswith(';'):
                sql = sql[:-1].strip()
            return sql
        
        # 전체 응답이 SQL인 경우
        response_clean = response.strip()
        if response_clean.upper().startswith("SELECT"):
            if response_clean.endswith(';'):
                response_clean = response_clean[:-1].strip()
            return response_clean
        
        return response.strip()
    
    async def generate_and_execute(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        LLM을 사용하여 SQL 쿼리를 생성하고 실행
        
        Args:
            user_request: 사용자 요청 (자연어)
            context: 추가 컨텍스트 정보
            
        Returns:
            쿼리 실행 결과 및 메타데이터
        """
        self.logger.info(
            f"🔍 SQL 쿼리 생성 시작",
            {"user_request": user_request, "has_context": context is not None}
        )
        
        try:
            # 1. LLM을 통해 SQL 쿼리 생성
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(user_request, context)}
            ]
            
            self.logger.debug("🤖 LLM 호출 중...")
            response = self.client.chat_completions_create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            llm_response = response.choices[0].message.content
            self.logger.debug(f"📥 LLM 응답 수신", {"response_preview": llm_response[:200]})
            
            # 2. SQL 쿼리 추출
            try:
                parsed_response = json.loads(llm_response)
                sql_query = parsed_response.get("sql_query", "")
                explanation = parsed_response.get("explanation", "")
            except json.JSONDecodeError:
                sql_query = self._extract_sql_from_response(llm_response)
                explanation = "LLM이 생성한 쿼리"
            
            if not sql_query:
                raise ValueError("생성된 SQL 쿼리가 비어있습니다.")
            
            self.logger.info(
                f"✅ SQL 쿼리 생성 완료",
                {"sql_query": sql_query, "explanation": explanation}
            )
            
            # 3. 쿼리 실행
            self.logger.debug("🚀 쿼리 실행 중...")
            results = await self.db_tool.execute_query(sql_query)
            
            self.logger.info(
                f"✅ 쿼리 실행 완료",
                {"results_count": len(results)}
            )
            
            return {
                "sql_query": sql_query,
                "explanation": explanation,
                "results": results,
                "results_count": len(results),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(
                f"❌ SQL 쿼리 생성/실행 실패",
                {"user_request": user_request, "error": str(e)},
                exc_info=True
            )
            return {
                "sql_query": "",
                "explanation": "",
                "results": [],
                "results_count": 0,
                "error": str(e),
                "status": "error"
            }

