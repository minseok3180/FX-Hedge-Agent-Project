"""RDB 도구 - 데이터베이스 연결, 쿼리 실행, 데이터 수정 통합 모듈"""
import pymysql
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from src.utils.settings import settings
from src.utils.logger import get_logger
from src.query.rdb_hard_queries import rdb_hard_queries
from src.utils.tools import (
    tool,
    handle_tool_error,
    ToolError,
    RDBQueryHardInput,
    RDBQueryLLMInput,
    RDBModifyInput
)

logger = get_logger("rdb-tool")

# 데이터베이스 메타데이터 (LLM 쿼리 생성용)
DB_METADATA = {
    "database_name": "fx_hedge",
    "tables": {
        "eiExchangeRate": {
            "description": "환율 및 경제 지표 데이터 테이블",
            "columns": {
                "date": {"type": "DATE", "description": "날짜 (YYYY-MM-DD)"},
                "usdkrw": {"type": "DOUBLE", "description": "USD/KRW 환율"},
                "미국수출금액": {"type": "DOUBLE", "description": "미국 수출 금액"},
                "미국수입금액": {"type": "DOUBLE", "description": "미국 수입 금액"},
                "외환보유액": {"type": "DOUBLE", "description": "외환 보유액"},
                "미국외환보유액": {"type": "DOUBLE", "description": "미국 외환 보유액"},
                "한국은행기준금리": {"type": "DOUBLE", "description": "한국은행 기준금리"},
                "정부대출금금리": {"type": "DOUBLE", "description": "정부 대출금 금리"},
                "시장금리": {"type": "DOUBLE", "description": "시장 금리"},
                "소비자물가지수": {"type": "DOUBLE", "description": "소비자 물가지수"},
                "수출물가지수": {"type": "DOUBLE", "description": "수출 물가지수"},
                "수입물가지수": {"type": "DOUBLE", "description": "수입 물가지수"},
                "경제성장률": {"type": "DOUBLE", "description": "경제 성장률"},
                "미국경제성장률": {"type": "DOUBLE", "description": "미국 경제 성장률"},
                "gdp": {"type": "DOUBLE", "description": "GDP"},
                "us_gdp": {"type": "DOUBLE", "description": "미국 GDP"},
                "주가지수": {"type": "DOUBLE", "description": "주가지수"},
                "미국주가지수": {"type": "DOUBLE", "description": "미국 주가지수"},
                "한국금리": {"type": "DOUBLE", "description": "한국 금리"},
                "미국금리": {"type": "DOUBLE", "description": "미국 금리"},
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

# 퓨샷 예시 (LLM 쿼리 생성용)
FEW_SHOT_EXAMPLES = [
    {
        "user_request": "2024-01-15의 USD/KRW 환율을 조회해줘",
        "sql_query": "SELECT date, usdkrw FROM eiExchangeRate WHERE date = '2024-01-15' LIMIT 1",
        "explanation": "특정 날짜의 환율만 조회하는 간단한 쿼리"
    },
    {
        "user_request": "최근 10일간의 환율과 기준금리 데이터를 가져와줘",
        "sql_query": "SELECT date, usdkrw, 한국은행기준금리 FROM eiExchangeRate ORDER BY date DESC LIMIT 10",
        "explanation": "최신 데이터를 날짜 내림차순으로 정렬하여 조회"
    },
    {
        "user_request": "2024년 1월부터 3월까지의 평균 환율을 계산해줘",
        "sql_query": "SELECT AVG(usdkrw) as avg_rate FROM eiExchangeRate WHERE date BETWEEN '2024-01-01' AND '2024-03-31'",
        "explanation": "날짜 범위를 지정하고 집계 함수를 사용하여 평균 계산"
    },
    {
        "user_request": "미국 금리가 5% 이상인 날짜들의 환율을 조회해줘",
        "sql_query": "SELECT date, usdkrw, 미국금리 FROM eiExchangeRate WHERE 미국금리 >= 5.0 ORDER BY date DESC",
        "explanation": "조건문을 사용하여 특정 조건을 만족하는 데이터만 필터링"
    }
]


class DatabaseConnection:
    """데이터베이스 연결 관리 클래스"""
    
    def __init__(self):
        self.logger = get_logger("database-connection")
        self._connection: Optional[pymysql.Connection] = None
    
    def _get_connection(self) -> pymysql.Connection:
        """
        데이터베이스 연결 반환 (재사용 또는 새로 생성)
        
        Returns:
            pymysql.Connection 객체
        """
        # 연결이 없거나 닫혀있으면 새로 생성
        try:
            if self._connection is None or not self._connection.open:
                self._connection = pymysql.connect(
                    host=settings.db_host,
                    port=settings.db_port,
                    user=settings.db_user,
                    password=settings.db_password,
                    database=settings.db_name,
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=False,
                    connect_timeout=10,
                    read_timeout=30,
                    write_timeout=30
                )
                self.logger.debug("✅ 데이터베이스 연결 성공")
        except pymysql.Error as e:
            # 연결 오류 시 기존 연결 정리
            if self._connection:
                try:
                    self._connection.close()
                except:
                    pass
                self._connection = None
            
            self.logger.error(
                f"❌ 데이터베이스 연결 실패",
                {
                    "host": settings.db_host,
                    "port": settings.db_port,
                    "database": settings.db_name,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
        except Exception as e:
            self.logger.error(
                f"❌ 데이터베이스 연결 중 예상치 못한 오류",
                {"error": str(e)},
                exc_info=True
            )
            raise
        
        return self._connection
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self._connection and self._connection.open:
            self._connection.close()
            self._connection = None
            self.logger.debug("🔌 데이터베이스 연결 종료")


# 전역 데이터베이스 연결 인스턴스
_db_connection = DatabaseConnection()


def _resolve_placeholders(query: str, state: Optional[Dict[str, Any]] = None) -> str:
    """
    쿼리의 placeholder를 state에서 가져온 값으로 치환
    
    Args:
        query: SQL 쿼리 문자열
        state: AgentState의 딕셔너리 (user_id, date 등 포함)
        
    Returns:
        placeholder가 치환된 쿼리 문자열
        
    Raises:
        ValueError: 필수 placeholder가 없을 때
    """
    if not state:
        # state가 없으면 placeholder가 있으면 오류 발생
        if re.search(r"\{(\w+)\}", query):
            raise ValueError("쿼리에 placeholder가 있지만 state가 제공되지 않았습니다.")
        return query
    
    # 지원하는 placeholder 목록
    placeholders = {
        "user_id": state.get("user_id") or state.get("current_context", {}).get("user_id"),
        "date": state.get("date") or state.get("current_context", {}).get("date")
    }
    
    # conversation_history에서 최신 date 추출
    if not placeholders.get("date"):
        history = state.get("conversation_history", [])
        if history:
            latest_turn = history[-1]
            placeholders["date"] = latest_turn.get("date")
    
    # user_id는 state의 user_id에서 추출
    if not placeholders.get("user_id"):
        placeholders["user_id"] = state.get("user_id")
    
    # Placeholder 치환
    resolved_query = query
    missing_placeholders = []
    
    for key, value in placeholders.items():
        # 쿼리에 해당 placeholder가 있는지 확인
        if re.search(f"\\{{{key}\\}}", query):
            if value:
                # {key} 형식의 placeholder 치환
                pattern = f"\\{{{key}\\}}"
                resolved_query = re.sub(pattern, str(value), resolved_query)
                logger.debug(
                    f"Placeholder 치환: {{{key}}} → {value}",
                    {"placeholder": key, "value": value}
                )
            else:
                missing_placeholders.append(key)
    
    # 필수 placeholder가 없으면 오류 발생
    if missing_placeholders:
        raise ValueError(
            f"필수 placeholder가 state에 없습니다: {', '.join(missing_placeholders)}. "
            f"state: {state}"
        )
    
    return resolved_query


@tool(args_schema=RDBQueryHardInput)
@handle_tool_error("rdb_query_hard")
async def rdb_query_hard(
    query_key: str,
    params: Optional[Tuple[Any, ...]] = None,
    state: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    하드코딩된 쿼리를 사용하여 RDB에서 데이터 조회
    
    Args:
        query_key: 실행할 쿼리의 키 (예: "get_by_date", "get_by_range", "get_latest")
        params: 쿼리 파라미터 (튜플) - placeholder 사용 시 None
        state: AgentState 딕셔너리 (placeholder 치환용)
        
    Returns:
        쿼리 결과 리스트
    """
    if query_key not in rdb_hard_queries:
        available = ", ".join(rdb_hard_queries.keys())
        raise ToolError(
            "rdb_query_hard",
            f"쿼리 키 '{query_key}'를 찾을 수 없습니다. 사용 가능한 쿼리: {available}"
        )
    
    query = rdb_hard_queries[query_key]
    
    # Placeholder가 있는지 확인
    has_placeholder = re.search(r"\{(\w+)\}", query)
    
    try:
        if has_placeholder:
            # Placeholder 치환
            query = _resolve_placeholders(query, state)
            # Placeholder 사용 시 params는 None으로 설정
            params = None
        
        # Tool 호출 및 쿼리 로깅
        logger.info(
            f"🔧 [TOOL CALL] rdb_query_hard 실행",
            {
                "tool_name": "rdb_query_hard",
                "query_key": query_key,
                "has_params": params is not None,
                "params": str(params) if params else None,
                "has_placeholder": bool(has_placeholder),
                "state_provided": state is not None
            }
        )
        
        logger.info(
            f"📝 [QUERY] RDB 쿼리 실행",
            {
                "query_key": query_key,
                "query": query,
                "params": str(params) if params else None
            }
        )
        
        conn = _db_connection._get_connection()
        
        # 쿼리 실행 전 상세 로깅
        logger.debug(
            f"🔍 [QUERY EXECUTION] 쿼리 실행 시작",
            {
                "query_key": query_key,
                "query_preview": query[:200] if len(query) > 200 else query,
                "query_length": len(query),
                "has_params": params is not None,
                "params": str(params) if params else None,
                "has_state": state is not None,
                "connection_open": conn.open if hasattr(conn, 'open') else None
            }
        )
        
        try:
            with conn.cursor() as cursor:
                execute_start = time.time()
                
                if params:
                    logger.debug(
                        f"📝 [QUERY EXECUTE] 파라미터화된 쿼리 실행",
                        {
                            "query_key": query_key,
                            "params": str(params),
                            "params_type": type(params).__name__
                        }
                    )
                    cursor.execute(query, params)
                else:
                    logger.debug(
                        f"📝 [QUERY EXECUTE] 일반 쿼리 실행",
                        {
                            "query_key": query_key,
                            "query": query
                        }
                    )
                    cursor.execute(query)
                
                execute_elapsed = time.time() - execute_start
                logger.debug(
                    f"⏱️  [QUERY EXECUTE] 쿼리 실행 완료",
                    {
                        "query_key": query_key,
                        "execute_elapsed_seconds": round(execute_elapsed, 4),
                        "rowcount": cursor.rowcount if hasattr(cursor, 'rowcount') else None
                    }
                )
                
                fetch_start = time.time()
                results = cursor.fetchall()
                fetch_elapsed = time.time() - fetch_start
                
                logger.debug(
                    f"📊 [QUERY FETCH] 결과 조회 완료",
                    {
                        "query_key": query_key,
                        "fetch_elapsed_seconds": round(fetch_elapsed, 4),
                        "results_count": len(results) if results else 0
                    }
                )
                
                # DictCursor를 사용하므로 결과는 이미 딕셔너리 리스트
                result_list = [dict(row) for row in results] if results else []
                
                logger.info(
                    f"✅ [TOOL RESULT] rdb_query_hard 완료",
                    {
                        "tool_name": "rdb_query_hard",
                        "query_key": query_key,
                        "rows_count": len(result_list),
                        "result_preview": result_list[:3] if result_list else [],
                        "total_execution_time": round(execute_elapsed + fetch_elapsed, 4)
                    }
                )
                
                return result_list
        except pymysql.Error as db_error:
            # 데이터베이스 오류 상세 로깅
            error_info = {
                "query_key": query_key,
                "query": query,
                "query_length": len(query),
                "has_params": params is not None,
                "params": str(params) if params else None,
                "error": str(db_error),
                "error_type": type(db_error).__name__,
                "error_code": db_error.args[0] if db_error.args else None,
                "error_message": db_error.args[1] if len(db_error.args) > 1 else None,
            }
            
            logger.error(
                f"❌ [DATABASE ERROR] 데이터베이스 오류 발생",
                error_info,
                exc_info=True
            )
            
            # 터미널에 직접 출력
            print(f"\n{'='*80}")
            print(f"❌ [DATABASE ERROR] rdb_query_hard 실행 중 오류 발생")
            print(f"{'='*80}")
            print(f"Query Key: {query_key}")
            print(f"Query: {query}")
            if params:
                print(f"Params: {params}")
            print(f"Error Type: {type(db_error).__name__}")
            print(f"Error Code: {db_error.args[0] if db_error.args else 'N/A'}")
            print(f"Error Message: {db_error.args[1] if len(db_error.args) > 1 else str(db_error)}")
            import traceback
            print(f"\nFull Traceback:")
            print(traceback.format_exc())
            print(f"{'='*80}\n")
            
            raise ToolError("rdb_query_hard", f"데이터베이스 오류: {str(db_error)}", db_error)
    except (ValueError, KeyError) as e:
        # Placeholder 관련 오류는 ToolError로 변환 (데코레이터가 처리)
        logger.error(
            f"❌ [PLACEHOLDER ERROR] Placeholder 오류",
            {
                "query_key": query_key,
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        raise
    except Exception as e:
        # 기타 예상치 못한 오류
        logger.error(
            f"❌ [UNEXPECTED ERROR] 예상치 못한 오류",
            {
                "query_key": query_key,
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        
        # 터미널에 직접 출력
        print(f"\n{'='*80}")
        print(f"❌ [UNEXPECTED ERROR] rdb_query_hard 실행 중 예상치 못한 오류")
        print(f"{'='*80}")
        print(f"Query Key: {query_key}")
        print(f"Query: {query}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        import traceback
        print(f"\nFull Traceback:")
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        
        raise


@tool(args_schema=RDBQueryLLMInput)
@handle_tool_error("rdb_query_llm")
async def rdb_query_llm(
    user_request: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    LLM이 쿼리문을 직접 작성하여 RDB에서 데이터 조회
    
    Args:
        user_request: 사용자 요청 (자연어)
        context: 추가 컨텍스트 정보
        
    Returns:
        쿼리 실행 결과 및 메타데이터
    """
    logger.info(
        f"🔍 SQL 쿼리 생성 시작",
        {"user_request": user_request, "has_context": context is not None}
    )
    
    try:
        # LLM을 통해 SQL 쿼리 생성
        from src.utils.llm import call_gpt
        
        metadata_str = json.dumps(DB_METADATA, ensure_ascii=False, indent=2)
        
        examples_str = "\n\n".join([
            f"예시 {i+1}:\n"
            f"사용자 요청: {ex['user_request']}\n"
            f"생성된 SQL: {ex['sql_query']}\n"
            f"설명: {ex['explanation']}"
            for i, ex in enumerate(FEW_SHOT_EXAMPLES)
        ])
        
        system_prompt = f"""당신은 MariaDB 데이터베이스에 대한 SQL 쿼리를 생성하는 전문가입니다.

## 데이터베이스 메타데이터
{metadata_str}

## 중요: 컬럼명 주의사항
**eiExchangeRate 테이블의 컬럼명은 한글을 사용합니다:**
- `date`: 날짜 (DATE)
- `usdkrw`: USD/KRW 환율 (DOUBLE)
- `미국수출금액`: 미국 수출 금액 (DOUBLE)
- `미국수입금액`: 미국 수입 금액 (DOUBLE)
- `외환보유액`: 외환 보유액 (DOUBLE)
- `미국외환보유액`: 미국 외환 보유액 (DOUBLE)
- `한국은행기준금리`: 한국은행 기준금리 (DOUBLE)
- `정부대출금금리`: 정부 대출금 금리 (DOUBLE)
- `시장금리`: 시장 금리 (DOUBLE)
- `소비자물가지수`: 소비자 물가지수 (DOUBLE)
- `수출물가지수`: 수출 물가지수 (DOUBLE)
- `수입물가지수`: 수입 물가지수 (DOUBLE)
- `경제성장률`: 경제 성장률 (DOUBLE)
- `미국경제성장률`: 미국 경제 성장률 (DOUBLE)
- `gdp`: GDP (DOUBLE)
- `us_gdp`: 미국 GDP (DOUBLE)
- `주가지수`: 주가지수 (DOUBLE)
- `미국주가지수`: 미국 주가지수 (DOUBLE)
- `한국금리`: 한국 금리 (DOUBLE)
- `미국금리`: 미국 금리 (DOUBLE)

**절대 영어 컬럼명(base, us_interest, reserve 등)을 사용하지 마세요. 반드시 한글 컬럼명을 사용하세요.**

## 쿼리 작성 규칙
1. **보안**: SQL Injection을 방지하기 위해 파라미터화된 쿼리를 사용하지 않고, 직접 값을 넣되 문자열은 작은따옴표로 감싸세요.
2. **날짜 형식**: 날짜는 반드시 'YYYY-MM-DD' 형식을 사용하세요.
3. **테이블명**: 대소문자를 구분하므로 정확한 테이블명을 사용하세요 (eiExchangeRate).
4. **컬럼명**: **반드시 한글 컬럼명을 정확히 사용하세요.** 영어 컬럼명은 사용하지 마세요.
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

        user_prompt = f"사용자 요청: {user_request}\n\n"
        
        if context:
            user_prompt += f"추가 컨텍스트:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        
        user_prompt += "위 요청에 맞는 SQL 쿼리를 생성해주세요."
        
        response = await call_gpt(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # SQL 쿼리 추출
        try:
            parsed_response = json.loads(response)
            sql_query = parsed_response.get("sql_query", "")
            explanation = parsed_response.get("explanation", "")
        except json.JSONDecodeError:
            sql_query = _extract_sql_from_response(response)
            explanation = "LLM이 생성한 쿼리"
        
        if not sql_query:
            raise ToolError("rdb_query_llm", "생성된 SQL 쿼리가 비어있습니다.")
        
        logger.info(
            f"✅ SQL 쿼리 생성 완료",
            {"sql_query": sql_query, "explanation": explanation}
        )
        
        # Tool 호출 및 쿼리 로깅
        logger.info(
            f"🔧 [TOOL CALL] rdb_query_llm 실행",
            {
                "tool_name": "rdb_query_llm",
                "user_request": user_request,
                "has_context": context is not None,
                "generated_query": sql_query,
                "explanation": explanation
            }
        )
        
        logger.info(
            f"📝 [QUERY] LLM 생성 쿼리 실행",
            {
                "query": sql_query,
                "user_request": user_request
            }
        )
        
        # 쿼리 실행
        conn = _db_connection._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            result_list = [dict(row) for row in results] if results else []
        
        logger.info(
            f"✅ [TOOL RESULT] rdb_query_llm 완료",
            {
                "tool_name": "rdb_query_llm",
                "query": sql_query,
                "results_count": len(result_list),
                "result_preview": result_list[:3] if result_list else []
            }
        )
        
        return {
            "sql_query": sql_query,
            "explanation": explanation,
            "results": result_list,
            "results_count": len(result_list),
            "status": "success"
        }
    except ToolError:
        # ToolError는 그대로 전파
        raise
    except Exception as e:
        # 기타 예외는 ToolError로 변환 (데코레이터가 처리)
        raise


def _extract_sql_from_response(response: str) -> str:
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


@tool(args_schema=RDBModifyInput)
@handle_tool_error("rdb_modify")
async def rdb_modify(
    query: str,
    params: Optional[Tuple[Any, ...]] = None
) -> Dict[str, Any]:
    """
    RDB 데이터 수정 (INSERT, UPDATE, DELETE)
    
    Args:
        query: 실행할 SQL 쿼리 (INSERT, UPDATE, DELETE)
        params: 쿼리 파라미터 (튜플)
        
    Returns:
        실행 결과 딕셔너리
    """
    # Tool 호출 및 쿼리 로깅
    logger.info(
        f"🔧 [TOOL CALL] rdb_modify 실행",
        {
            "tool_name": "rdb_modify",
            "query_preview": query[:100],
            "has_params": params is not None,
            "params": str(params) if params else None
        }
    )
    
    logger.info(
        f"📝 [QUERY] RDB 수정 쿼리 실행",
        {
            "query": query,
            "params": str(params) if params else None
        }
    )
    
    # 쿼리 타입 확인 (SELECT는 허용하지 않음)
    query_upper = query.strip().upper()
    if query_upper.startswith("SELECT"):
        raise ToolError(
            "rdb_modify",
            "SELECT 쿼리는 rdb_modify에서 사용할 수 없습니다. rdb_query_hard 또는 rdb_query_llm을 사용하세요."
        )
    
    try:
        conn = _db_connection._get_connection()
        with conn.cursor() as cursor:
            if params:
                affected_rows = cursor.execute(query, params)
            else:
                affected_rows = cursor.execute(query)
            
            # 트랜잭션 커밋
            conn.commit()
            
            logger.info(
                f"✅ [TOOL RESULT] rdb_modify 완료",
                {
                    "tool_name": "rdb_modify",
                    "query": query,
                    "affected_rows": affected_rows,
                    "query_type": query_upper.split()[0]
                }
            )
            
            return {
                "success": True,
                "affected_rows": affected_rows,
                "query": query,
                "status": "success"
            }
    except ToolError:
        # ToolError는 그대로 전파
        try:
            conn.rollback()
        except:
            pass
        raise
    except pymysql.Error as e:
        # 데이터베이스 오류
        logger.error(
            f"❌ 데이터베이스 오류",
            {"query": query[:100], "error": str(e)},
            exc_info=True
        )
        try:
            conn.rollback()
        except:
            pass
        raise ToolError("rdb_modify", f"데이터베이스 오류: {str(e)}", e)

