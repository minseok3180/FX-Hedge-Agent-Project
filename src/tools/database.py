"""데이터베이스 연결 및 쿼리 실행 도구"""
import pymysql
from typing import List, Dict, Any, Optional
from src.utils.settings import settings
from src.utils.logger import get_logger


class DatabaseTool:
    """MariaDB 데이터베이스 연결 및 쿼리 실행 도구"""
    
    def __init__(self):
        self.logger = get_logger("database-tool")
        self._connection: Optional[pymysql.Connection] = None
    
    def _get_connection(self) -> pymysql.Connection:
        """
        데이터베이스 연결 반환 (재사용 또는 새로 생성)
        
        Returns:
            pymysql.Connection 객체
        """
        if self._connection is None or not self._connection.open:
            try:
                self._connection = pymysql.connect(
                    host=settings.db_host,
                    port=settings.db_port,
                    user=settings.db_user,
                    password=settings.db_password,
                    database=settings.db_name,
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=False
                )
                self.logger.debug("✅ 데이터베이스 연결 성공")
            except Exception as e:
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
        
        return self._connection
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        SELECT 쿼리 실행 (읽기 전용)
        
        Args:
            query: 실행할 SQL 쿼리
            params: 쿼리 파라미터 (튜플)
            
        Returns:
            쿼리 결과 리스트 (딕셔너리 형태)
        """
        self.logger.debug(
            f"🔍 쿼리 실행",
            {"query_preview": query[:100], "has_params": params is not None}
        )
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                results = cursor.fetchall()
                
                # DictCursor를 사용하므로 결과는 이미 딕셔너리 리스트
                result_list = [dict(row) for row in results] if results else []
                
                self.logger.debug(
                    f"✅ 쿼리 실행 완료",
                    {"rows_count": len(result_list)}
                )
                
                return result_list
        except Exception as e:
            self.logger.error(
                f"❌ 쿼리 실행 실패",
                {"query_preview": query[:100], "error": str(e)},
                exc_info=True
            )
            raise
    
    async def execute_modify(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        INSERT, UPDATE, DELETE 쿼리 실행 (수정 작업)
        
        Args:
            query: 실행할 SQL 쿼리
            params: 쿼리 파라미터 (튜플)
            
        Returns:
            실행 결과 딕셔너리 (affected_rows 포함)
        """
        self.logger.debug(
            f"✏️ 수정 쿼리 실행",
            {"query_preview": query[:100], "has_params": params is not None}
        )
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                if params:
                    affected_rows = cursor.execute(query, params)
                else:
                    affected_rows = cursor.execute(query)
                
                conn.commit()
                
                self.logger.info(
                    f"✅ 수정 쿼리 실행 완료",
                    {"affected_rows": affected_rows}
                )
                
                return {
                    "success": True,
                    "affected_rows": affected_rows,
                    "status": "success"
                }
        except Exception as e:
            conn.rollback()
            self.logger.error(
                f"❌ 수정 쿼리 실행 실패",
                {"query_preview": query[:100], "error": str(e)},
                exc_info=True
            )
            return {
                "success": False,
                "error": str(e),
                "affected_rows": 0,
                "status": "error"
            }
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self._connection and self._connection.open:
            self._connection.close()
            self._connection = None
            self.logger.debug("🔌 데이터베이스 연결 종료")

