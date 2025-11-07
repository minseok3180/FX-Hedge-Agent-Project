"""데이터베이스 쿼리 도구"""
import pymysql
from typing import List, Dict, Any, Optional
from src.config.settings import settings


class DatabaseTool:
    """MariaDB 데이터베이스 쿼리 도구"""
    
    def __init__(self):
        self.host = settings.db_host
        self.port = settings.db_port
        self.user = settings.db_user
        self.password = settings.db_password
        self.database = settings.db_name
        self.connection = None
    
    def _get_connection(self):
        """데이터베이스 연결 생성"""
        if self.connection is None or not self.connection.open:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        return self.connection
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        SQL 쿼리 실행
        
        Args:
            query: SQL 쿼리 문자열
            params: 쿼리 파라미터 (튜플)
            
        Returns:
            쿼리 결과 리스트
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                results = cursor.fetchall()
                return list(results)
        except Exception as e:
            print(f"데이터베이스 쿼리 에러: {e}")
            return []
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        테이블 스키마 정보 조회
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            스키마 정보 딕셔너리
        """
        query = f"DESCRIBE {table_name}"
        return await self.execute_query(query)
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.connection and self.connection.open:
            self.connection.close()
            self.connection = None

