"""데이터베이스 쿼리 도구"""
import pymysql
import json
from datetime import date, datetime
from typing import List, Dict, Any, Optional
from src.config.settings import settings


SELECT_COLUMNS = """
        date,
        usdkrw,
        `미국수출금액`,
        `미국수입금액`,
        `외환보유액`,
        `미국외환보유액`,
        `한국은행기준금리`,
        `시장금리`,
        `소비자물가지수`,
        `수출물가지수`,
        `수입물가지수`,
        NULL AS us_current,
        `미국경제성장률`,
        `us_gdp`,
        `미국주가지수`,
        `미국금리`,
        `정부대출금금리`,
        `경제성장률`,
        `gdp`,
        `주가지수`,
        `한국금리`
"""


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
    
    def _serialize_value(self, value: Any) -> Any:
        """날짜/시간 객체를 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return value
    
    def _serialize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """행의 모든 값을 JSON 직렬화 가능한 형태로 변환"""
        return {k: self._serialize_value(v) for k, v in row.items()}
    
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
                # 날짜/시간 객체를 문자열로 변환
                return [self._serialize_row(row) for row in results]
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
    
    async def get_exchange_rate_by_date(self, date: str) -> List[Dict[str, Any]]:
        """
        특정 일자의 환율 정보 조회
        
        Args:
            date: 날짜 (YYYY-MM-DD 형식)
            
        Returns:
            환율 정보 리스트
        """
        query = f"""
        SELECT 
            {SELECT_COLUMNS}
        FROM eiExchangeRate
        WHERE date = %s
        ORDER BY date DESC
        LIMIT 1
        """
        return await self.execute_query(query, (date,))
    
    async def get_exchange_rate_range(self, start_date: str, end_date: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        날짜 범위의 환율 정보 조회
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD 형식)
            end_date: 종료 날짜 (YYYY-MM-DD 형식)
            limit: 최대 조회 개수
            
        Returns:
            환율 정보 리스트
        """
        query = f"""
        SELECT 
            {SELECT_COLUMNS}
        FROM eiExchangeRate
        WHERE date BETWEEN %s AND %s
        ORDER BY date DESC
        LIMIT %s
        """
        return await self.execute_query(query, (start_date, end_date, limit))
    
    async def get_latest_exchange_rate(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        최신 환율 정보 조회
        
        Args:
            limit: 조회할 최신 데이터 개수
            
        Returns:
            환율 정보 리스트
        """
        query = f"""
        SELECT 
            {SELECT_COLUMNS}
        FROM eiExchangeRate
        ORDER BY date DESC
        LIMIT %s
        """
        return await self.execute_query(query, (limit,))
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.connection and self.connection.open:
            self.connection.close()
            self.connection = None

