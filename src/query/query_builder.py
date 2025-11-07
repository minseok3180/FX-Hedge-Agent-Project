"""SQL 쿼리 빌더"""
from typing import Dict, Any, Optional, List


class QueryBuilder:
    """SQL 쿼리 생성 헬퍼 클래스"""
    
    @staticmethod
    def build_select_query(
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        SELECT 쿼리 생성
        
        Args:
            table: 테이블 이름
            columns: 선택할 컬럼 리스트 (None이면 *)
            where: WHERE 조건 딕셔너리
            order_by: 정렬 컬럼
            limit: 결과 제한 수
            
        Returns:
            SQL 쿼리 문자열
        """
        # SELECT 절
        if columns:
            columns_str = ", ".join(columns)
        else:
            columns_str = "*"
        
        query = f"SELECT {columns_str} FROM {table}"
        
        # WHERE 절
        if where:
            conditions = []
            for key, value in where.items():
                if isinstance(value, str):
                    conditions.append(f"{key} = '{value}'")
                else:
                    conditions.append(f"{key} = {value}")
            query += " WHERE " + " AND ".join(conditions)
        
        # ORDER BY 절
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # LIMIT 절
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    
    @staticmethod
    def build_aggregate_query(
        table: str,
        aggregate_func: str,
        column: str,
        group_by: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        집계 쿼리 생성
        
        Args:
            table: 테이블 이름
            aggregate_func: 집계 함수 (COUNT, SUM, AVG, MAX, MIN 등)
            column: 집계할 컬럼
            group_by: 그룹화할 컬럼
            where: WHERE 조건
            
        Returns:
            SQL 쿼리 문자열
        """
        query = f"SELECT {aggregate_func}({column}) as result"
        
        if group_by:
            query += f", {group_by}"
        
        query += f" FROM {table}"
        
        if where:
            conditions = []
            for key, value in where.items():
                if isinstance(value, str):
                    conditions.append(f"{key} = '{value}'")
                else:
                    conditions.append(f"{key} = {value}")
            query += " WHERE " + " AND ".join(conditions)
        
        if group_by:
            query += f" GROUP BY {group_by}"
        
        return query

