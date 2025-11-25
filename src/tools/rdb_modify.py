"""RDB 데이터 수정 툴 - 하드코딩된 쿼리를 사용하여 RDB 데이터 수정"""
from typing import Dict, Any, Optional, List
from src.tools.database import DatabaseTool
from src.query.rdb_modify_queries import rdb_modify_queries
from src.utils.logger import get_logger


class RDBModifyTool:
    """하드코딩된 쿼리를 사용하여 RDB 데이터를 수정하는 툴 (INSERT, UPDATE, DELETE)"""
    
    def __init__(self):
        self.db_tool = DatabaseTool()
        self.logger = get_logger("rdb_modify_tool")
        self.available_queries = rdb_modify_queries
    
    def list_queries(self) -> List[str]:
        """
        사용 가능한 수정 쿼리 목록 반환
        
        Returns:
            쿼리 키 리스트
        """
        return list(self.available_queries.keys())
    
    async def execute_by_key(self, query_key: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        쿼리 키를 사용하여 수정 쿼리 실행
        
        Args:
            query_key: 실행할 쿼리의 키
            params: 쿼리 파라미터 (튜플)
            
        Returns:
            실행 결과 딕셔너리
        """
        if query_key not in self.available_queries:
            available = ", ".join(self.available_queries.keys())
            raise KeyError(
                f"쿼리 키 '{query_key}'를 찾을 수 없습니다. "
                f"사용 가능한 쿼리: {available}"
            )
        
        query = self.available_queries[query_key]
        return await self.execute(query, params)
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        수정 쿼리 실행 (INSERT, UPDATE, DELETE)
        
        Args:
            query: 실행할 SQL 쿼리 (INSERT, UPDATE, DELETE)
            params: 쿼리 파라미터 (튜플)
            
        Returns:
            실행 결과 딕셔너리
        """
        self.logger.info(
            f"🔧 데이터 수정 쿼리 실행 시작",
            {"query_preview": query[:100], "has_params": params is not None}
        )
        
        # 쿼리 타입 확인 (SELECT는 허용하지 않음)
        query_upper = query.strip().upper()
        if query_upper.startswith("SELECT"):
            error_msg = "SELECT 쿼리는 rdb_modify에서 사용할 수 없습니다. rdb_query를 사용하세요."
            self.logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "affected_rows": 0,
                "status": "error"
            }
        
        try:
            conn = self.db_tool._get_connection()
            with conn.cursor() as cursor:
                if params:
                    affected_rows = cursor.execute(query, params)
                else:
                    affected_rows = cursor.execute(query)
                
                # 트랜잭션 커밋
                conn.commit()
                
                self.logger.info(
                    f"✅ 데이터 수정 완료",
                    {"affected_rows": affected_rows, "query_type": query_upper.split()[0]}
                )
                
                return {
                    "success": True,
                    "affected_rows": affected_rows,
                    "query": query,
                    "status": "success"
                }
        except Exception as e:
            self.logger.error(
                f"❌ 데이터 수정 실패",
                {"query": query[:100], "error": str(e)},
                exc_info=True
            )
            # 롤백
            try:
                conn.rollback()
            except:
                pass
            
            return {
                "success": False,
                "error": str(e),
                "affected_rows": 0,
                "status": "error"
            }
    
    async def insert(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        INSERT 쿼리 실행
        
        Args:
            query: INSERT 쿼리
            params: 쿼리 파라미터 (튜플)
            
        Returns:
            실행 결과
        """
        return await self.execute(query, params)
    
    async def update(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        UPDATE 쿼리 실행
        
        Args:
            query: UPDATE 쿼리
            params: 쿼리 파라미터 (튜플)
            
        Returns:
            실행 결과
        """
        return await self.execute(query, params)
    
    async def delete(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        DELETE 쿼리 실행
        
        Args:
            query: DELETE 쿼리
            params: 쿼리 파라미터 (튜플)
            
        Returns:
            실행 결과
        """
        return await self.execute(query, params)
    
    async def execute_batch(self, queries: List[str], params_list: Optional[List[tuple]] = None) -> Dict[str, Any]:
        """
        여러 쿼리를 배치로 실행 (트랜잭션)
        
        Args:
            queries: 실행할 쿼리 리스트
            params_list: 각 쿼리에 대한 파라미터 리스트 (None이면 파라미터 없음)
            
        Returns:
            실행 결과
        """
        self.logger.info(
            f"🔧 배치 쿼리 실행 시작",
            {"queries_count": len(queries)}
        )
        
        try:
            conn = self.db_tool._get_connection()
            total_affected_rows = 0
            
            with conn.cursor() as cursor:
                for i, query in enumerate(queries):
                    # SELECT 쿼리 체크
                    query_upper = query.strip().upper()
                    if query_upper.startswith("SELECT"):
                        error_msg = f"SELECT 쿼리는 rdb_modify에서 사용할 수 없습니다. (쿼리 {i+1})"
                        self.logger.error(f"❌ {error_msg}")
                        conn.rollback()
                        return {
                            "success": False,
                            "error": error_msg,
                            "affected_rows": 0,
                            "status": "error"
                        }
                    
                    params = params_list[i] if params_list and i < len(params_list) else None
                    
                    if params:
                        affected_rows = cursor.execute(query, params)
                    else:
                        affected_rows = cursor.execute(query)
                    
                    total_affected_rows += affected_rows
                    self.logger.debug(
                        f"✅ 쿼리 {i+1}/{len(queries)} 실행 완료",
                        {"affected_rows": affected_rows}
                    )
                
                # 모든 쿼리 성공 시 커밋
                conn.commit()
                
                self.logger.info(
                    f"✅ 배치 쿼리 실행 완료",
                    {"total_affected_rows": total_affected_rows, "queries_count": len(queries)}
                )
                
                return {
                    "success": True,
                    "affected_rows": total_affected_rows,
                    "queries_count": len(queries),
                    "status": "success"
                }
        except Exception as e:
            self.logger.error(
                f"❌ 배치 쿼리 실행 실패",
                {"queries_count": len(queries), "error": str(e)},
                exc_info=True
            )
            # 롤백
            try:
                conn.rollback()
            except:
                pass
            
            return {
                "success": False,
                "error": str(e),
                "affected_rows": 0,
                "status": "error"
            }

