"""RDB 하드 쿼리 툴 - Query 폴더에 정의된 쿼리를 사용하여 결과 반환"""
import re
from typing import List, Dict, Any, Optional
from src.tools.database import DatabaseTool
from src.query.rdb_hard_queries import rdb_hard_queries
from src.utils.logger import get_logger


class RDBHardTool:
    """Query 폴더에 정의된 쿼리를 사용하여 RDB에서 데이터를 가져오는 범용 툴"""
    
    def __init__(self):
        self.db_tool = DatabaseTool()
        self.available_queries = rdb_hard_queries
        self.logger = get_logger("rdb_hard_tool")
    
    def list_queries(self) -> List[str]:
        """
        사용 가능한 쿼리 목록 반환
        
        Returns:
            쿼리 키 리스트
        """
        return list(self.available_queries.keys())
    
    def _resolve_placeholders(self, query: str, state: Optional[Dict[str, Any]] = None) -> str:
        """
        쿼리의 placeholder를 state에서 가져온 값으로 치환
        
        Args:
            query: SQL 쿼리 문자열
            state: AgentState의 딕셔너리 (user_id, date 등 포함)
            
        Returns:
            placeholder가 치환된 쿼리 문자열
        """
        if not state:
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
        for key, value in placeholders.items():
            if value:
                # {key} 형식의 placeholder 치환
                pattern = f"\\{{{key}\\}}"
                resolved_query = re.sub(pattern, str(value), resolved_query)
                self.logger.debug(
                    f"Placeholder 치환: {{{key}}} → {value}",
                    {"placeholder": key, "value": value}
                )
        
        return resolved_query
    
    async def execute(
        self, 
        query_key: str, 
        params: Optional[tuple] = None,
        state: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        지정된 쿼리를 실행하여 결과 반환
        
        Args:
            query_key: 실행할 쿼리의 키 (예: "get_by_date")
            params: 쿼리 파라미터 (튜플) - placeholder 사용 시 None
            state: AgentState 딕셔너리 (placeholder 치환용)
            
        Returns:
            쿼리 결과 리스트
            
        Raises:
            KeyError: 지정된 쿼리 키가 존재하지 않을 때
        """
        if query_key not in self.available_queries:
            available = ", ".join(self.available_queries.keys())
            raise KeyError(
                f"쿼리 키 '{query_key}'를 찾을 수 없습니다. "
                f"사용 가능한 쿼리: {available}"
            )
        
        query = self.available_queries[query_key]
        
        # Placeholder가 있는지 확인
        has_placeholder = re.search(r"\{(\w+)\}", query)
        
        if has_placeholder:
            # Placeholder 치환
            query = self._resolve_placeholders(query, state)
            # Placeholder 사용 시 params는 None으로 설정
            params = None
        
        return await self.db_tool.execute_query(query, params)
    
    async def get_by_date(
        self, 
        date: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        특정 일자의 경제 지표 조회 (편의 메서드)
        
        Args:
            date: 날짜 (YYYY-MM-DD 형식) - None이면 state에서 가져옴
            state: AgentState 딕셔너리 (date가 None일 때 사용)
            
        Returns:
            경제 지표 정보 리스트
        """
        if date:
            return await self.execute("get_by_date", (date,), state)
        else:
            return await self.execute("get_by_date", None, state)
    
    async def get_by_range(
        self, 
        start_date: str, 
        end_date: str, 
        limit: int = 100,
        state: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        날짜 범위의 경제 지표 조회 (편의 메서드)
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD 형식)
            end_date: 종료 날짜 (YYYY-MM-DD 형식)
            limit: 최대 조회 개수
            state: AgentState 딕셔너리
            
        Returns:
            경제 지표 정보 리스트
        """
        return await self.execute("get_by_range", (start_date, end_date, limit), state)
    
    async def get_latest(
        self, 
        limit: int = 10,
        state: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        최신 경제 지표 조회 (편의 메서드)
        
        Args:
            limit: 조회할 최신 데이터 개수
            state: AgentState 딕셔너리
            
        Returns:
            경제 지표 정보 리스트
        """
        return await self.execute("get_latest", (limit,), state)

