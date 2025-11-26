"""쿼리 모듈 - RDB 접근 툴에서 사용하는 SQL 쿼리문 저장"""
from .rdb_hard_queries import rdb_hard_queries
from .rdb_modify_queries import rdb_modify_queries

__all__ = ["rdb_hard_queries", "rdb_modify_queries"]

