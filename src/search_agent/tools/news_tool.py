#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
from typing import Optional, Dict, Any, List

from search_agent.config import CHROMA_DIR, RAG_DIR

class NewsTool:
    """
    뉴스 수집/검색 도구
    - 매일 아침 수집된 뉴스가 이미 RAG에 적재되어 있다고 가정
    - 검색 시 Chroma DB를 조회하여 관련 기사를 반환
    - 내부적으로 RAG/rag_query.py를 서브프로세스로 호출 (의존 최소화)
    """

    def search(
        self,
        query: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        top_k: int = 10,
        collection: str = "news_unstructured",
        where: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        반환: rag_query.py의 표준 출력(문자열)
        """
        cmd = [
            "python",
            str(RAG_DIR / "rag_query.py"),
            "--db", str(CHROMA_DIR),
            "--collection", collection,
            "--query", query,
            "--n", str(top_k),
        ]
        # 날짜 필터를 rag_query.py에서 지원하면 where 전달
        _where = where or {}
        if start:
            _where["start"] = start
        if end:
            _where["end"] = end
        if _where:
            cmd += ["--where", json.dumps(_where, ensure_ascii=False)]
        out = subprocess.check_output(cmd, text=True)
        return out
