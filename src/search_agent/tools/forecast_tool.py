# src/search_agent/tool/forecast_tool.py

"""
Forecast Tool
-------------
Chroma DB에 저장된 시계열 모델 예측 결과를 불러오기 위한 툴.

- 대상 컬렉션: model_forecast
- 주요 기능: 날짜/기간을 받아 해당 구간의 예측 결과 반환
"""

import chromadb
from datetime import datetime
from typing import List, Dict, Any


class ForecastTool:
    def __init__(self, chroma_path: str = "./RAG/chroma_store", collection_name: str = "model_forecast"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)

    def query_forecast(self, start_date: str, end_date: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        특정 날짜 구간의 예측 결과를 조회
        - start_date, end_date: "YYYY-MM-DD" 형식
        """
        results = self.collection.query(
            query_texts=[f"{start_date} ~ {end_date} forecast"],
            n_results=top_k,
            where={
                "date": {"$gte": start_date, "$lte": end_date}
            }
        )
        docs = []
        for meta, doc in zip(results["metadatas"], results["documents"]):
            docs.append({
                "date": meta.get("date", ""),
                "type": meta.get("type", "forecast"),
                "forecast": doc
            })
        return docs


if __name__ == "__main__":
    tool = ForecastTool()
    out = tool.query_forecast("2025-09-04", "2025-09-10", top_k=5)
    for r in out:
        print(r)
