# src/tools/qdrant_client.py

from typing import List, Dict, Optional
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from src.config.settings import settings


class QdrantTool:
    """
    Qdrant VectorDB Wrapper
    - create_collection() : 벡터 저장용 컬렉션 생성
    - upsert_points()     : vectors + payloads 저장
    - search()            : 의미 기반 검색
    """

    def __init__(
        self,
        collection: str = None,
        vector_size: int = 3072,  # 반드시 text-embedding-3-large 차원
        distance: str = "Cosine"
    ):
        self.host = settings.qdrant_host
        self.port = settings.qdrant_port
        self.api_key = settings.qdrant_api_key

        self.collection = collection or os.getenv("QDRANT_COLLECTION", "hedge_fund_docs")
        self.vector_size = vector_size
        self.distance = getattr(qm.Distance, distance.upper())

        # Client 초기화
        if self.api_key:
            self.client = QdrantClient(
                url=f"http://{self.host}:{self.port}",
                api_key=self.api_key
            )
        else:
            self.client = QdrantClient(
                host=self.host,
                port=self.port
            )

    def create_collection(self):
        """컬렉션이 없으면 새로 생성"""
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection in names:
            return  # 이미 있음

        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.vector_size,
                distance=self.distance
            )
        )

    def upsert_points(self, vectors: List[List[float]], payloads: List[Dict]):
        """벡터 + 페이로드 업서트"""
        self.create_collection()

        points = []
        for idx, (vec, payload) in enumerate(zip(vectors, payloads)):
            points.append(
                qm.PointStruct(
                    id=idx, 
                    vector=vec,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search(self, query_vector: List[float], top_k: int = 5):
        """유사 문서 검색"""
        self.create_collection()

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k
        )

        out = []
        for r in results:
            out.append({
                "score": float(r.score),
                "payload": r.payload
            })
        return out
