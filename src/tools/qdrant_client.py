"""Qdrant 벡터 데이터베이스 클라이언트"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
from src.config.settings import settings


class QdrantTool:
    """Qdrant 벡터 데이터베이스 도구"""
    
    def __init__(self, collection_name: str = "fx_hedge_data"):
        self.host = settings.qdrant_host
        self.port = settings.qdrant_port
        self.api_key = settings.qdrant_api_key
        self.collection_name = collection_name
        
        # Qdrant 클라이언트 초기화
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
    
    async def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        벡터 유사도 검색
        
        Args:
            query_vector: 검색할 벡터
            limit: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 컬렉션 존재 여부 확인
            try:
                collections = self.client.get_collections().collections
                collection_names = [col.name for col in collections]
                
                if self.collection_name not in collection_names:
                    # 컬렉션이 없으면 빈 리스트 반환 (에러 없이)
                    return []
            except Exception:
                # 컬렉션 확인 실패 시에도 계속 진행 (검색 시 에러 처리)
                pass
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception:
            # 모든 에러를 무시하고 빈 리스트 반환
            return []
    
    async def create_collection(self, vector_size: int = 384):
        """컬렉션 생성"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
        except Exception as e:
            print(f"컬렉션 생성 에러 (이미 존재할 수 있음): {e}")
    
    async def upsert_points(self, points: List[PointStruct]):
        """포인트 업서트"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        except Exception as e:
            print(f"포인트 업서트 에러: {e}")

