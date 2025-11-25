"""Qdrant 벡터 데이터베이스 클라이언트"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
from src.utils.settings import settings
from src.utils.logger import get_logger


class QdrantTool:
    """Qdrant 벡터 데이터베이스 도구"""
    
    def __init__(self, collection_name: str = "fx_hedge_data"):
        self.logger = get_logger("qdrant-tool")
        self.collection_name = collection_name
        
        # GCP Qdrant URL 우선 사용, 없으면 host/port 방식 사용
        qdrant_url = settings.qdrant_url
        qdrant_host = settings.qdrant_host
        qdrant_port = settings.qdrant_port
        self.api_key = settings.qdrant_api_key
        
        # Qdrant 클라이언트 초기화
        try:
            if qdrant_url:
                # GCP Qdrant: 전체 URL 사용 (HTTPS)
                if not self.api_key:
                    self.logger.warning("⚠️  GCP Qdrant URL이 설정되었지만 API 키가 없습니다.")
                
                self.logger.info(
                    f"🔗 GCP Qdrant 연결 중...",
                    {"url": qdrant_url, "collection": collection_name}
                )
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=self.api_key
                )
                self.logger.info("✅ GCP Qdrant 연결 성공")
            elif qdrant_host:
                # 로컬 Qdrant: host/port 방식
                if self.api_key:
                    # API 키가 있는 경우 (로컬에서도 인증 사용 가능)
                    self.logger.info(
                        f"🔗 로컬 Qdrant 연결 중 (인증)...",
                        {"host": qdrant_host, "port": qdrant_port, "collection": collection_name}
                    )
                    self.client = QdrantClient(
                        url=f"http://{qdrant_host}:{qdrant_port}",
                        api_key=self.api_key
                    )
                else:
                    # API 키 없는 경우
                    self.logger.info(
                        f"🔗 로컬 Qdrant 연결 중...",
                        {"host": qdrant_host, "port": qdrant_port, "collection": collection_name}
                    )
                    self.client = QdrantClient(
                        host=qdrant_host,
                        port=qdrant_port
                    )
                self.logger.info("✅ 로컬 Qdrant 연결 성공")
            else:
                raise ValueError("Qdrant URL 또는 Host가 설정되지 않았습니다. QDRANT_URL 또는 QDRANT_HOST 환경 변수를 설정하세요.")
        except Exception as e:
            self.logger.error(
                f"❌ Qdrant 연결 실패",
                {
                    "url": qdrant_url,
                    "host": qdrant_host,
                    "port": qdrant_port,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
    
    async def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        벡터 유사도 검색
        
        Args:
            query_vector: 검색할 벡터
            limit: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        self.logger.debug(
            f"🔍 벡터 검색 시작",
            {"collection": self.collection_name, "vector_size": len(query_vector), "limit": limit}
        )
        
        try:
            # 컬렉션 존재 여부 확인
            try:
                collections = self.client.get_collections().collections
                collection_names = [col.name for col in collections]
                
                if self.collection_name not in collection_names:
                    self.logger.warning(
                        f"⚠️  컬렉션이 존재하지 않음",
                        {"collection": self.collection_name, "available_collections": collection_names}
                    )
                    return []
            except Exception as e:
                self.logger.warning(
                    f"⚠️  컬렉션 확인 실패 (검색 계속 진행)",
                    {"error": str(e)}
                )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            self.logger.info(
                f"✅ 벡터 검색 완료",
                {"collection": self.collection_name, "results_count": len(results)}
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception as e:
            self.logger.error(
                f"❌ 벡터 검색 실패",
                {"collection": self.collection_name, "error": str(e)},
                exc_info=True
            )
            # 에러 발생 시 빈 리스트 반환 (기존 동작 유지)
            return []
    
    async def create_collection(self, vector_size: int = 384):
        """컬렉션 생성"""
        self.logger.info(
            f"📦 컬렉션 생성 시작",
            {"collection": self.collection_name, "vector_size": vector_size}
        )
        
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            self.logger.info(f"✅ 컬렉션 생성 완료: {self.collection_name}")
        except Exception as e:
            # 컬렉션이 이미 존재하는 경우는 정상 동작
            if "already exists" in str(e).lower() or "already_exists" in str(e).lower():
                self.logger.info(f"ℹ️  컬렉션이 이미 존재함: {self.collection_name}")
            else:
                self.logger.error(
                    f"❌ 컬렉션 생성 실패",
                    {"collection": self.collection_name, "error": str(e)},
                    exc_info=True
                )
                raise
    
    async def upsert_points(self, points: List[PointStruct]):
        """포인트 업서트"""
        self.logger.info(
            f"📝 포인트 업서트 시작",
            {"collection": self.collection_name, "points_count": len(points)}
        )
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            self.logger.info(
                f"✅ 포인트 업서트 완료",
                {"collection": self.collection_name, "points_count": len(points)}
            )
        except Exception as e:
            self.logger.error(
                f"❌ 포인트 업서트 실패",
                {"collection": self.collection_name, "error": str(e)},
                exc_info=True
            )
            raise

