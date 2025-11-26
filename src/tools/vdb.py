"""Qdrant 벡터 데이터베이스 도구"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional, Union
from src.utils.settings import settings
from src.utils.logger import get_logger
from src.utils.tools import (
    tool,
    handle_tool_error,
    ToolError,
    VDBSearchInput,
    VDBCreateCollectionInput,
    VDBUpsertPointsInput
)

logger = get_logger("vdb-tool")

# 전역 Qdrant 클라이언트 인스턴스
_qdrant_client: Optional[QdrantClient] = None
_default_collection = "fx_hedge_data"


def _get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트 인스턴스 반환 (싱글톤 패턴)"""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_url = settings.qdrant_url
        qdrant_host = settings.qdrant_host
        qdrant_port = settings.qdrant_port
        api_key = settings.qdrant_api_key
        
        try:
            if qdrant_url:
                # GCP Qdrant: 전체 URL 사용 (HTTPS)
                if not api_key:
                    logger.warning("⚠️  GCP Qdrant URL이 설정되었지만 API 키가 없습니다.")
                
                logger.info(
                    f"🔗 GCP Qdrant 연결 중...",
                    {"url": qdrant_url}
                )
                _qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=api_key
                )
                logger.info("✅ GCP Qdrant 연결 성공")
            elif qdrant_host:
                # 로컬 Qdrant: host/port 방식
                if api_key:
                    logger.info(
                        f"🔗 로컬 Qdrant 연결 중 (인증)...",
                        {"host": qdrant_host, "port": qdrant_port}
                    )
                    _qdrant_client = QdrantClient(
                        url=f"http://{qdrant_host}:{qdrant_port}",
                        api_key=api_key
                    )
                else:
                    logger.info(
                        f"🔗 로컬 Qdrant 연결 중...",
                        {"host": qdrant_host, "port": qdrant_port}
                    )
                    _qdrant_client = QdrantClient(
                        host=qdrant_host,
                        port=qdrant_port
                    )
                logger.info("✅ 로컬 Qdrant 연결 성공")
            else:
                raise ValueError("Qdrant URL 또는 Host가 설정되지 않았습니다. QDRANT_URL 또는 QDRANT_HOST 환경 변수를 설정하세요.")
        except Exception as e:
            logger.error(
                f"❌ Qdrant 연결 실패",
                {
                    "url": qdrant_url,
                    "host": qdrant_host,
                    "port": qdrant_port,
                    "error": str(e)
                },
                exc_info=True
            )
            raise ToolError("vdb", f"Qdrant 연결 실패: {str(e)}", e)
    return _qdrant_client


def _check_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """
    컬렉션 존재 여부 확인
    
    Args:
        client: Qdrant 클라이언트
        collection_name: 확인할 컬렉션 이름
        
    Returns:
        컬렉션 존재 여부
    """
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        return collection_name in collection_names
    except Exception as e:
        logger.warning(
            f"⚠️  컬렉션 확인 실패",
            {"collection": collection_name, "error": str(e)}
        )
        return False


@tool(args_schema=VDBSearchInput)
@handle_tool_error("vdb_search")
async def vdb_search(
    query_vector: List[float],
    collection_name: str = _default_collection,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    벡터 유사도 검색
    
    Args:
        query_vector: 검색할 벡터
        collection_name: 컬렉션 이름 (기본값: "fx_hedge_data")
        limit: 반환할 결과 수 (기본값: 5)
        
    Returns:
        검색 결과 리스트 (id, score, payload 포함)
    """
    logger.debug(
        f"🔍 벡터 검색 시작",
        {"collection": collection_name, "vector_size": len(query_vector), "limit": limit}
    )
    
    try:
        client = _get_qdrant_client()
        
        # 컬렉션 존재 여부 확인
        if not _check_collection_exists(client, collection_name):
            available_collections = []
            try:
                collections = client.get_collections().collections
                available_collections = [col.name for col in collections]
            except Exception:
                pass
            
            logger.warning(
                f"⚠️  컬렉션이 존재하지 않음",
                {"collection": collection_name, "available_collections": available_collections}
            )
            return []
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        logger.info(
            f"✅ 벡터 검색 완료",
            {"collection": collection_name, "results_count": len(results)}
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
        logger.error(
            f"❌ 벡터 검색 실패",
            {"collection": collection_name, "error": str(e)},
            exc_info=True
        )
        # 에러 발생 시 빈 리스트 반환 (검색 실패는 치명적이지 않음)
        # ToolError는 handle_tool_error 데코레이터가 처리하므로 여기서는 빈 리스트 반환
        return []


@tool(args_schema=VDBCreateCollectionInput)
@handle_tool_error("vdb_create_collection")
async def vdb_create_collection(
    collection_name: str = _default_collection,
    vector_size: int = 384
) -> Dict[str, Any]:
    """
    Qdrant 컬렉션 생성
    
    Args:
        collection_name: 컬렉션 이름 (기본값: "fx_hedge_data")
        vector_size: 벡터 크기 (기본값: 384)
        
    Returns:
        생성 결과 딕셔너리
    """
    logger.info(
        f"📦 컬렉션 생성 시작",
        {"collection": collection_name, "vector_size": vector_size}
    )
    
    try:
        client = _get_qdrant_client()
        
        # 컬렉션이 이미 존재하는지 확인
        if _check_collection_exists(client, collection_name):
            logger.info(f"ℹ️  컬렉션이 이미 존재함: {collection_name}")
            return {
                "success": True,
                "collection_name": collection_name,
                "vector_size": vector_size,
                "status": "already_exists"
            }
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info(f"✅ 컬렉션 생성 완료: {collection_name}")
        return {
            "success": True,
            "collection_name": collection_name,
            "vector_size": vector_size,
            "status": "created"
        }
    except Exception as e:
        # 컬렉션이 이미 존재하는 경우는 정상 동작 (race condition 대비)
        error_str = str(e).lower()
        if "already exists" in error_str or "already_exists" in error_str:
            logger.info(f"ℹ️  컬렉션이 이미 존재함: {collection_name}")
            return {
                "success": True,
                "collection_name": collection_name,
                "vector_size": vector_size,
                "status": "already_exists"
            }
        else:
            logger.error(
                f"❌ 컬렉션 생성 실패",
                {"collection": collection_name, "error": str(e)},
                exc_info=True
            )
            raise ToolError("vdb_create_collection", f"컬렉션 생성 실패: {str(e)}", e)


@tool(args_schema=VDBUpsertPointsInput)
@handle_tool_error("vdb_upsert_points")
async def vdb_upsert_points(
    points: List[Union[Dict[str, Any], PointStruct]],
    collection_name: str = _default_collection
) -> Dict[str, Any]:
    """
    Qdrant 포인트 업서트
    
    Args:
        points: 업서트할 포인트 리스트 (PointStruct 형식의 딕셔너리)
        collection_name: 컬렉션 이름 (기본값: "fx_hedge_data")
        
    Returns:
        업서트 결과 딕셔너리
    """
    logger.info(
        f"📝 포인트 업서트 시작",
        {"collection": collection_name, "points_count": len(points)}
    )
    
    try:
        client = _get_qdrant_client()
        
        # 컬렉션 존재 여부 확인
        if not _check_collection_exists(client, collection_name):
            logger.warning(
                f"⚠️  컬렉션이 존재하지 않음 (업서트 전에 생성 필요)",
                {"collection": collection_name}
            )
            raise ToolError(
                "vdb_upsert_points",
                f"컬렉션 '{collection_name}'이 존재하지 않습니다. 먼저 컬렉션을 생성하세요."
            )
        
        # 딕셔너리를 PointStruct로 변환
        point_structs = []
        for point in points:
            if isinstance(point, dict):
                point_structs.append(PointStruct(**point))
            elif isinstance(point, PointStruct):
                point_structs.append(point)
            else:
                raise ToolError(
                    "vdb_upsert_points",
                    f"지원하지 않는 포인트 타입: {type(point)}. dict 또는 PointStruct를 사용하세요."
                )
        
        if not point_structs:
            logger.warning("⚠️  업서트할 포인트가 없음")
            return {
                "success": True,
                "collection_name": collection_name,
                "points_count": 0,
                "status": "no_points"
            }
        
        client.upsert(
            collection_name=collection_name,
            points=point_structs
        )
        logger.info(
            f"✅ 포인트 업서트 완료",
            {"collection": collection_name, "points_count": len(point_structs)}
        )
        return {
            "success": True,
            "collection_name": collection_name,
            "points_count": len(point_structs),
            "status": "success"
        }
    except ToolError:
        # ToolError는 그대로 전파
        raise
    except Exception as e:
        logger.error(
            f"❌ 포인트 업서트 실패",
            {"collection": collection_name, "error": str(e)},
            exc_info=True
        )
        raise ToolError("vdb_upsert_points", f"포인트 업서트 실패: {str(e)}", e)

