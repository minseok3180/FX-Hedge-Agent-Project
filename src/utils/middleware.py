"""FastAPI 미들웨어 - 요청/응답 추적"""
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from src.utils.logger import get_logger

logger = get_logger("fastapi-middleware")


class TracingMiddleware(BaseHTTPMiddleware):
    """요청/응답 추적 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        # 요청 ID 생성
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 요청 정보 로깅
        logger.info(
            f"📥 요청 수신: {request.method} {request.url.path}",
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_host": request.client.host if request.client else None,
            }
        )
        
        # 요청 본문 읽기 (POST 요청인 경우)
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                body = body_bytes.decode("utf-8") if body_bytes else None
                # 요청 본문을 다시 스트림에 넣어야 함
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                request._receive = receive
            except Exception as e:
                logger.warning(f"요청 본문 읽기 실패: {e}")
        
        if body:
            try:
                import json
                body_json = json.loads(body)
                logger.debug(f"요청 본문: {body_json}", {"request_id": request_id})
            except:
                logger.debug(f"요청 본문 (텍스트): {body[:200]}...", {"request_id": request_id})
        
        try:
            # 다음 미들웨어/엔드포인트 실행
            response = await call_next(request)
            
            # 응답 정보 로깅
            elapsed = time.time() - start_time
            logger.info(
                f"📤 응답 전송: {request.method} {request.url.path}",
                {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "elapsed_time": elapsed,
                }
            )
            
            # 응답 헤더에 요청 ID 추가
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"❌ 요청 처리 실패: {request.method} {request.url.path}",
                {
                    "request_id": request_id,
                    "elapsed_time": elapsed,
                    "error": str(e)
                },
                exc_info=True
            )
            raise

