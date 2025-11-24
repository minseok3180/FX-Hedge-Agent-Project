"""FastAPI 메인 애플리케이션"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.config.settings import settings
from src.supervisor.supervisor import Supervisor
from src.utils.logger import get_logger
from src.utils.middleware import TracingMiddleware

# Logger 초기화
logger = get_logger("fastapi-app")

# LangSmith tracing 설정 (애플리케이션 시작 시 초기화)
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"
    logger.info("✅ LangSmith 추적 활성화", {"project": settings.langsmith_project})
else:
    logger.warning("⚠️  LangSmith 추적 비활성화 (API 키가 없거나 비활성화됨)")

# FastAPI 앱 초기화
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version
)

# 추적 미들웨어 추가 (CORS보다 먼저)
app.add_middleware(TracingMiddleware)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supervisor 초기화
logger.info("🚀 Supervisor 초기화 중...")
supervisor = Supervisor()
logger.info("✅ Supervisor 초기화 완료")


# 요청/응답 모델
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    answer: str
    agent: str
    metadata: Optional[Dict[str, Any]] = None


# API 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "FX Hedge Agent API",
        "version": settings.api_version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"}


@app.get("/agents")
async def get_agents():
    """사용 가능한 에이전트 목록 조회"""
    return supervisor.get_available_agents()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    채팅 메시지 처리
    
    Supervisor가 적절한 에이전트를 선택하여 작업을 수행합니다.
    """
    logger.info(
        f"💬 채팅 요청 수신",
        {
            "message_length": len(request.message),
            "has_context": request.context is not None
        }
    )
    
    try:
        # Supervisor를 통해 작업 라우팅 및 실행
        result = await supervisor.route_task(request.message)
        
        if result.get("status") == "error":
            error_msg = result.get("error", "알 수 없는 에러")
            logger.error(f"❌ Supervisor 에러: {error_msg}", {"error": error_msg})
            raise HTTPException(status_code=500, detail=error_msg if error_msg else "알 수 없는 에러")
        
        logger.info(
            f"✅ 채팅 응답 생성 완료",
            {
                "agent": result.get("agent", ""),
                "answer_length": len(result.get("answer", "")),
                "status": result.get("status")
            }
        )
        
        return ChatResponse(
            answer=result.get("answer", ""),
            agent=result.get("agent", ""),
            metadata={
                "supervisor_decision": result.get("supervisor_decision"),
                "task": result.get("task"),
                "status": result.get("status")
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API 에러: {str(e)}", {"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/{agent_name}")
async def direct_agent_call(agent_name: str, request: ChatRequest):
    """
    특정 에이전트에 직접 요청
    
    Args:
        agent_name: 에이전트 이름 (web_search_agent, rag_agent)
        request: 채팅 요청
    """
    logger.info(
        f"🎯 직접 에이전트 호출: {agent_name}",
        {
            "agent_name": agent_name,
            "message_length": len(request.message)
        }
    )
    
    try:
        if agent_name not in supervisor.agents:
            logger.warning(f"⚠️  에이전트를 찾을 수 없음: {agent_name}")
            raise HTTPException(status_code=404, detail=f"에이전트를 찾을 수 없습니다: {agent_name}")
        
        agent = supervisor.agents[agent_name]
        result = await agent.execute(request.message, request.context)
        
        if result.get("status") == "error":
            logger.error(
                f"❌ 에이전트 실행 실패: {agent_name}",
                {"error": result.get("error", "알 수 없는 에러")}
            )
            raise HTTPException(status_code=500, detail=result.get("error", "알 수 없는 에러"))
        
        logger.info(
            f"✅ 에이전트 실행 완료: {agent_name}",
            {
                "agent": result.get("agent", ""),
                "status": result.get("status")
            }
        )
        
        return ChatResponse(
            answer=result.get("answer", ""),
            agent=result.get("agent", ""),
            metadata=result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 직접 에이전트 호출 에러: {str(e)}", {"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

