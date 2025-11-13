"""FastAPI 메인 애플리케이션"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.config.settings import settings
from src.supervisor.supervisor import Supervisor

# LangSmith tracing 설정 (애플리케이션 시작 시 초기화)
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"

# FastAPI 앱 초기화
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supervisor 초기화
supervisor = Supervisor()


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
    try:
        # Supervisor를 통해 작업 라우팅 및 실행
        result = await supervisor.route_task(request.message)
        
        if result.get("status") == "error":
            error_msg = result.get("error", "알 수 없는 에러")
            print(f"Supervisor 에러: {error_msg}")  # 디버깅용
            raise HTTPException(status_code=500, detail=error_msg if error_msg else "알 수 없는 에러")
        
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
        import traceback
        error_trace = traceback.format_exc()
        print(f"API 에러: {str(e)}\n{error_trace}")  # 디버깅용
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/{agent_name}")
async def direct_agent_call(agent_name: str, request: ChatRequest):
    """
    특정 에이전트에 직접 요청
    
    Args:
        agent_name: 에이전트 이름 (web_search_agent, rag_agent, docs_agent)
        request: 채팅 요청
    """
    try:
        if agent_name not in supervisor.agents:
            raise HTTPException(status_code=404, detail=f"에이전트를 찾을 수 없습니다: {agent_name}")
        
        agent = supervisor.agents[agent_name]
        result = await agent.execute(request.message, request.context)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "알 수 없는 에러"))
        
        return ChatResponse(
            answer=result.get("answer", ""),
            agent=result.get("agent", ""),
            metadata=result
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

