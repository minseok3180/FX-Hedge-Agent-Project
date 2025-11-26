# FX Hedge Agent Project

환율 헷지 전략을 위한 멀티 에이전트 시스템 (Multi-Agent System for FX Hedge Strategy)

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [기술 스택](#기술-스택)
- [설치 및 설정](#설치-및-설정)
- [사용 방법](#사용-방법)
- [아키텍처](#아키텍처)
- [API 문서](#api-문서)
- [개발 가이드](#개발-가이드)
- [라이선스](#라이선스)

## 🎯 개요

FX Hedge Agent는 LangChain과 LangGraph를 활용한 멀티 에이전트 시스템으로, 환율 헷지 전략 수립을 위한 정보 수집, 분석, 의사결정을 자동화합니다. Supervisor 에이전트가 여러 전문 에이전트들을 오케스트레이션하여 사용자의 질의에 대한 종합적인 답변을 제공합니다.

### 핵심 특징

- 🤖 **멀티 에이전트 시스템**: Supervisor가 여러 전문 에이전트를 조율
- 🔍 **다양한 정보 소스**: RDB, VDB, 웹 검색을 통한 정보 수집
- 🧠 **LLM 기반 분석**: GPT 모델을 활용한 자연어 처리 및 분석
- 📊 **LangSmith 추적**: 모든 LLM 호출의 자동 추적 및 모니터링
- 🔧 **모듈화된 구조**: 확장 가능하고 유지보수가 용이한 아키텍처

## ✨ 주요 기능

### 1. 멀티 에이전트 오케스트레이션
- **Supervisor**: 전체 워크플로우 관리 및 라우팅
- **Market Information Agent**: 시장 정보 수집 및 분석
- **ReAsk Agent**: 불명확한 질의에 대한 재질문
- **ReAct Agent**: Reasoning과 Acting을 통한 문제 해결
- **HandsOff Agent**: 사용자에게 직접 전달

### 2. 도구 (Tools)
- **RDB Tools**: MariaDB를 통한 환율 데이터 조회 및 수정
  - 하드코딩된 쿼리 실행
  - LLM 기반 동적 쿼리 생성
- **VDB Tools**: Qdrant 벡터 DB를 통한 유사 문서 검색
- **Web Search**: Google Custom Search API를 통한 최신 정보 수집

### 3. 상태 관리
- 세션별 상태 관리
- 대화 히스토리 추적
- 참조 및 액션 기록

## 📁 프로젝트 구조

```
FX-Hedge-Agent-Project/
├── src/                          # 메인 소스 코드
│   ├── agents/                   # 에이전트 구현
│   │   ├── market_information_agent.py
│   │   ├── react_agent.py
│   │   ├── reask_agent.py
│   │   ├── handsoff_agent.py
│   │   └── supervisor/           # Supervisor 에이전트
│   │       └── supervisor.py
│   ├── tools/                    # 도구 구현 (실질적인 코드만)
│   │   ├── rdb.py               # RDB 도구
│   │   ├── vdb.py               # VDB 도구
│   │   └── web_search.py        # 웹 검색 도구
│   ├── utils/                    # 유틸리티 모듈 (추상화/지원 코드)
│   │   ├── llm.py               # LLM 관련 (LangChain, OpenAI, LangSmith)
│   │   ├── tools.py             # Tool 관련 (에러 처리, 스키마, decorator)
│   │   ├── agents.py            # BaseAgent 클래스
│   │   ├── state.py             # State 관리
│   │   ├── logger.py            # 로깅
│   │   ├── settings.py          # 설정 관리
│   │   └── middleware.py        # FastAPI 미들웨어
│   ├── prompts/                  # 프롬프트 정의
│   │   ├── supervisor_routing.py
│   │   ├── market_information_instruction.py
│   │   └── ...
│   └── query/                    # SQL 쿼리 정의
│       ├── rdb_hard_queries.py
│       └── rdb_modify_queries.py
├── api/                          # FastAPI 애플리케이션
│   └── main.py
├── frontend/                     # 프론트엔드 (선택사항)
├── notebook/                     # 실험 및 노트북
├── scripts/                      # 유틸리티 스크립트
├── config.json                   # 설정 파일
├── requirements.txt              # Python 의존성
└── README.md                     # 이 파일
```

## 🛠 기술 스택

### 핵심 프레임워크
- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **LangGraph**: 상태 기반 멀티 에이전트 오케스트레이션
- **FastAPI**: 고성능 웹 API 프레임워크
- **Pydantic**: 데이터 검증 및 설정 관리

### LLM & 추적
- **OpenAI GPT**: 자연어 처리 및 분석
- **LangSmith**: LLM 호출 추적 및 모니터링

### 데이터베이스
- **MariaDB**: 관계형 데이터베이스 (환율 데이터)
- **Qdrant**: 벡터 데이터베이스 (유사 문서 검색)

### 기타
- **Python 3.8+**: 메인 프로그래밍 언어
- **pymysql**: MariaDB 연결
- **qdrant-client**: Qdrant 클라이언트
- **requests**: HTTP 요청

## 🚀 설치 및 설정

### 1. 저장소 클론

```bash
git clone <repository-url>
cd FX-Hedge-Agent-Project
```

### 2. Python 가상 환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일을 생성하거나 환경 변수를 설정합니다:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4"  # 기본값: gpt-4.1

# 데이터베이스
export DB_HOST="localhost"
export DB_PORT="3306"
export DB_USER="your-db-user"
export DB_PASSWORD="your-db-password"
export DB_NAME="fx_hedge"

# Qdrant
export QDRANT_URL="http://localhost:6333"  # 또는
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export QDRANT_API_KEY="your-qdrant-api-key"  # 선택사항

# 웹 검색
export WEB_SEARCH_API_KEY="your-google-api-key"
export WEB_SEARCH_ENGINE_ID="your-search-engine-id"

# LangSmith (선택사항)
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="fx-hedge-agent"
export LANGSMITH_TRACING="true"
```

### 5. 데이터베이스 설정

MariaDB와 Qdrant가 실행 중이어야 합니다. 필요시 Docker Compose를 사용할 수 있습니다:

```bash
docker-compose up -d
```

## 💻 사용 방법

### API 서버 실행

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 `http://localhost:8000`에서 API에 접근할 수 있습니다.

### API 엔드포인트

#### 1. 채팅 요청

```bash
POST /chat
Content-Type: application/json

{
  "message": "2024-01-15의 USD/KRW 환율을 조회해줘",
  "date": "2024-01-15",
  "user_id": "user123",
  "context": {}
}
```

**응답:**

```json
{
  "answer": "2024-01-15의 USD/KRW 환율은 1,300원입니다.",
  "references": [
    {
      "source": "rdb",
      "data": {...}
    }
  ],
  "actions": [
    {
      "tool": "rdb_query_hard",
      "input": {...},
      "output": {...}
    }
  ],
  "status": "completed"
}
```

#### 2. 에이전트 목록 조회

```bash
GET /agents
```

#### 3. 특정 에이전트 직접 호출

```bash
POST /agent/{agent_name}
Content-Type: application/json

{
  "task": "작업 설명",
  "context": {}
}
```

### Python 코드에서 직접 사용

```python
from src.agents.supervisor.supervisor import Supervisor
from src.utils.state import StateManager

# Supervisor 초기화
supervisor = Supervisor()
state_manager = StateManager()

# 사용자 질의 처리
user_query = "2024-01-15의 USD/KRW 환율을 조회해줘"
user_id = "user123"
date = "2024-01-15"

# State 조회 또는 생성
state = state_manager.get_state(user_id)

# Supervisor 실행
result = await supervisor.process_query(
    user_query=user_query,
    user_id=user_id,
    date=date,
    agent_state=state
)

print(result["final_answer"])
```

## 🏗 아키텍처

### 멀티 에이전트 시스템 구조

```
┌─────────────────────────────────────────┐
│           Supervisor Agent              │
│  (LangGraph 기반 오케스트레이션)        │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
   ┌───▼───┐      ┌───▼───┐
   │ ReAsk │      │Routing│
   │ Agent │      │Decision│
   └───┬───┘      └───┬───┘
       │              │
       │      ┌───────┴───────┐
       │      │               │
   ┌───▼───┐ ┌▼───┐      ┌───▼───┐
   │Market │ │ReAct│      │HandsOff│
   │Info   │ │Agent│      │ Agent │
   │Agent  │ └────┘      └───────┘
   └───┬───┘
       │
   ┌───┴──────────────────┐
   │                       │
┌──▼──┐  ┌───▼──┐  ┌─────▼──┐
│ RDB │  │ VDB  │  │ Web    │
│Tool │  │Tool  │  │Search  │
└─────┘  └──────┘  └────────┘
```

### 데이터 흐름

1. **사용자 질의** → Supervisor
2. **ReAsk Agent**: 질의 명확성 확인
3. **Routing Decision**: 적절한 에이전트 선택
4. **Agent Execution**: 선택된 에이전트가 도구 사용하여 정보 수집
5. **HandsOff Agent**: 필요시 사용자에게 직접 전달
6. **Final Answer**: 최종 답변 생성

### 주요 컴포넌트

#### 1. Supervisor (LangGraph)
- StateGraph를 사용한 상태 기반 워크플로우 관리
- Checkpoint를 통한 상태 영속성
- 조건부 라우팅을 통한 에이전트 선택

#### 2. Tools
- `@tool` decorator를 사용한 함수형 도구
- Pydantic 스키마를 통한 타입 안정성
- 통일된 에러 처리

#### 3. Utils
- **llm.py**: LLM 호출, LangSmith 추적, 메시지 변환
- **tools.py**: Tool 관리, 에러 처리, 스키마 정의
- **agents.py**: BaseAgent 클래스
- **state.py**: 상태 관리

## 📚 API 문서

API 서버 실행 후 다음 URL에서 자동 생성된 문서를 확인할 수 있습니다:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 개발 가이드

### 새로운 에이전트 추가

1. `src/agents/` 디렉토리에 새 에이전트 파일 생성
2. `BaseAgent`를 상속받아 구현:

```python
from src.utils.agents import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="my_agent",
            description="내 에이전트 설명"
        )
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None):
        # 구현
        pass
```

3. `src/agents/__init__.py`에 추가
4. `src/agents/supervisor/supervisor.py`의 `_build_graph()`에 노드 추가

### 새로운 도구 추가

1. `src/tools/` 디렉토리에 새 도구 파일 생성
2. `@tool` decorator와 `@handle_tool_error` 사용:

```python
from src.utils.tools import tool, handle_tool_error

@tool
@handle_tool_error("my_tool")
async def my_tool(param: str) -> Dict[str, Any]:
    """도구 설명"""
    # 구현
    pass
```

3. `src/tools/__init__.py`에 추가
4. `src/utils/tools.py`의 `get_all_tools()`에 추가

### 코드 스타일

- **타입 힌트**: 모든 함수에 타입 힌트 사용
- **문서화**: 모든 클래스와 함수에 docstring 작성
- **에러 처리**: `ToolError`를 통한 통일된 에러 처리
- **로깅**: `get_logger()`를 통한 구조화된 로깅

## 🧪 테스트

```bash
# 테스트 실행 (테스트 파일이 있는 경우)
pytest tests/
```

## 📝 라이선스

이 프로젝트의 라이선스는 `LICENSE` 파일을 참조하세요.

## 🤝 기여

기여를 환영합니다! 이슈를 생성하거나 Pull Request를 제출해주세요.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

---

**참고**: 이 프로젝트는 LangChain과 LangGraph의 최신 기능을 활용하여 구축되었으며, 지속적으로 업데이트되고 있습니다.

