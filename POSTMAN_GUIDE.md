# Postman 테스트 가이드

## 1. Docker Compose 실행

프로젝트 루트 디렉토리에서 다음 명령어를 실행하세요:

```bash
docker compose up -d
```

또는 로그를 보면서 실행하려면:

```bash
docker compose up
```

## 2. 서비스 확인

서비스가 정상적으로 실행되었는지 확인:

```bash
# 컨테이너 상태 확인
docker compose ps

# API 로그 확인
docker compose logs -f api
```

## 3. Postman 테스트

### 기본 정보
- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`

---

### 엔드포인트 1: 헬스 체크

**GET** `http://localhost:8000/health`

**응답 예시:**
```json
{
  "status": "healthy"
}
```

---

### 엔드포인트 2: 루트 엔드포인트

**GET** `http://localhost:8000/`

**응답 예시:**
```json
{
  "message": "FX Hedge Agent API",
  "version": "1.0.0",
  "status": "running"
}
```

---

### 엔드포인트 3: 사용 가능한 에이전트 목록

**GET** `http://localhost:8000/agents`

**응답 예시:**
```json
{
  "web_search_agent": {
    "name": "web_search_agent",
    "description": "..."
  },
  "rag_agent": {
    "name": "rag_agent",
    "description": "..."
  }
}
```

---

### 엔드포인트 4: Supervisor를 통한 채팅 (추천)

**POST** `http://localhost:8000/chat`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "message": "2024-01-15 환율 정보 알려줘"
}
```

**다른 예시 질문:**
```json
{
  "message": "최근 10일간 환율 데이터 보여줘"
}
```

```json
{
  "message": "2024-01-01부터 2024-01-31까지 환율 정보"
}
```

**응답 예시:**
```json
{
  "answer": "2024-01-15일의 환율 정보는 다음과 같습니다...",
  "agent": "rag_agent",
  "metadata": {
    "supervisor_decision": {
      "agent": "rag_agent",
      "reasoning": "데이터베이스 조회가 필요한 질문입니다."
    },
    "task": "2024-01-15 환율 정보 알려줘",
    "status": "success"
  }
}
```

---

### 엔드포인트 5: RAG 에이전트 직접 호출

**POST** `http://localhost:8000/agent/rag_agent`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "message": "2024-01-15 환율 정보 알려줘"
}
```

**응답 예시:**
```json
{
  "answer": "2024-01-15일의 환율 정보는 다음과 같습니다...",
  "agent": "rag_agent",
  "metadata": {
    "agent": "rag_agent",
    "task": "2024-01-15 환율 정보 알려줘",
    "db_results": [
      {
        "date": "2024-01-15",
        "usdkrw": 1320.5,
        "us_ex": 1000000,
        ...
      }
    ],
    "status": "success"
  }
}
```

---

### 엔드포인트 6: Web Search 에이전트 직접 호출

**POST** `http://localhost:8000/agent/web_search_agent`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "message": "최신 환율 뉴스 알려줘"
}
```

---

## 4. 테스트 시나리오

### 시나리오 1: 특정 날짜 환율 조회
```json
POST http://localhost:8000/chat
{
  "message": "2024-01-15 환율 정보"
}
```

### 시나리오 2: 날짜 범위 조회
```json
POST http://localhost:8000/chat
{
  "message": "2024-01-01부터 2024-01-31까지 환율 데이터"
}
```

### 시나리오 3: 최신 데이터 조회
```json
POST http://localhost:8000/chat
{
  "message": "최근 5일간 환율 데이터"
}
```

### 시나리오 4: 다양한 날짜 형식
```json
POST http://localhost:8000/chat
{
  "message": "2024년 1월 15일 환율 정보"
}
```

---

## 5. 문제 해결

### API가 응답하지 않는 경우
1. 컨테이너가 실행 중인지 확인: `docker compose ps`
2. 로그 확인: `docker compose logs api`
3. 포트가 사용 중인지 확인: `lsof -i :8000`

### 데이터베이스 연결 오류
1. MariaDB 컨테이너 상태 확인: `docker compose logs mariadb`
2. 환경 변수 확인: `.env` 파일에 DB 정보가 올바른지 확인

### LangSmith 추적 확인
- LangSmith 대시보드에서 `fx-hedge-agent` 프로젝트 확인
- API 호출이 추적되는지 확인

---

## 6. 환경 변수 확인

`.env` 파일에 다음 변수들이 설정되어 있어야 합니다:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

# LangSmith
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=fx-hedge-agent
LANGSMITH_TRACING=true

# Database
DB_HOST=mariadb
DB_PORT=3306
DB_USER=fx_user
DB_PASSWORD=fx_password
DB_NAME=fx_hedge

# Qdrant (선택사항)
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=

# Web Search (선택사항)
WEB_SEARCH_API_KEY=
WEB_SEARCH_ENGINE_ID=
```

