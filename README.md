# FX Hedge Agent Project

환율 헤지펀트 매니지먼트를 돕는 챗봇 에이전트 시스템

## 프로젝트 개요

- **목표**: K intelligence 해커톤 2025 (Track1: AI Agent 개발) 참가
- **주제**: FX Hedge Agent
- **아키텍처**: Supervisor 구조의 멀티 에이전트 시스템
- **LLM**: GPT-4.1 모델 기반
- **인프라**: GCP MariaDB, Qdrant (로컬 개발용 Docker Compose 지원)

## 프로젝트 구조

```
FX-Hedge-Agent-Project-1/
├── api/                        # API 관련 코드
│   ├── __init__.py
│   └── main.py                 # FastAPI 메인 애플리케이션
├── frontend/                   # 프론트엔드 (향후 개발)
│   ├── public/                 # 정적 파일
│   ├── src/                    # 소스 코드
│   │   ├── components/         # 컴포넌트
│   │   ├── pages/              # 페이지
│   │   ├── hooks/              # 커스텀 훅
│   │   ├── services/           # API 서비스
│   │   ├── utils/              # 유틸리티
│   │   └── styles/             # 스타일
│   └── README.md
├── src/                        # 에이전트 모델 관련 코드
│   ├── config/                 # 설정 로더
│   │   ├── __init__.py
│   │   └── settings.py         # JSON + .env 로더
│   ├── supervisor/             # Supervisor (멀티 에이전트 오케스트레이터)
│   │   ├── __init__.py
│   │   └── supervisor.py
│   ├── agents/                 # 하위 에이전트들
│   │   ├── __init__.py
│   │   ├── base_agent.py       # 기본 에이전트 클래스
│   │   ├── web_search_agent.py # 웹서치 에이전트
│   │   └── rag_agent.py        # RAG 에이전트
│   ├── tools/                  # 에이전트가 사용하는 도구들
│   │   ├── __init__.py
│   │   ├── web_search.py       # 웹 검색 도구
│   │   ├── database.py         # MariaDB 쿼리 도구
│   │   └── qdrant_client.py    # Qdrant 벡터 DB 클라이언트
│   ├── prompts/                # 프롬프트 템플릿
│   │   ├── __init__.py
│   │   ├── supervisor_prompt.py
│   │   ├── web_search_prompt.py
│   │   └── rag_prompt.py
│   ├── query/                  # 쿼리 빌더
│   │   ├── __init__.py
│   │   └── query_builder.py
│   └── models/                 # 모델 관련 (향후 확장)
│       └── __init__.py
├── config.json                 # 설정 템플릿 (환경 변수 플레이스홀더)
├── .env                        # 환경 변수 (gitignore, 실제 값)
├── .env.example               # 환경 변수 예시
├── docker-compose.yml          # Docker Compose 설정
├── Dockerfile                  # API 컨테이너 이미지
├── requirements.txt            # Python 의존성
└── README.md
```

## 주요 기능

### 1. Supervisor 구조
- 사용자 질문을 분석하여 적절한 하위 에이전트에게 작업 할당
- 멀티 에이전트 시스템의 중앙 오케스트레이터 역할

### 2. 하위 에이전트

#### Web Search Agent
- 환율 정보를 웹에서 검색
- 최신 환율 뉴스, 시장 동향 조회

#### RAG Agent
- MariaDB에서 데이터 조회
- Qdrant 벡터 검색을 통한 RAG (Retrieval-Augmented Generation)
- 과거 데이터 분석 및 통계 제공

## 설치 및 실행

### 1. 환경 변수 설정

`.env.example`을 참고하여 `.env` 파일을 생성하고 필요한 값들을 설정하세요:

```bash
cp .env.example .env
# .env 파일을 열어서 실제 값 입력
```

**설정 구조:**
- `config.json`: 설정 구조 템플릿 (환경 변수 플레이스홀더 포함, Git에 포함)
- `.env`: 실제 환경 변수 값 (Git에 포함되지 않음, `.gitignore`에 추가됨)
- 빌드 시 `.env`의 값들이 `config.json`의 플레이스홀더를 대체

필수 환경 변수:
- `OPENAI_API_KEY`: OpenAI API 키
- `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`: MariaDB 연결 정보
- `QDRANT_HOST`, `QDRANT_PORT`: Qdrant 연결 정보

### 2. Docker Compose로 실행

```bash
docker compose up -d
```

이 명령어는 다음 서비스들을 시작합니다:
- FastAPI 애플리케이션 (포트 8000)
- MariaDB (포트 3306)
- Qdrant (포트 6333)

### 3. 로컬 개발 환경에서 실행

```bash
# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# FastAPI 서버 실행
python -m uvicorn api.main:app --reload
```

## API 사용법

### 1. 헬스 체크

```bash
curl http://localhost:8000/health
```

### 2. 사용 가능한 에이전트 조회

```bash
curl http://localhost:8000/agents
```

### 3. 채팅 요청 (Supervisor가 자동으로 에이전트 선택)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "오늘 USD/KRW 환율은 얼마인가요?"
  }'
```

### 4. 특정 에이전트에 직접 요청

```bash
# 웹서치 에이전트
curl -X POST http://localhost:8000/agent/web_search_agent \
  -H "Content-Type: application/json" \
  -d '{
    "message": "최근 환율 동향을 알려주세요"
  }'

# RAG 에이전트
curl -X POST http://localhost:8000/agent/rag_agent \
  -H "Content-Type: application/json" \
  -d '{
    "message": "지난 달 USD/KRW 평균 환율을 조회해주세요"
  }'
```

## Postman 사용 예시

1. **POST** `/chat`
   - Body (JSON):
     ```json
     {
       "message": "USD/KRW 환율 정보를 알려주세요"
     }
     ```

2. **GET** `/agents`
   - 사용 가능한 에이전트 목록 조회

3. **POST** `/agent/{agent_name}`
   - 특정 에이전트에 직접 요청

## 향후 계획

- [ ] 환율 시계열 예측 모델 통합
- [ ] 추가 하위 에이전트 구현
- [ ] 프론트엔드 개발
- [ ] 모니터링 및 로깅 시스템 구축
- [ ] 성능 최적화

## 참고 자료

- HedgeAgents: A Balanced-aware Multi-agent Financial Trading System
  - https://arxiv.org/html/2502.13165v1

## 라이선스

LICENSE 파일을 참조하세요.