# 외환 헷지전략 에이전트 (FX Hedge Strategy Agent)

외환 헷지전략을 분석하고 트레이딩을 진행하는 멀티 에이전트 LLMOps 모델입니다.

## 프로젝트 구조

```
src/
├── __init__.py
├── config.py                 # 설정 관리 (DB, LLM 모델)
├── main.py                   # 메인 실행 파일
├── agent/                    # 에이전트 모듈들
│   ├── __init__.py
│   ├── base_agent.py         # 기본 에이전트 클래스
│   ├── search_agent.py       # 검색 에이전트 (뉴스, 시계열 데이터)
│   └── trading_agent.py     # 거래 에이전트 (RAG, 거래 실행)
└── supervisor/               # 중앙 관리자
    ├── __init__.py
    └── supervisor.py         # Supervisor 구현
```

## 주요 기능

### 1. Search Agent
- **웹 API를 통한 헷지 관련 뉴스 검색**
- **TSDB 시계열 정보 조회** (환율 데이터)
- **시장 지표 분석**
- **경제 캘린더 검색**

### 2. Trading Agent
- **VDB 헷지 전략 정보 RAG 참조**
- **사용자 포트폴리오 관리**
- **외환 거래 실행**
- **헷지 비율 계산**
- **전략 시뮬레이션**

### 3. Supervisor
- **사용자 질문 분석 및 라우팅**
- **에이전트 간 작업 조율**
- **결과 통합 및 피드백 제공**
- **대화 컨텍스트 관리**

## 데이터베이스 구성

### TSDB (Time Series Database)
- 환율 및 외환 관련 시계열 데이터 저장
- 기술적 지표 계산 및 저장

### VDB (Vector Database)
- 헷지 전략 정보 벡터화 저장
- RAG를 통한 전략 검색 및 분석

### RDB (Relational Database)
- 사용자 정보 및 거래 기록
- 포트폴리오 관리 데이터

## LLM 모델

- **KT Mi:dm 2.0 모델** 사용 (transformers 라이브러리)
- **모델명**: `K-intelligence/Midm-2.0-Base-Instruct`
- **최대 토큰**: 4,096 토큰
- **한국어 특화** 모델로 외환 헷지전략 분석에 최적화
- **로컬 실행**: API 키 없이 transformers 라이브러리를 통해 직접 로드
- **설정 파일**을 통한 모델 관리

## 실행 방법

### Streamlit 웹 앱 (권장)
```bash
# 간단한 실행
python run_app.py

# 또는 직접 실행
streamlit run src/streamlit_app.py
```

### 대화형 모드
```bash
python src/main.py --mode interactive
```

### 배치 모드
```bash
python src/main.py --mode batch --queries "질문1" "질문2" "질문3"
```

### 설정 검증
```bash
python src/main.py --config-check
```

## 환경 설정

환경변수를 통한 설정 관리:

```bash
# 데이터베이스 설정
export TSDB_HOST=localhost
export TSDB_PORT=8086
export VDB_HOST=localhost
export VDB_PORT=8000
export RDB_HOST=localhost
export RDB_PORT=5432

# LLM 설정 (로컬 환경)
export LLM_MODEL_NAME=K-intelligence/Midm-2.0-Base-Instruct
export LLM_USE_LOCAL=true
export LLM_DEVICE=auto
export LLM_MAX_TOKENS=4096
export EMBEDDING_MODEL=jhgan/ko-sroberta-multitask

# 에이전트 설정
export SEARCH_TIMEOUT=30
export TRADING_TIMEOUT=60
export MAX_ITERATIONS=10
```

## 향후 개발 예정

### Auto Collector (추후 추가)
- 매일 정해진 시간에 시계열 데이터 수집
- LSTM 모델을 통한 예측 정보 생성
- 자동화된 데이터 파이프라인

## MVP 범위

현재 MVP에는 다음 기능들이 포함됩니다:
- ✅ Search Agent (뉴스 검색, 시계열 조회)
- ✅ Trading Agent (RAG 분석, 거래 실행)
- ✅ Supervisor (라우팅, 피드백)
- ✅ 설정 관리 시스템
- ✅ 대화형 및 배치 모드

## 기술 스택

- **Python 3.8+**
- **asyncio** (비동기 처리)
- **aiohttp** (웹 API 호출)
- **PostgreSQL** (RDB)
- **InfluxDB** (TSDB)
- **Chroma/Weaviate** (VDB)
- **KT Mi:dm 2.0** (LLM)
