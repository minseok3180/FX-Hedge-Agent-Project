# 💱 KT 믿음 mini 기반 외환 거래 AI 에이전트

KT의 믿음 mini 모델을 활용한 지능형 외환 거래 에이전트 시스템입니다. RAG(Retrieval-Augmented Generation) 기술과 실시간 시장 데이터 분석을 통해 전문적인 거래 조언을 제공합니다.

## 🚀 주요 기능

### 📊 실시간 시장 분석
- 주요 통화쌍(EUR/USD, GBP/USD, USD/JPY 등) 실시간 데이터 수집
- 기술적 지표 계산 (RSI, MACD, 볼린저 밴드, 이동평균)
- 시장 심리 및 추세 분석

### 🤖 AI 거래 추천
- KT 믿음 mini 모델 기반 지능형 거래 신호 생성
- RSI, MACD, 추세 분석을 통한 종합적 매매 신호
- 목표가격 및 손절매 가격 자동 계산
- 거래 신뢰도 및 신호 강도 제공

### 📈 포트폴리오 관리
- 실시간 포지션 추적
- 손익 계산 및 성과 분석
- 승률 및 수익률 통계

### 💬 AI 상담
- 외환 거래 관련 질문에 대한 전문가 수준 답변
- RAG 시스템을 통한 지식 베이스 활용
- 한국어 자연어 처리

## 🛠️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FX Data       │    │   KT 믿음       │    │   FX Agent      │
│   Collector     │    │   mini RAG      │    │                 │
│                 │    │   System        │    │                 │
│ • Yahoo Finance │    │ • LLM Model     │    │ • Market        │
│ • Technical     │    │ • Vector DB     │    │   Analysis      │
│   Indicators    │    │ • Knowledge     │    │ • Trading       │
│ • Market        │    │   Base          │    │   Signals       │
│   Sentiment     │    │ • Document      │    │ • Portfolio     │
└─────────────────┘    │   Retrieval     │    │   Management    │
                       └─────────────────┘    └─────────────────┘
```

## 📋 요구사항

### 시스템 요구사항
- Python 3.8 이상
- 최소 8GB RAM (16GB 권장)
- GPU 지원 시 더 빠른 처리 가능

### 필수 패키지
- `transformers`: KT 믿음 mini 모델 로딩
- `torch`: 딥러닝 프레임워크
- `langchain`: RAG 시스템 구축
- `chromadb`: 벡터 데이터베이스
- `yfinance`: 외환 데이터 수집
- `streamlit`: 웹 인터페이스
- `plotly`: 차트 시각화

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd FX-Hedge-Agent-Project
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 실행

#### 콘솔 모드 (기본)
```bash
python src/main.py
```

#### 웹 앱 모드
```bash
python src/main.py --mode web
```

#### 직접 Streamlit 실행
```bash
cd src
streamlit run streamlit_app.py
```

## 📱 사용법

### 콘솔 모드
1. **시스템 초기화**: KT 믿음 mini 모델 및 외환 지식 베이스 로딩
2. **메뉴 선택**: 시장 분석, AI 거래 추천, 포트폴리오 현황, AI 상담 중 선택
3. **통화쌍 선택**: 분석할 주요 통화쌍 선택
4. **결과 확인**: AI 분석 결과 및 거래 추천 확인
5. **거래 실행**: 시뮬레이션 모드로 거래 실행

### 웹 앱 모드
1. **브라우저 자동 실행**: Streamlit 앱이 자동으로 브라우저에서 열림
2. **탭 기반 인터페이스**: 
   - 📊 시장 분석: 실시간 차트 및 기술적 지표
   - 🤖 AI 거래 추천: 지능형 거래 신호 생성
   - 📈 포트폴리오: 포지션 관리 및 성과 분석
   - 💬 AI 상담: 전문가 수준 답변

## 🔧 주요 모듈 설명

### `fx_data_collector.py`
- Yahoo Finance API를 통한 실시간 외환 데이터 수집
- 기술적 지표 계산 (RSI, MACD, 볼린저 밴드, 이동평균)
- 시장 심리 지표 생성

### `kt_rag_system.py`
- KT 믿음 mini 모델 로딩 및 관리
- ChromaDB를 활용한 벡터 데이터베이스 구축
- RAG 기반 질의응답 시스템

### `fx_agent.py`
- 시장 분석 및 거래 신호 생성
- 포트폴리오 관리 및 포지션 추적
- 리스크 관리 및 손익 계산

### `streamlit_app.py`
- 사용자 친화적 웹 인터페이스
- 실시간 차트 및 데이터 시각화
- 대화형 AI 상담 시스템

## 📊 지원 통화쌍

- **EUR/USD** (EURUSD=X): 유로/달러
- **GBP/USD** (GBPUSD=X): 파운드/달러
- **USD/JPY** (USDJPY=X): 달러/엔
- **USD/CHF** (USDCHF=X): 달러/프랑
- **AUD/USD** (AUDUSD=X): 호주달러/달러
- **USD/CAD** (USDCAD=X): 달러/캐나다달러
- **NZD/USD** (NZDUSD=X): 뉴질랜드달러/달러
- **EUR/JPY** (EURJPY=X): 유로/엔

## ⚠️ 주의사항

### 투자 위험 고지
- 이 시스템은 교육 및 연구 목적으로 제작되었습니다
- 실제 거래에 사용할 경우 발생하는 손실에 대해 책임지지 않습니다
- 투자는 본인의 판단과 책임 하에 진행해야 합니다

### 시스템 제한사항
- 인터넷 연결이 필요합니다
- Yahoo Finance API의 데이터 가용성에 의존합니다
- 모델 로딩 시 초기 시간이 소요될 수 있습니다

## 🔮 향후 개발 계획

- [ ] 실시간 뉴스 감정 분석 통합
- [ ] 다중 시간대 분석 지원
- [ ] 백테스팅 기능 추가
- [ ] 위험 관리 알고리즘 고도화
- [ ] 모바일 앱 개발

## 📞 지원 및 문의

시스템 사용 중 문제가 발생하거나 개선 사항이 있으시면 이슈를 등록해 주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**⚠️ 투자 경고**: 이 시스템은 교육 목적으로만 사용하시기 바랍니다. 실제 투자 결정은 충분한 검토와 전문가 상담을 통해 이루어져야 합니다.