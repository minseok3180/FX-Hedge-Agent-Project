# 환 리스크 헷지 강화학습 에이전트

## 📋 프로젝트 개요

이 프로젝트는 yonju 폴더에서 수집된 경제 데이터와 chaewon 폴더에서 구현된 시계열 예측 모델의 결과를 통합하여, 강화학습 기반의 환 리스크 헷지 전략 에이전트를 구현합니다.

## 🎯 주요 기능

### 1. 데이터 통합
- **yonju 폴더**: ECOS API를 통한 경제 데이터 수집 및 전처리
- **chaewon 폴더**: TimeXer 스타일 시계열 예측 모델
- **통합**: 과거 2년 데이터 + 향후 1달 예측 데이터

### 2. 강화학습 환경
- **상태 공간**: 730일 과거 데이터 + 31일 예측 데이터 + 포지션 정보
- **행동 공간**: 9가지 헷지 전략 (A0~A8)
- **보상 함수**: 환율 변동에 따른 수익/손실 + 리스크 조정
- **데이터**: usekrw(target) 컬럼을 사용한 USD/KRW 환율 데이터

### 3. 헷지 전략
- **A0**: 유지 (Hold)
- **A1**: 선도 1M로 +25%p 증액
- **A2**: 선도 3M로 +25%p 증액
- **A3**: 선도 최근만기에서 -25%p 감액/청산
- **A4**: 선물 근월물로 +25%p 증액
- **A5**: 선물 근월물에서 -25%p 감액/청산
- **A6**: 롤오버 (Rollover)
- **A7**: 전량 청산 (Flatten)
- **A8**: 스위치 25%p (Switch)

## 🏗️ 프로젝트 구조

```
notebook/minseok/
├── README.md                 # 프로젝트 설명서
├── requirements.txt          # 필요한 패키지 목록
├── data_integration.py      # 데이터 통합 스크립트
├── hedge_environment.py     # 강화학습 환경 구현
├── hedge_agent.py          # PPO 기반 강화학습 에이전트
├── main.py                 # 메인 실행 파일
├── test_agent.py           # 에이전트 테스트 스크립트
├── integrated_data/         # 통합된 데이터 저장소
├── models/                  # 학습된 모델 저장소
└── results/                 # 결과 및 시각화 저장소
```

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv hedge_env
source hedge_env/bin/activate  # Linux/Mac
# 또는
hedge_env\Scripts\activate     # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행
```bash
# 모든 단계 실행 (데이터 통합 → 학습 → 평가 → 백테스팅)
python main.py --mode all --episodes 1000 --eval-episodes 100
```

### 3. 단계별 실행
```bash
# 데이터 통합만 실행
python main.py --mode data

# 학습만 실행 (기존 데이터 사용)
python main.py --mode train --episodes 1000

# 평가만 실행 (기존 모델 사용)
python main.py --mode eval --eval-episodes 100

# 백테스팅만 실행
python main.py --mode backtest
```

### 4. 사용자 지정 데이터 사용
```bash
# 특정 데이터 파일 사용
python main.py --mode all --data-path /path/to/your/data.csv
```

## 📊 주요 파라미터

### 환경 파라미터
- **초기 자본금**: 1,000,000 USD (기본값)
- **최대 헷지 비율**: 100% (기본값)
- **거래 비용**: 0.1% (기본값)
- **무위험 수익률**: 2% (기본값)

### 학습 파라미터
- **학습률**: 3e-4
- **할인 계수**: 0.99
- **GAE 람다**: 0.95
- **PPO 클립 비율**: 0.2
- **가치 손실 계수**: 0.5
- **엔트로피 계수**: 0.01

## 🔍 결과 분석

### 1. 학습 과정
- 에피소드별 보상 변화
- 손실 함수 수렴 과정
- 액션 분포 변화

### 2. 성능 지표
- 평균 보상 및 표준편차
- 샤프 비율
- 최종 자본금

### 3. 백테스팅 결과
- 자본금 변화 추이
- 헷지 비율 변화
- 액션 선택 패턴
- 누적 보상

## 📈 사용 예시

### 1. 빠른 테스트
```bash
# 간단한 테스트 실행
python test_agent.py
```

### 2. 커스텀 학습
```python
from hedge_environment import HedgeEnvironment
from hedge_agent import HedgeAgent

# 환경 생성
env = HedgeEnvironment(data_path="your_data.csv")

# 에이전트 생성
agent = HedgeAgent(state_dim=765, action_dim=9)

# 학습 실행
training_history = agent.train(env, num_episodes=500)

# 결과 시각화
agent.plot_training_history()
```

### 3. 모델 저장/로드
```python
# 모델 저장
agent.save_model("my_hedge_model.pth")

# 모델 로드
agent.load_model("my_hedge_model.pth")
```

## 🛠️ 개발 및 확장

### 1. 새로운 헷지 전략 추가
`hedge_environment.py`의 `HedgeAction` enum과 `_execute_action` 메서드를 수정

### 2. 보상 함수 커스터마이징
`_calculate_reward` 메서드를 수정하여 다양한 리스크 지표 반영

### 3. 상태 공간 확장
`_get_state` 메서드를 수정하여 추가 경제 지표 포함

## 📝 주의사항

1. **데이터 품질**: 실제 운영시에는 yonju와 chaewon의 실제 데이터 사용
2. **리스크 관리**: 실제 거래에 적용시 추가적인 리스크 관리 로직 필요
3. **성능 최적화**: 대용량 데이터 처리시 배치 처리 및 멀티프로세싱 고려
4. **모델 검증**: 실제 시장 환경에서의 성능 검증 필요

## 🤝 기여 방법

1. 이슈 리포트 생성
2. 기능 요청 제안
3. 코드 풀 리퀘스트
4. 문서 개선 제안

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 통해 연락해 주세요.
