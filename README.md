# 0820 kick-off meeting
##  Project Goal
- K intelligence 해커톤 2025 (Track1: AI Agent 개발) 참가 (due to 9/10 09:59)
- 주제 : FX Hedge Agent 

##  Detail
- HedgeAgents: A Balanced-aware Multi-agent Financial Trading System ( https://arxiv.org/html/2502.13165v1 )의 아이디어를 backbone삼아 구현
- KT의 믿:음을 LLM agent로 사용
- 데이터 및 추론 도메인은 '환율(FX)에 대한 Hedge 전략'으로 제한
- 환율에 대한 time-series forecasting model을 서브로 구현하여 예측환율 정보까지 RAG형식으로 적용하는 것을 시도
  (이에 대한 추가 논문 및 모델 논의 필요 / 적용 가능 및 개선 유무 정도...)
- B2B/B2G 환경에서 현업에 도움되는 방식으로 output 설정 (ex. chatbot)

## To-do
- meeting : 09/24(일) 14:00~16:00 예정(비대면)
- 기반이 되는 논문 스터디 및 적용 계획 확인
- backbone code 작성