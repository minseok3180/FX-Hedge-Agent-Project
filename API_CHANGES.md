# API 구조 변경 사항

## POST 요청 구조 변경

### 과거 구조 (Deprecated)
```json
{
  "message": "2024-01-15 환율 정보 알려줘"
}
```

### 현재 구조 (Current)
```json
{
  "message": "2024-01-15 환율 정보 알려줘",
  "date": "2024-01-15",
  "user_id": "M000831"
}
```

### 필수 필드
- `message` (string): 사용자 질의
- `date` (string): 질문하는 날짜 (YYYY-MM-DD 형식)
- `user_id` (string): 사용자 아이디

### 선택 필드
- `context` (object, optional): 추가 컨텍스트 정보

### 예시 요청
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "2024-01-15 환율 정보 알려줘",
    "date": "2024-01-15",
    "user_id": "M000831"
  }'
```

### 응답 구조
```json
{
  "answer": "2024-01-15의 USD/KRW 환율은 1,157.8원입니다.",
  "reference": [
    {
      "source": "rdb",
      "query": "SELECT date, usdkrw FROM eiExchangeRate WHERE date = '2024-01-15'",
      "results_count": 1,
      "metadata": {}
    }
  ],
  "action": [],
  "agent": "market_information",
  "metadata": {
    "supervisor_decision": {...},
    "task": "...",
    "status": "success"
  }
}
```

## RDB 쿼리 Placeholder 지원

RDB 쿼리에서 `{user_id}`, `{date}` 등의 placeholder를 사용할 수 있으며, 실행 시 state에서 값을 자동으로 추출하여 치환합니다.

### 지원하는 Placeholder
- `{user_id}`: 현재 사용자 ID
- `{date}`: 현재 질문 날짜 (YYYY-MM-DD)

### 사용 예시

#### 쿼리 정의 (rdb_hard_queries.py)
```python
rdb_hard_queries = {
    "get_by_date": """
        SELECT date, usdkrw
        FROM eiExchangeRate
        WHERE date = '{date}'
        LIMIT 1
    """,
    "get_user_info_by_id": """
        SELECT user_id, user_name, user_krw, user_usd
        FROM user_info
        WHERE user_id = '{user_id}'
        LIMIT 1
    """,
    "get_all_users": """
        SELECT user_id, user_name, user_krw, user_usd
        FROM user_info
    """
}
```

#### 에이전트에서 사용
```python
# MarketInformationAgent에서
async def execute(self, task: str, context: Optional[Dict[str, Any]] = None):
    # context에는 state가 포함되어 있음
    state = context.get("state") if context else None
    
    # Placeholder 사용 시 state 전달
    results = await self.rdb_hard.execute("get_by_date", state=state)
    # {date}가 자동으로 state에서 추출된 값으로 치환됨
```

#### 실행 시 자동 치환
- `{date}` → state에서 가져온 date 값 (예: "2024-01-15")
- `{user_id}` → state에서 가져온 user_id 값 (예: "M000831")

### State에서 값 추출 우선순위
1. `state.get("user_id")` 또는 `state.get("date")`
2. `state.get("current_context", {}).get("user_id")` 또는 `state.get("current_context", {}).get("date")`
3. `state.get("conversation_history", [])[-1].get("date")` (최신 대화 턴의 date)

### 주의사항
- Placeholder는 문자열 치환 방식으로 동작합니다.
- SQL Injection 방지를 위해 placeholder 값은 검증됩니다.
- `%s` 파라미터 바인딩과 함께 사용할 수 없습니다 (placeholder 사용 시).
- Placeholder가 없는 쿼리는 기존처럼 `params` 튜플을 사용할 수 있습니다.

