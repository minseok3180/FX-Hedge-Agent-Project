import json

# USD 뉴스 결과 불러오기
with open("usd_news.json", "r", encoding="utf-8") as f:
    usd_news = json.load(f)

# 간단한 Rule-based RiskScore 예시
def calculate_risk_score(usd_news):
    # risk 값에 따라 점수 부여
    risk_mapping = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
        "unknown": 0.5
    }
    
    # 뉴스 risk + volatility_impact 고려
    base_score = risk_mapping.get(usd_news.get("risk", "unknown"), 0.5)
    vol_adjust = usd_news.get("volatility_impact", 0.5)
    
    # 최종 RiskScore: base_score * volatility 영향
    risk_score = base_score * (0.5 + 0.5 * vol_adjust)
    return round(risk_score, 2)

risk_score = calculate_risk_score(usd_news)
print("USD RiskScore:", risk_score)
