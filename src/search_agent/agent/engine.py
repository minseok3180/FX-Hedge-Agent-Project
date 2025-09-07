#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd

from search_agent.tool import ECOSTool, YFTool, NewsTool, ForecastTool
from search_agent.config import DEFAULT_LOOKBACK_DAYS, TICKER_ALIAS


class SearchAgent:
    """
    Tools:
      - NewsTool        : 뉴스 RAG 검색
      - YFTool          : 야후 파이낸스(환율·해외시장 일별)
      - ECOSTool        : ECOS(한국 거시 지표)
      - ForecastTool    : 시계열 모델 예측 결과(RAG/Chroma)
    """

    def __init__(self):
        self.news = NewsTool()
        self.yf = YFTool()
        self.ecos = ECOSTool()
        self.forecast = ForecastTool()

    # ------------ 날짜 파싱 ------------
    def _parse_date_range(self, text: str) -> Tuple[str, str]:
        m = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        if len(m) >= 2:
            return m[0], m[1]
        if len(m) == 1:
            return m[0], m[0]
        today = datetime.today().date()
        start = (today - timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        return start, end

    # ------------ 간단 라우팅 ------------
    def run(self, raw_query: str):
        ql = raw_query.lower()
        start, end = self._parse_date_range(raw_query)

        # 1) Forecast 우선 라우팅
        if any(k in ql for k in ["예측", "forecast", "prediction", "모델 결과"]):
            return self.forecast.search(start=start, end=end, top_k=10)

        # 2) News
        if "뉴스" in ql or "기사" in ql:
            return self.news.search(query=raw_query, start=start, end=end, top_k=8)

        # 3) ECOS (코드 직접 지정 또는 거시 키워드)
        if re.search(r"[0-9A-Z]{6}\|?[A-Za-z0-9]*", raw_query) or any(k in ql for k in ["gdp", "cpi", "물가", "pmi", "경상수지", "수출", "수입", "금리"]):
            # 코드가 명시되면 그대로
            m = re.search(r"([0-9A-Z]{6})(\|[A-Za-z0-9]+)*", raw_query)
            if m:
                stat = m.group(1)
                items = raw_query.split("|")[1:] if "|" in raw_query else []
                return self.ecos.fetch(stat_code=stat, item_codes=items, start=start, end=end, freq="M")
            # 코드가 없다면 예시 표로 기본 응답(운영에선 카탈로그로 해상 추천)
            return self.ecos.fetch(stat_code="901Y014", item_codes=["A"], start=start, end=end, freq="M")

        # 4) YF (환율/해외시장)
        if any(k in ql for k in ["usdkrw", "원달러", "환율", "sp500", "nasdaq", "다우", "dow", "vix", "wti", "gold", "us10y", "tnx"]):
            alias = "usdkrw"
            for a in TICKER_ALIAS.keys():
                if a in ql:
                    alias = a
                    break
            if "원달러" in ql or "환율" in ql:
                alias = "usdkrw"
            return self.yf.fetch(alias, start=start, end=end, field="Close")

        # 기본값: 원달러 YF
        return self.yf.fetch("usdkrw", start=start, end=end, field="Close")
