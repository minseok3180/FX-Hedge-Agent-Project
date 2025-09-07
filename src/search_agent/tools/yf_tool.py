#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import pandas as pd
import yfinance as yf

from search_agent.config import TICKER_ALIAS


class YFTool:
    """
    야후 파이낸스(일별) 지표 조회 전용 툴
    - alias_or_ticker: 'usdkrw', '^GSPC', 'USDKRW=X' 등
    - field: 'Close' 기본 (없으면 'Adj Close'로 대체)
    """

    def fetch(
        self,
        alias_or_ticker: str,
        start: str,
        end: str,
        field: str = "Close",
    ) -> pd.DataFrame:
        ticker = TICKER_ALIAS.get(alias_or_ticker.lower(), alias_or_ticker)
        df = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=False, progress=False
        )
        if df.empty:
            return pd.DataFrame()
        col = field if field in df.columns else ("Adj Close" if field == "Close" and "Adj Close" in df.columns else None)
        if col is None:
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = df[col].copy()
        s.index = pd.to_datetime(s.index).date
        return s.to_frame(name=f"{ticker}_{col}".lower())
