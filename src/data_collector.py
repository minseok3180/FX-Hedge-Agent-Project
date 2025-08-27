import os
import datetime as dt
import random
from typing import List, Tuple
import pandas as pd


def _today_str():
    return dt.date.today().strftime("%Y-%m-%d")


def collect_fx_timeseries(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    시계열 환율 데이터. yfinance 사용 시 네트워크/API 불가 시 더미 데이터로 폴백.
    결과 컬럼: date, close, ret
    """
    end = end or _today_str()
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df is None or df.empty:
            raise RuntimeError("Empty from yfinance")
        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Close": "close"})
        df["ret"] = df["close"].pct_change().fillna(0.0)
        return df[["date", "close", "ret"]]
    except Exception:
        # 더미 생성
        days = pd.bdate_range(start=start, end=end)
        close = 1200.0
        rows = []
        random.seed(42)
        for d in days:
            ret = random.gauss(0.0, 0.002)
            close *= (1 + ret)
            rows.append({"date": d.date(), "close": close, "ret": ret})
        return pd.DataFrame(rows)


def collect_fx_news(q: str = "달러 원화 환율", n: int = 20) -> List[Tuple[str, str, str]]:
    """
    (date, title, snippet)
    네트워크 불가 시 더미 뉴스 반환.
    """
    try:
        from duckduckgo_search import DDGS

        items: List[Tuple[str, str, str]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=n):
                items.append((_today_str(), r.get("title", ""), r.get("body", "")))
        if not items:
            raise RuntimeError("Empty search")
        return items
    except Exception:
        return [(_today_str(), f"환율 뉴스 {i}", f"원/달러 관련 더미 뉴스 본문 {i}") for i in range(n)]


def save_collected(base_dir: str, ts: pd.DataFrame, news: List[Tuple[str, str, str]]) -> tuple[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    ts_path = os.path.join(base_dir, "fx_timeseries.csv")
    news_path = os.path.join(base_dir, "fx_news.csv")
    ts.to_csv(ts_path, index=False)
    pd.DataFrame(news, columns=["date", "title", "content"]).to_csv(news_path, index=False)
    return ts_path, news_path


