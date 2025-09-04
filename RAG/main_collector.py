# main_collector.py
# 2020-01-01 ~ 2025-09-03 ECOS 정형데이터 + 2025-09-03 환율 관련 뉴스 수집 + YF 일별 데이터 병합
# 저장 경로:
#   - fx_data/wide_20200101_20250903.csv  (ECOS + YF 병합 후 최종본)
#   - news_data/daily_data_20250902.json, news_articles_20250902.csv, news_summary_20250902.csv

import os
import json
from pathlib import Path
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- 프로젝트 내 모듈 ---
from fx_data.specs import load_specs
from fx_data.ecos_client import fetch_auto
from fx_data.transform import to_value_frame, expand_to_daily
from fx_data.merge_wide import merge_wide

from news_data.collector import DataCollector

# --- 외부 ---
try:
    import yfinance as yf
except Exception as e:
    raise ImportError("yfinance가 필요합니다. pip install yfinance 후 다시 시도하세요.") from e

ROOT = Path(__file__).resolve().parent
FX_DIR = ROOT / "fx_data"
NEWS_DIR = ROOT / "news_data"
FX_DIR.mkdir(exist_ok=True, parents=True)
NEWS_DIR.mkdir(exist_ok=True, parents=True)

SPEC_CSV = ROOT / "fx_data" / "series_specs.csv"  # 사양 파일
START = "2020-01-01"
END   = "2025-09-03"                              # 오늘(가정)
NEWS_DAY = dt.date(2025, 9, 2)                   # 전날 뉴스 수집 

# =========================
# YF 기본 티커 매핑(접두어)
# =========================
DEFAULT_YF_TICKERS: Dict[str, str] = {
    # FX
    "KRW=X": "usdkrw",
    "JPY=X": "usdjpy",
    "CNY=X": "usdcny",
    "EURUSD=X": "eurusd",
    # US equity indices
    "^GSPC": "spx",
    "^DJI": "dji",
    "^IXIC": "ndx",
    "^RUT": "rut",
    # Rates (CBOE yields)
    "^TNX": "us10y",   # 10Y
    "^FVX": "us5y",    # 5Y
    "^IRX": "us3m",    # 3M
    # Dollar / Vol
    "DX-Y.NYB": "dxy",
    "^VIX": "vix",
    # Commodities
    "GC=F": "gold",
    "CL=F": "wti",
    "SI=F": "silver",
    "HG=F": "copper",
}


# =========================
# ECOS → 일일 와이드 CSV 생성
# =========================
def collect_ecos_to_daily(spec_csv: Path, start: str, end: str) -> Path:
    """ECOS 사양을 읽어 일단위 와이드 CSV 생성."""
    specs = load_specs(str(spec_csv))

    series_inputs: List[Tuple[str, pd.DataFrame]] = []
    for sp in specs:
        raw = fetch_auto(
            stat_code=sp.stat_code,
            cycle=sp.cycle,
            item_code=sp.item_code,
            start=start,
            end=end
        )
        slim = to_value_frame(raw, sp.cycle)                 # ['date','value']
        daily = expand_to_daily(slim, sp.cycle, start, end)  # 일단위 확장
        name = sp.name if getattr(sp, "name", None) else f"value_{sp.stat_code}"
        series_inputs.append((name, daily))

    wide = merge_wide(series_inputs)
    wide["date"] = pd.to_datetime(wide["date"])

    # 전체 일자 인덱스 → ffill
    idx = pd.date_range(start=start, end=end, freq="D")
    wide = (
        wide.set_index("date")
            .reindex(idx)
            .ffill()
            .rename_axis("date")
            .reset_index()
    )
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")

    out_csv = FX_DIR / f"wide_{start.replace('-','')}_{end.replace('-','')}.csv"
    wide.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


# =========================
# YF 수집 유틸
# =========================
def _fetch_one_ticker_daily(ticker: str, prefix: str, start: str, end: str) -> pd.DataFrame:
    """단일 YF 티커를 start~end 일자 캘린더에 맞춰 일별로 정렬하고 파생을 생성."""
    start_buf = (pd.to_datetime(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_buf   = (pd.to_datetime(end)   + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    tkr = yf.Ticker(ticker)
    hist = tkr.history(start=start_buf, end=end_buf, auto_adjust=False)  # 원시 OHLC
    if hist is None or hist.empty:
        raise RuntimeError(f"empty history for {ticker}")

    hist = hist.reset_index()  # 'Date' 포함
    rename_map = {
        "Date": "date",
        "Open": f"{prefix}_open",
        "High": f"{prefix}_high",
        "Low": f"{prefix}_low",
        "Close": f"{prefix}_close",
        "Adj Close": f"{prefix}_adj_close",
        "Volume": f"{prefix}_volume",
        "Dividends": f"{prefix}_dividends",
        "Stock Splits": f"{prefix}_stock_splits",
    }
    # 존재하는 열만 리네임
    rename_map = {k: v for k, v in rename_map.items() if k in hist.columns}
    hist = hist.rename(columns=rename_map)

    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)

    # 지정 구간 일자 캘린더로 재인덱싱
    idx = pd.date_range(start=start, end=end, freq="D")
    hist = (
        hist.set_index("date")
            .reindex(idx)
            .rename_axis("date")
            .reset_index()
    )

    # 주말/휴일 보정: OHLCV ffill
    base_cols = [f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close", f"{prefix}_volume"]
    for c in base_cols:
        if c in hist.columns:
            hist[c] = hist[c].ffill()

    # 파생
    if {f"{prefix}_high", f"{prefix}_low"}.issubset(hist.columns):
        hist[f"{prefix}_range"] = hist[f"{prefix}_high"] - hist[f"{prefix}_low"]
    if {f"{prefix}_high", f"{prefix}_close"}.issubset(hist.columns):
        hist[f"{prefix}_spread"] = hist[f"{prefix}_high"] - hist[f"{prefix}_close"]
    if f"{prefix}_close" in hist.columns:
        close = hist[f"{prefix}_close"].astype(float)
        hist[f"{prefix}_lr"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    return hist


def collect_yf_bundle(start: str, end: str,
                      tickers: Dict[str, str],
                      skip_errors: bool = True) -> pd.DataFrame:
    """여러 YF 티커 일괄 수집 후 'date' 기준 병합."""
    frames: List[pd.DataFrame] = []
    keys = list(tickers.items())
    for i, (ticker, prefix) in enumerate(keys, 1):
        print(f"[YF {i}/{len(keys)}] {ticker} -> {prefix}  {start}~{end}")
        try:
            df_t = _fetch_one_ticker_daily(ticker, prefix, start, end)
            frames.append(df_t)
        except Exception as e:
            msg = f"yfinance fetch failed: {ticker} ({e})"
            if skip_errors:
                print("  " + msg + "  [skip]")
                continue
            else:
                raise

    if not frames:
        raise RuntimeError("no yahoo frames collected")

    out = frames[0]
    for k in range(1, len(frames)):
        out = out.merge(frames[k], on="date", how="left")

    return out


# =========================
# 뉴스 하루치 수집
# =========================
def collect_news_for(date_obj: dt.date):
    """하루치 환율 관련 뉴스 수집 후 표준 파일명으로 저장."""
    dc = DataCollector()
    parsed, saved = dc.collect_daily_data(
        date_obj,
        max_articles_per_keyword=3
    )

    ds = date_obj.strftime("%Y%m%d")

    # JSON 저장
    (NEWS_DIR / f"daily_data_{ds}.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 뉴스 기사 CSV
    arts = parsed.get("news_articles", [])
    if isinstance(arts, list) and len(arts) > 0:
        pd.DataFrame(arts).to_csv(
            NEWS_DIR / f"news_articles_{ds}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # 요약 CSV
    summ = parsed.get("summary", {})
    if isinstance(summ, dict) and len(summ) > 0:
        pd.DataFrame(
            [{"date": date_obj.strftime("%Y-%m-%d"), **summ}]
        ).to_csv(
            NEWS_DIR / f"news_summary_{ds}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    return parsed


# =========================
# 메인
# =========================
def main():
    print("[1/3] ECOS 수집 시작")
    fx_csv = collect_ecos_to_daily(SPEC_CSV, START, END)
    print(f"정형데이터(ECOS) 저장: {fx_csv}")

    print("[2/3] YF 수집 및 병합 시작")
    # ECOS CSV 읽기 → YF 병합 → 모두 0인 컬럼 제거 → 동일 경로로 저장
    df_ecos = pd.read_csv(fx_csv)

    try:
        yf_df = collect_yf_bundle(START, END, DEFAULT_YF_TICKERS, skip_errors=True)
        df_merged = df_ecos.merge(yf_df, on="date", how="left")

        # YF 파트 ffill(캘린더 정렬상 NaN 보정)
        yf_cols = [c for c in yf_df.columns if c != "date"]
        if yf_cols:
            df_merged[yf_cols] = df_merged[yf_cols].ffill()

        # 모두 0인 컬럼 제거(일부 무의미한 시리즈 방지)
        non_date_cols = [c for c in df_merged.columns if c != "date"]
        drop_cols = [c for c in non_date_cols if (df_merged[c].fillna(0) == 0).all()]
        if drop_cols:
            print(f"[INFO] 모두 0인 {len(drop_cols)}개 컬럼 제거: {drop_cols}")
            df_merged = df_merged.drop(columns=drop_cols)

        # 최종 저장(ECOS+YF)
        df_merged.to_csv(fx_csv, index=False, encoding="utf-8-sig")
        print(f"정형데이터(ECOS+YF) 갱신 저장: {fx_csv}  rows={len(df_merged)}, cols={len(df_merged.columns)}")
    except Exception as e:
        print(f"[WARN] YF 병합에 실패하여 ECOS만 유지합니다: {e}")

    print("[3/3] 뉴스 수집 시작")
    _ = collect_news_for(NEWS_DAY)
    print(f"비정형데이터 저장 경로: {NEWS_DIR}")


if __name__ == "__main__":
    main()
