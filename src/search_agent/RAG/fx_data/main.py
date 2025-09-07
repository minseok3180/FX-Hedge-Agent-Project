#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
사용 예:
  python main.py --specs series_specs.csv --start 2020-01-01 --end 2025-03 --out fx_data/wide_20200101_20250903.csv
옵션:
  --no_yf             : 야후 병합 비활성화
  --yf_skip_errors    : 일부 티커 실패해도 계속 진행
  --yf_extra T1,T2    : 기본 리스트에 추가 티커를 쉼표로 더함 (예: --yf_extra AAPL,MSFT)
"""

import argparse
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from specs import load_specs
from ecos_client import fetch_auto
from transform import to_value_frame, expand_to_daily
from merge_wide import merge_wide

# 기본 야후 티커와 접두어(prefix) 매핑
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
    "^TNX": "us10y",
    "^FVX": "us5y",
    "^IRX": "us3m",
    # Dollar / Vol
    "DX-Y.NYB": "dxy",
    "^VIX": "vix",
    # Commodities
    "GC=F": "gold",
    "CL=F": "wti",
    "SI=F": "silver",
    "HG=F": "copper",
}

# ---------------------- 주기별 날짜 정규화 ----------------------
def normalize_date_by_cycle(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    c = cycle.upper()
    if c == "M":
        df["date"] = d.dt.to_period("M").dt.to_timestamp(how="S")
    elif c == "Q":
        df["date"] = d.dt.to_period("Q").dt.to_timestamp(how="S")
    elif c in ("A", "Y"):
        df["date"] = d.dt.to_period("Y").dt.to_timestamp(how="S")
    else:
        df["date"] = d
    return df

# ---------------------- 단일 티커 수집 ----------------------
def fetch_one_ticker_daily(ticker: str, prefix: str, start: str, end: str) -> pd.DataFrame:
    # 히스토리 버퍼
    start_buf = (pd.to_datetime(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_buf   = (pd.to_datetime(end)   + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    tkr = yf.Ticker(ticker)
    hist = tkr.history(start=start_buf, end=end_buf, auto_adjust=False)  # 원시 OHLC
    if hist.empty:
        raise RuntimeError(f"empty history for {ticker}")

    # 정규화
    hist = hist.reset_index()
    # 열 이름 표준화 (야후는 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' 기본)
    rename_map = {
        "Date": "date", "Open": f"{prefix}_open", "High": f"{prefix}_high",
        "Low": f"{prefix}_low", "Close": f"{prefix}_close", "Volume": f"{prefix}_volume",
        "Dividends": f"{prefix}_dividends", "Stock Splits": f"{prefix}_stock_splits",
        "Adj Close": f"{prefix}_adj_close",
    }
    for c in list(rename_map):
        if c not in hist.columns:
            rename_map.pop(c, None)
    hist = hist.rename(columns=rename_map)
    # 타임존 제거
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)

    # 목표 범위 일자 캘린더로 재인덱싱
    idx = pd.date_range(start=start, end=end, freq="D")
    hist = (hist.set_index("date").reindex(idx).rename_axis("date").reset_index())

    # ffill로 주말/휴일 보정
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

    # 날짜 문자열화
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    return hist

# ---------------------- 여러 티커 일괄 수집 ----------------------
def fetch_yf_bundle(start: str, end: str, tickers: Dict[str, str], skip_errors: bool = True) -> pd.DataFrame:
    frames = []
    for i, (ticker, prefix) in enumerate(tickers.items(), 1):
        print(f"[YF {i}/{len(tickers)}] {ticker} -> {prefix}  {start}~{end}")
        try:
            df_t = fetch_one_ticker_daily(ticker, prefix, start, end)
            frames.append(df_t)
        except Exception as e:
            msg = f"yfinance fetch failed: {ticker} ({e})"
            if skip_errors:
                print("  " + msg + "  [skip]")
                continue
            else:
                raise RuntimeError(msg)
    if not frames:
        raise RuntimeError("no yahoo frames collected")
    # 날짜 기준 병합 (순차적으로 merge)
    out = frames[0]
    for k in range(1, len(frames)):
        out = out.merge(frames[k], on="date", how="left")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="series_specs.csv 경로")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="최종 CSV 경로")
    ap.add_argument("--no_yf", action="store_true", help="야후 병합 비활성화")
    ap.add_argument("--yf_skip_errors", action="store_true", help="일부 티커 실패시 계속")
    ap.add_argument("--yf_extra", type=str, default="", help="쉼표로 추가 티커 (예: AAPL,MSFT)")
    args = ap.parse_args()

    # specs 경로 보정(스크립트 기준)
    script_dir = Path(__file__).resolve().parent
    specs_path = Path(args.specs)
    if not specs_path.exists():
        alt = (script_dir / args.specs).resolve()
        if alt.exists():
            specs_path = alt
        else:
            raise FileNotFoundError(f"specs not found: {args.specs}")

    # ---------------- ECOS 수집 → 일일 와이드 ----------------
    specs = load_specs(str(specs_path))
    series_wide_inputs: List[Tuple[str, pd.DataFrame]] = []

    for i, sp in enumerate(specs, 1):
        stat = sp.stat_code
        cyc  = sp.cycle.upper()
        name = sp.name if sp.name else f"value_{stat}"

        print(f"[ECOS {i}/{len(specs)}] {stat} ({cyc}) {args.start}~{args.end}")
        raw = fetch_auto(stat_code=stat, cycle=cyc, item_code=sp.item_code,
                         start=args.start, end=args.end)
        slim = to_value_frame(raw, cyc)
        slim = normalize_date_by_cycle(slim, cyc)
        daily = expand_to_daily(slim, cyc, args.start, args.end)
        series_wide_inputs.append((name, daily))

    # ECOS 와이드 병합
    wide = merge_wide(series_wide_inputs)
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.sort_values("date")
    idx = pd.date_range(start=args.start, end=args.end, freq="D")
    wide = (
        wide.set_index("date")
            .reindex(idx)
            .ffill()
            .rename_axis("date")
            .reset_index()
    )
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")

    # ---------------- 야후 번들 병합 ----------------
    if not args.no_yf:
        # 기본 리스트 + 추가 티커
        yf_map = DEFAULT_YF_TICKERS.copy()
        if args.yf_extra:
            # 접두어는 자동 생성(소문자, 기호 제거)
            extra = [t.strip() for t in args.yf_extra.split(",") if t.strip()]
            for t in extra:
                prefix = (
                    t.lower()
                     .replace("=", "")
                     .replace("^", "")
                     .replace("-", "")
                     .replace(".", "")
                     .replace("/", "")
                )
                if prefix in yf_map.values():
                    prefix = f"{prefix}2"
                yf_map[t] = prefix

        print(f"[YF] total tickers: {len(yf_map)}")
        yf_df = fetch_yf_bundle(args.start, args.end, yf_map, skip_errors=args.yf_skip_errors)
        # 병합 및 최소 보정
        wide = wide.merge(yf_df, on="date", how="left")
        yf_cols = [c for c in yf_df.columns if c != "date"]
        if yf_cols:
            wide[yf_cols] = wide[yf_cols].ffill()


    # ---------------- 저장 ----------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)


    # === 모두 0인 컬럼 제거 ===
    non_date_cols = [c for c in wide.columns if c != "date"]
    drop_cols = [c for c in non_date_cols if (wide[c].fillna(0) == 0).all()]
    if drop_cols:
        print(f"[INFO] 모두 0인 {len(drop_cols)}개 컬럼 제거: {drop_cols}")
        wide = wide.drop(columns=drop_cols)

    wide.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"완료: {args.out}  rows={len(wide)}, cols={len(wide.columns)}")


if __name__ == "__main__":
    main()
