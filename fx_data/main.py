#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
사용 예:
  python main.py --specs series_specs.csv --start 2020-01-01 --end 2025-06-30 --out tmp.csv
"""

import argparse
from typing import List, Tuple
import pandas as pd

from specs import load_specs
from ecos_client import fetch_auto
from transform import to_value_frame, expand_to_daily
from merge_wide import merge_wide

# --- 추가: 주기별 날짜 정규화(월/분기/연 → 해당 기간 시작일) ---
def normalize_date_by_cycle(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """
    df: columns ['date', 'value'] 형태를 가정
    cycle: 'D' | 'M' | 'Q' | 'A'
    """
    d = pd.to_datetime(df["date"])
    c = cycle.upper()
    if c == "M":
        df["date"] = d.dt.to_period("M").dt.to_timestamp(how="S")  # 월초
    elif c == "Q":
        df["date"] = d.dt.to_period("Q").dt.to_timestamp(how="S")  # 분기초
    elif c in ("A", "Y"):
        df["date"] = d.dt.to_period("Y").dt.to_timestamp(how="S")  # 연초
    else:  # 'D' 등
        df["date"] = d
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="시계열 사양 CSV 경로(날짜 없음)")
    ap.add_argument("--start", required=True, help="전역 시작일 YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="전역 종료일 YYYY-MM-DD")
    ap.add_argument("--out",   required=True, help="최종 저장 경로(CSV)")
    args = ap.parse_args()

    specs = load_specs(args.specs)

    series_wide_inputs: List[Tuple[str, pd.DataFrame]] = []

    for i, sp in enumerate(specs, 1):
        stat = sp.stat_code
        cyc  = sp.cycle.upper()
        name = sp.name if sp.name else f"value_{stat}"

        print(f"[{i}/{len(specs)}] {stat} ({cyc}) {args.start}~{args.end} 수집 중...")

        # 1) 원자료 수집
        raw = fetch_auto(stat_code=stat, cycle=cyc, item_code=sp.item_code,
                         start=args.start, end=args.end)

        # 2) ['date','value']로 축소
        slim = to_value_frame(raw, cyc)

        # 3) 월/분기/연 값을 기간 시작일로 정규화(월말값 → 월초값 등)
        slim = normalize_date_by_cycle(slim, cyc)

        # 4) 일단위로 확장(각 기간의 상수로 퍼뜨림; 내부 ffill 사용)
        daily = expand_to_daily(slim, cyc, args.start, args.end)

        series_wide_inputs.append((name, daily))

    # 5) 와이드 병합
    wide = merge_wide(series_wide_inputs)

    # 6) 안전장치: 전역 범위 일단위 재인덱싱 후 ffill로 빈칸 보정
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.sort_values("date")
    idx = pd.date_range(start=args.start, end=args.end, freq="D")
    wide = (wide.set_index("date")
                 .reindex(idx)
                 .ffill()
                 .rename_axis("date")
                 .reset_index())

    # 7) 저장
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")
    wide.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"완료: {args.out}")

if __name__ == "__main__":
    main()
