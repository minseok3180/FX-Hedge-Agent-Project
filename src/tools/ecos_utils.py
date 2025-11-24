#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ECOS 데이터 처리 유틸리티 함수들"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal
import pandas as pd

# ============================================================================
# 사양 파일 로드
# ============================================================================

@dataclass
class SeriesSpec:
    """시계열 사양 데이터 클래스"""
    stat_code: str
    cycle: str
    item_code1: Optional[str]
    item_code2: Optional[str]
    name: Optional[str]

def load_specs(path: str) -> List[SeriesSpec]:
    """시계열 사양 CSV 파일을 읽어서 SeriesSpec 리스트로 변환"""
    df = pd.read_csv(path, comment="#").fillna("")
    cols = {c.lower().strip(): c for c in df.columns}
    for req in ("stat_code", "cycle"):
        if req not in cols:
            raise ValueError(f"사양 파일에 '{req}' 컬럼이 필요합니다.")

    specs: List[SeriesSpec] = []
    for _, row in df.iterrows():
        stat_code = str(row[cols["stat_code"]]).strip()
        if not stat_code:
            continue
        cycle = str(row[cols["cycle"]]).strip().upper()
        item_code1 = str(row[cols["item_code1"]]).strip() if "item_code1" in cols else ""
        item_code2 = str(row[cols["item_code2"]]).strip() if "item_code2" in cols else ""
        name = str(row[cols["name"]]).strip() if "name" in cols else ""
        specs.append(SeriesSpec(stat_code, cycle, item_code1, item_code2, name))
    return specs

# ============================================================================
# 데이터 변환
# ============================================================================

Cycle = Literal["D", "M", "Q", "A"]

def parse_ecos_time_to_date(s: str, cycle: Cycle) -> pd.Timestamp:
    """ECOS TIME 값을 pandas Timestamp로 변환"""
    s = str(s)
    if cycle == "D":
        return pd.to_datetime(s, format="%Y%m%d")
    elif cycle == "M":
        return pd.to_datetime(s, format="%Y%m") + pd.offsets.MonthEnd(0)
    elif cycle == "Q":
        if "Q" in s:
            year, q = s.split("Q")
            q = int(q)
            month = q * 3
            return pd.Timestamp(year=int(year), month=month, day=1) + pd.offsets.MonthEnd(0)
        else:
            ts = pd.to_datetime(s, format="%Y%m")
            q_month = ((ts.month - 1) // 3 + 1) * 3
            return pd.Timestamp(year=ts.year, month=q_month, day=1) + pd.offsets.MonthEnd(0)
    elif cycle == "A":
        return pd.Timestamp(year=int(s), month=12, day=31)
    else:
        raise ValueError(f"알 수 없는 주기: {cycle}")

def to_value_frame(df_ecos: pd.DataFrame, cycle: Cycle) -> pd.DataFrame:
    """ECOS 응답을 ['date', 'value'] 형식으로 변환"""
    if df_ecos.empty:
        return pd.DataFrame(columns=["date", "value"])
    cols = {c.lower(): c for c in df_ecos.columns}
    time_col = cols.get("time", "TIME")
    value_col = cols.get("data_value", "DATA_VALUE")

    out = df_ecos[[time_col, value_col]].copy()
    out.columns = ["_time", "_val"]
    out["_time"] = out["_time"].astype(str)
    out["_val"] = pd.to_numeric(out["_val"], errors="coerce")
    out = out.dropna(subset=["_val"])

    out["date"] = out["_time"].map(lambda s: parse_ecos_time_to_date(s, cycle))
    out = out[["date", "_val"]].rename(columns={"_val": "value"})
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out

def expand_to_daily(frame: pd.DataFrame, cycle: Cycle, start: str, end: str) -> pd.DataFrame:
    """월/분기/연 데이터를 일단위로 확장"""
    if frame.empty:
        daily_index = pd.date_range(start=start, end=end, freq="D")
        return pd.DataFrame({"date": daily_index, "value": pd.Series(index=daily_index, dtype="float")})

    frame = frame.sort_values("date")
    daily_index = pd.date_range(start=start, end=end, freq="D")
    daily = pd.DataFrame(index=daily_index).reset_index().rename(columns={"index": "date"})

    if cycle == "D":
        daily = daily.merge(frame, on="date", how="left")
        return daily

    tmp = daily.merge(frame, on="date", how="left").sort_values("date")
    tmp["value"] = tmp["value"].ffill()

    first_rep_day = frame["date"].min()
    tmp.loc[tmp["date"] < first_rep_day, "value"] = pd.NA

    if cycle in ("M", "Q", "A"):
        if cycle == "M":
            rep_days = set(frame["date"].dt.to_period("M").dt.to_timestamp("M").dt.date.tolist())
            last_days = tmp["date"].dt.to_period("M").dt.to_timestamp("M").dt.date
        elif cycle == "Q":
            rep_days = set(frame["date"].dt.to_period("Q").dt.to_timestamp("Q").dt.date.tolist())
            last_days = tmp["date"].dt.to_period("Q").dt.to_timestamp("Q").dt.date
        else:  # A
            rep_days = set(frame["date"].dt.to_period("A").dt.to_timestamp("A").dt.date.tolist())
            last_days = tmp["date"].dt.to_period("A").dt.to_timestamp("A").dt.date

        same_period_has_rep = last_days.map(lambda d: d in rep_days)
        tmp.loc[~same_period_has_rep, "value"] = pd.NA

    return tmp[["date", "value"]]

def normalize_date_by_cycle(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """주기별 날짜 정규화 (월/분기/연 → 해당 기간 시작일)"""
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

# ============================================================================
# 데이터 병합
# ============================================================================

def merge_wide(series_list: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """여러 시계열을 날짜 기준으로 와이드 형식으로 병합"""
    out = None
    for col, df in series_list:
        df2 = df.rename(columns={"value": col})
        if out is None:
            out = df2.copy()
        else:
            out = out.merge(df2, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out

