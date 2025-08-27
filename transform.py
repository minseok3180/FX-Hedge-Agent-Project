#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Literal
import pandas as pd

Cycle = Literal["D", "M", "Q", "A"]

def parse_ecos_time_to_date(s: str, cycle: Cycle) -> pd.Timestamp:
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
