#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
import pandas as pd

def merge_wide(series_list: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    out = None
    for col, df in series_list:
        df2 = df.rename(columns={"value": col})
        if out is None:
            out = df2.copy()
        else:
            out = out.merge(df2, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out
