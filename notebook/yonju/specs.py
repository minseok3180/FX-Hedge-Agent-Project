#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd

@dataclass
class SeriesSpec:
    stat_code: str
    cycle: str
    item_code: Optional[str]
    name: Optional[str]

def load_specs(path: str) -> List[SeriesSpec]:
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
        item_code = str(row[cols["item_code"]]).strip() if "item_code" in cols else ""
        name  = str(row[cols["name"]]).strip() if "name" in cols else ""
        specs.append(SeriesSpec(stat_code, cycle, item_code, name))
    return specs
