#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Optional, List
import re
import requests
import pandas as pd

BASE_URL = "https://ecos.bok.or.kr/api"

class EcosError(RuntimeError):
    pass

def _get_api_key() -> str:
    key = os.environ.get("ECOS_API_KEY", "39X0Z0PK0AU0B8JTWX2G").strip()
    if not key:
        raise EcosError("환경변수 ECOS_API_KEY가 설정되어 있지 않습니다.")
    return key

def list_items(stat_code: str, start_row: int = 1, end_row: int = 2000) -> pd.DataFrame:
    key = _get_api_key()
    url = f"{BASE_URL}/StatisticItemList/{key}/json/kr/{start_row}/{end_row}/{stat_code}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = data.get("StatisticItemList", {}).get("row", [])
    return pd.DataFrame(rows)

def fetch_stat(
    stat_code: str,
    cycle: str,
    item_code: str,
    start: str,
    end: str,
    start_row: int = 1,
    end_row:   int = 100000,
    max_retries: int = 3,
    backoff_sec: float = 1.5,
) -> pd.DataFrame:
    key = _get_api_key()
    cycle = cycle.upper()
    if cycle not in {"D", "M", "Q", "A"}:
        raise EcosError(f"알 수 없는 주기(cycle): {cycle}")

    def fmt_date(dt: str) -> str:
        y, m, d = dt.split("-") if "-" in dt else (dt[:4], dt[4:6], dt[6:8] if len(dt) == 8 else "01")
        if cycle == "D":
            return f"{y}{m}{d}"
        elif cycle in {"M", "Q"}:
            return f"{y}{m}"
        elif cycle == "A":
            return f"{y}"
        return dt

    start_fmt = fmt_date(start)
    end_fmt   = fmt_date(end)

    # 주의: ECOS는 다차원 코드일 때 item_code 구간을 "코드1/코드2/..." 형태로 받습니다.
    url = f"{BASE_URL}/StatisticSearch/{key}/json/kr/{start_row}/{end_row}/{stat_code}/{cycle}/{start_fmt}/{end_fmt}/{item_code}"

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            rows = data.get("StatisticSearch", {}).get("row", [])
            return pd.DataFrame(rows)
        except Exception as e:
            last_err = e
            time.sleep(backoff_sec * attempt)
    raise EcosError(f"ECOS 호출 실패: {url}\n원인: {last_err}")

def _compose_item_code_from_row(row: pd.Series, columns: List[str]) -> Optional[str]:
    """
    주어진 items의 단일 행에서 ECOS 호출용 item_code 문자열을 구성.
    우선순위:
      1) ITEM_CODE 단일 열이 있으면 그 값 단독 사용
      2) ITEM_CODE1, ITEM_CODE2, ... 가 있으면 숫자 순으로 '/' 연결
    빈 문자열은 무시.
    """
    # 대소문자 무시 컬럼 접근
    cols_map = {c.lower(): c for c in columns}
    if "item_code" in cols_map:
        val = str(row[cols_map["item_code"]]).strip()
        if val:
            return val

    # ITEM_CODE1, ITEM_CODE2, ... 수집
    code_parts = []
    pat = re.compile(r"^item_code(\d+)$", re.I)
    for lc, orig in sorted(
        ((c.lower(), c) for c in columns if c.lower().startswith("item_code")),
        key=lambda t: (int(pat.match(t[0]).group(1)) if pat.match(t[0]) else 0)
    ):
        if lc == "item_code":  # 이미 위에서 처리
            continue
        val = str(row[orig]).strip()
        if val:
            code_parts.append(val)

    if code_parts:
        return "/".join(code_parts)

    return None  # 구성 실패

def resolve_item_code(stat_code: str, desired: Optional[str]) -> str:
    """
    series_specs.csv의 item_code 값(desired)이 다음 열들 중 어느 것과도 일치하면
    그 행의 ITEM_CODE*를 사용해 호출용 item_code를 우선 결정.
    - 매칭 대상 열: ITEM_CODE, ITEM_CODE1, ITEM_CODE2, ..., ITEM_NAME, ITEM_NAME1, ...
    - desired가 비었거나 매칭이 없으면: 항목 목록의 '첫 번째' 행으로 결정.
    - 최후 수단: '*AA'
    """
    items = list_items(stat_code)
    if items.empty:
        return "*AA"

    # 대소문자 무시 접근
    cols = list(items.columns)
    l2o = {c.lower(): c for c in cols}

    # 후보 열 목록 구성
    code_cols = [c for c in cols if c.lower() == "item_code" or c.lower().startswith("item_code")]
    name_cols = [c for c in cols if c.lower() == "item_name" or c.lower().startswith("item_name")]

    # 1) desired가 주어진 경우: 코드/이름 열에서 매칭되는 첫 행 탐색
    target_row = None
    if desired is not None and str(desired).strip() != "":
        desired_s = str(desired).strip()
        # 정밀(완전 일치) 매칭
        for c in code_cols + name_cols:
            # 문자열 비교를 위해 결측치 방지
            match_idx = items[c].astype(str) == desired_s
            if match_idx.any():
                target_row = items[match_idx].iloc[0]
                break

        # 완전 일치가 없으면 느슨한 부분 일치(특히 ITEM_NAME 계열에 유용)
        if target_row is None:
            desired_u = desired_s.upper()
            for c in name_cols:
                series_upper = items[c].astype(str).str.upper()
                match_idx = series_upper.str.contains(re.escape(desired_u), na=False)
                if match_idx.any():
                    target_row = items[match_idx].iloc[0]
                    break

        if target_row is not None:
            code = _compose_item_code_from_row(target_row, cols)
            if code:
                return code

    # 2) desired가 비었거나(공란) 매칭 실패 시: '첫 번째' 행 사용
    first_row = items.iloc[0]
    code = _compose_item_code_from_row(first_row, cols)
    if code:
        return code

    # 3) 최후 수단
    return "*AA"

def fetch_auto(
    stat_code: str,
    cycle: str,
    item_code: Optional[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    item_code 인자에는 series_specs.csv의 item_code 열 값이 들어온다고 가정.
    - 값이 있으면 resolve_item_code로 해당 값을 우선 매칭하여 ITEM_CODE*를 결정
    - 값이 비어 있으면(혹은 매칭 실패) 항목 목록의 첫 번째 코드 사용
    - 최후 수단으로 '*AA'
    """
    desired = item_code.strip() if item_code else None
    ic = resolve_item_code(stat_code, desired)
    return fetch_stat(stat_code, cycle, ic, start, end)
