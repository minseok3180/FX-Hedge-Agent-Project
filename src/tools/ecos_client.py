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
    """ECOS API 키를 환경변수에서 가져옴"""
    key = os.environ.get("ECOS_API_KEY", "").strip()
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

def resolve_item_code(stat_code: str, desired1: Optional[str] = None, desired2: Optional[str] = None) -> str:
    """
    series_specs.csv의 item_code1, item_code2 값을 사용하여 ECOS 호출용 item_code를 결정.
    - item_code1이 비어있으면: 항목 목록의 '첫 번째' 행의 ITEM_CODE1 사용
    - item_code1이 있으면: 해당 값과 매칭되는 행을 찾아 ITEM_CODE1 사용
    - item_code2가 비어있지 않으면: item_code1/item_code2 형태로 조합
    - item_code2가 비어있으면: item_code1만 사용
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

    # item_code1 결정
    resolved_code1 = None
    
    # item_code1이 주어진 경우: 매칭되는 행 찾기
    if desired1 is not None and str(desired1).strip() != "":
        desired1_s = str(desired1).strip()
        target_row = None
        
        # 정밀(완전 일치) 매칭
        for c in code_cols + name_cols:
            match_idx = items[c].astype(str) == desired1_s
            if match_idx.any():
                target_row = items[match_idx].iloc[0]
                break

        # 완전 일치가 없으면 느슨한 부분 일치
        if target_row is None:
            desired1_u = desired1_s.upper()
            for c in name_cols:
                series_upper = items[c].astype(str).str.upper()
                match_idx = series_upper.str.contains(re.escape(desired1_u), na=False)
                if match_idx.any():
                    target_row = items[match_idx].iloc[0]
                    break

        if target_row is not None:
            # ITEM_CODE1 추출
            if "item_code1" in l2o:
                val = str(target_row[l2o["item_code1"]]).strip()
                if val:
                    resolved_code1 = val
            elif "item_code" in l2o:
                val = str(target_row[l2o["item_code"]]).strip()
                if val:
                    resolved_code1 = val
    
    # item_code1이 비어있거나 매칭 실패 시: 첫 번째 행의 ITEM_CODE1 사용
    if resolved_code1 is None:
        first_row = items.iloc[0]
        if "item_code1" in l2o:
            val = str(first_row[l2o["item_code1"]]).strip()
            if val:
                resolved_code1 = val
        elif "item_code" in l2o:
            val = str(first_row[l2o["item_code"]]).strip()
            if val:
                resolved_code1 = val
    
    # 최후 수단
    if resolved_code1 is None:
        return "*AA"
    
    # item_code2 처리
    if desired2 is not None and str(desired2).strip() != "":
        desired2_s = str(desired2).strip()
        # item_code2는 직접 사용 (매칭 없이)
        return f"{resolved_code1}/{desired2_s}"
    else:
        return resolved_code1

def fetch_auto(
    stat_code: str,
    cycle: str,
    item_code1: Optional[str] = None,
    item_code2: Optional[str] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    series_specs.csv의 item_code1, item_code2 값을 사용하여 ECOS API 호출.
    - item_code1이 비어있으면: 항목 목록의 첫 번째 ITEM_CODE1 사용
    - item_code1이 있으면: 해당 값과 매칭하여 ITEM_CODE1 결정
    - item_code2가 비어있지 않으면: item_code1/item_code2 형태로 조합
    - item_code2가 비어있으면: item_code1만 사용
    """
    # 빈 문자열을 None으로 변환
    desired1 = item_code1.strip() if item_code1 and str(item_code1).strip() else None
    desired2 = item_code2.strip() if item_code2 and str(item_code2).strip() else None
    
    ic = resolve_item_code(stat_code, desired1, desired2)
    return fetch_stat(stat_code, cycle, ic, start, end)
