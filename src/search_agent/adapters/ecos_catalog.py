#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECOS 툴 (검색/조회/정보/배치)
- 위치 권장: search_agent/tools/ecos_tool.py
- 의존: search_agent/RAG/fx_data/{ecos_client, transform, merge_wide}.py

서브커맨드
  - find : 키워드로 series_specs.csv 카탈로그 검색(후보 시리즈 나열)
  - info : 특정 key(ALIAS 또는 STAT.ITEM.CYCLE)의 메타와 샘플 확인
  - get  : 특정 key의 시계열을 ECOS에서 즉시 조회해 표준 출력(JSON/CSV/Parquet)
  - build: (하위호환) specs 전수 수집→일별 확장→와이드 병합→CSV 저장

키 개념
  - key 해석 규칙: ALIAS 우선, 없으면 "STAT_CODE.ITEM_CODE.CYCLE" 문법 파싱
  - 표준 스키마: ['date', 'value'] (get) / 와이드는 ['date', <col...>]
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- 내부 모듈 임포트 (패키지/직접 실행 양쪽 지원) ---
try:
    from search_agent.RAG.fx_data.ecos_client import fetch_auto
    from search_agent.RAG.fx_data.transform import to_value_frame, expand_to_daily
    from search_agent.RAG.fx_data.merge_wide import merge_wide
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from search_agent.RAG.fx_data.ecos_client import fetch_auto
    from search_agent.RAG.fx_data.transform import to_value_frame, expand_to_daily
    from search_agent.RAG.fx_data.merge_wide import merge_wide


# ---------------------------
# 공통 유틸
# ---------------------------

REQ_COLS = {"STAT_CODE", "ITEM_CODE", "CYCLE"}
OPT_COLS = {
    "ALIAS", "NAME_KR", "NAME_EN", "UNIT",
    "START_DATE", "END_DATE",
    # 검색 품질 향상을 위한 선택 메타:
    "TAGS", "DESC"
}

def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def _datestr(x) -> str:
    return pd.to_datetime(x).strftime("%Y-%m-%d")

def _compact(x) -> str:
    return pd.to_datetime(x).strftime("%Y%m%d")

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in str(s))

def _all_zero(series: pd.Series) -> bool:
    s = series.dropna()
    return (not s.empty) and (s == 0).all()

def _human_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return ""
    df = pd.DataFrame(rows)[cols]
    buf = io.StringIO()
    df.to_string(buf, index=False)
    return buf.getvalue()


# ---------------------------
# 스펙 로더 및 카탈로그
# ---------------------------

def load_specs(specs_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(specs_path)
    df = _norm_cols(df)
    df.columns = [c.upper() for c in df.columns]

    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"series_specs.csv 필수 컬럼 누락: {missing}")

    for c in ["STAT_CODE", "ITEM_CODE", "CYCLE"]:
        df[c] = df[c].astype(str).str.strip()

    # 선택 컬럼 기본 정리
    if "ALIAS" in df.columns:
        df["ALIAS"] = df["ALIAS"].astype(str).str.strip()
        df.loc[df["ALIAS"].isin(["", "nan", "None"]), "ALIAS"] = np.nan

    for c in ["NAME_KR", "NAME_EN", "UNIT", "TAGS", "DESC"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # 날짜 컬럼 파싱(전역 from/to가 실제 호출에 사용됨)
    for c in ["START_DATE", "END_DATE"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 검색용 JOIN 키
    df["KEY_FMT"] = df["STAT_CODE"].str.strip() + "." + df["ITEM_CODE"].str.strip() + "." + df["CYCLE"].str.lower()
    return df


def resolve_key(df: pd.DataFrame, key: str) -> pd.Series:
    """
    key: ALIAS 또는 'STAT.ITEM.CYCLE'
    """
    key = str(key).strip()
    # 1) ALIAS 우선
    if "ALIAS" in df.columns:
        hit = df[df["ALIAS"].str.lower() == key.lower()] if pd.notna(key) else pd.DataFrame()
        if len(hit) == 1:
            return hit.iloc[0]
        if len(hit) > 1:
            # 중복 ALIAS는 비권장. 일단 첫 행.
            return hit.iloc[0]
    # 2) STAT.ITEM.CYCLE
    if key.count(".") == 2:
        hit = df[df["KEY_FMT"].str.lower() == key.lower()]
        if len(hit) >= 1:
            return hit.iloc[0]
    raise KeyError(f"키를 해석할 수 없습니다: {key} (ALIAS 또는 'STAT.ITEM.CYCLE' 필요)")


def search_specs(df: pd.DataFrame, query: str, limit: int = 10, fuzzy: bool = True) -> pd.DataFrame:
    """
    간단한 랭킹: 완전일치 > 접두어/정규 > 부분일치
    대상 컬럼: ALIAS, NAME_KR, NAME_EN, STAT_CODE, ITEM_CODE, UNIT, TAGS, DESC, CYCLE
    """
    q = str(query).strip()
    if not q:
        return df.head(limit)

    cols = [c for c in ["ALIAS","NAME_KR","NAME_EN","STAT_CODE","ITEM_CODE","UNIT","TAGS","DESC","CYCLE"] if c in df.columns]
    cand = df.copy()

    # 점수 계산
    scores = np.zeros(len(cand), dtype=float)
    for c in cols:
        s = cand[c].astype(str).str.lower()
        eq = (s == q.lower()).astype(float) * 3.0
        pre = s.str.startswith(q.lower()).astype(float) * 2.0
        contains = s.str.contains(q.lower(), na=False).astype(float) * 1.0
        scores += np.maximum(eq, np.maximum(pre, contains))
    # ALIAS가 있으면 가산점
    if "ALIAS" in cand.columns:
        has_alias = cand["ALIAS"].notna().astype(float) * 0.2
        scores += has_alias

    cand = cand.assign(_score=scores).sort_values(["_score","ALIAS"], ascending=[False, True])
    cand = cand[cand["_score"] > 0]
    return cand.head(limit)


# ---------------------------
# ECOS 단건 수집 → 표준화
# ---------------------------

def fetch_one_series(
    stat_code: str,
    item_code: str,
    cycle: str,
    start_date: str,
    end_date: str,
    lang: str = "kr",
    sleep_sec: float = 0.05,
) -> pd.DataFrame:
    """
    fetch_auto → to_value_frame → expand_to_daily 파이프라인.
    반환 스키마: ['date','value'] (date='YYYY-MM-DD')
    """
    df = fetch_auto(
        stat_code=stat_code,
        item_code=item_code,
        cycle=cycle,
        start=_datestr(start_date),
        end=_datestr(end_date),
        lang=lang,
    )
    time.sleep(sleep_sec)

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "value"])

    df = _norm_cols(df)

    # 1) 표준 값 프레임 변환
    try:
        vdf = to_value_frame(df)
    except Exception:
        date_col = None
        for cand in ["DATE", "TIME", "YYYYMM", "YYYY-MM", "YYYY", "DT"]:
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            date_col = df.columns[0]
        val_col = None
        for cand in ["VALUE", "VAL", "OBS_VALUE", "DATA_VALUE"]:
            if cand in df.columns:
                val_col = cand
                break
        if val_col is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            val_col = num_cols[0] if num_cols else df.columns[-1]
        vdf = df[[date_col, val_col]].copy()
        vdf.columns = ["date", "value"]

    vdf["date"] = pd.to_datetime(vdf["date"], errors="coerce")
    vdf = vdf.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 2) 일자 확장
    try:
        edf = expand_to_daily(vdf, method="ffill")
    except Exception:
        edf = vdf.set_index("date").asfreq("D").ffill().reset_index()
        edf.columns = ["date", "value"]

    edf["date"] = edf["date"].dt.strftime("%Y-%m-%d")
    return edf[["date", "value"]]


# ---------------------------
# 서브커맨드 구현
# ---------------------------

def cmd_find(args) -> int:
    df = load_specs(args.specs)
    out = search_specs(df, args.q, limit=args.limit, fuzzy=bool(args.fuzzy))
    rows = []
    for _, r in out.iterrows():
        rows.append({
            "rank": len(rows)+1,
            "key": r["ALIAS"] if pd.notna(r.get("ALIAS", np.nan)) else r["KEY_FMT"],
            "name_kr": r.get("NAME_KR", ""),
            "name_en": r.get("NAME_EN", ""),
            "cycle": r.get("CYCLE", ""),
            "unit": r.get("UNIT", ""),
            "stat": r["STAT_CODE"], "item": r["ITEM_CODE"],
        })
    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        table = _human_table(rows, ["rank","key","name_kr","cycle","unit","stat","item"])
        print(table or "(검색 결과 없음)")
    return 0


def cmd_info(args) -> int:
    df = load_specs(args.specs)
    row = resolve_key(df, args.key)
    card = {
        "key": row["ALIAS"] if pd.notna(row.get("ALIAS", np.nan)) else row["KEY_FMT"],
        "stat_code": row["STAT_CODE"],
        "item_code": row["ITEM_CODE"],
        "cycle": row["CYCLE"],
        "name_kr": row.get("NAME_KR", ""),
        "name_en": row.get("NAME_EN", ""),
        "unit": row.get("UNIT", ""),
        "tags": row.get("TAGS", ""),
        "desc": row.get("DESC", ""),
        "start_date": str(row.get("START_DATE", "")) if pd.notna(row.get("START_DATE", np.nan)) else "",
        "end_date": str(row.get("END_DATE", "")) if pd.notna(row.get("END_DATE", np.nan)) else "",
    }
    print(json.dumps(card, ensure_ascii=False, indent=2))

    if args.sample:
        s = fetch_one_series(
            stat_code=row["STAT_CODE"], item_code=row["ITEM_CODE"], cycle=row["CYCLE"],
            start_date=args.sample_from, end_date=args.sample_to, lang=args.lang, sleep_sec=args.sleep
        )
        print("\n# sample")
        print(s.head(args.sample).to_string(index=False))
    return 0


def _emit_frame(df: pd.DataFrame, fmt: str) -> None:
    if fmt == "json":
        print(df.to_json(orient="records", force_ascii=False))
    elif fmt == "csv":
        sys.stdout.write(df.to_csv(index=False))
    elif fmt == "parquet":
        # 표준 출력으로 바이너리 덤프는 비권장. 경고 후 CSV로 대체.
        sys.stderr.write("[warn] parquet는 파일 출력에 적합합니다. stdout은 CSV로 대체합니다.\n")
        sys.stdout.write(df.to_csv(index=False))
    else:
        # 기본 표 포맷
        buf = io.StringIO()
        df.to_string(buf, index=False, max_rows=20)
        print(buf.getvalue())


def cmd_get(args) -> int:
    df = load_specs(args.specs)
    row = resolve_key(df, args.key)

    series_df = fetch_one_series(
        stat_code=row["STAT_CODE"], item_code=row["ITEM_CODE"], cycle=row["CYCLE"],
        start_date=args.from_date, end_date=args.to_date, lang=args.lang, sleep_sec=args.sleep
    )

    if not bool(args.daily):
        # 원주기 반환을 원하면 일별 확장 이전의 관측일만 추리기.
        # expand_to_daily를 이미 거쳤으므로, 원본 날짜만 남기려면 drop_duplicates 로 근사.
        series_df["dt"] = pd.to_datetime(series_df["date"])
        # 원주기 포인트 감지를 위해 1차 차분이 NaN이 되는 지점만 추출하는 간단 근사
        orig_mask = series_df["value"].ne(series_df["value"].shift(1))
        out_df = series_df.loc[orig_mask, ["date","value"]].reset_index(drop=True)
        series_df.drop(columns=["dt"], inplace=True)
    else:
        out_df = series_df

    # 파일 저장 여부와 관계없이 stdout 출력
    _emit_frame(out_df, args.format)

    # 필요시 저장
    if bool(args.save):
        out_dir = Path(args.out_dir) if args.out_dir else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        key_name = row["ALIAS"] if pd.notna(row.get("ALIAS", np.nan)) else row["KEY_FMT"]
        key_safe = _safe_name(key_name)
        f, t = _compact(args.from_date), _compact(args.to_date)
        if args.format == "json":
            path = out_dir / f"{key_safe}_{f}_{t}.json"
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(json.loads(out_df.to_json(orient="records", force_ascii=False)), fp, ensure_ascii=False, indent=2)
        else:
            path = out_dir / f"{key_safe}_{f}_{t}.csv"
            out_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[saved] {path}", file=sys.stderr)

    return 0


def cmd_build(args) -> int:
    """
    기존 배치 파이프라인(하위호환): 전수 수집 → 와이드 CSV 저장 + 리포트 JSON 저장
    """
    report = run_build(
        specs_path=args.specs,
        out_dir=args.out_dir,
        from_date=args.from_date,
        to_date=args.to_date,
        lang=args.lang,
        save_raw=bool(args.save_raw),
        drop_all_zero=bool(args.drop_all_zero),
        sleep=args.sleep,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[done] wide_csv: {report['wide_csv']}", file=sys.stderr)
    return 0


# ---------------------------
# 배치(빌드) 실행기 (하위호환)
# ---------------------------

def run_build(
    specs_path: str,
    out_dir: str,
    from_date: str,
    to_date: str,
    lang: str = "kr",
    save_raw: bool = True,
    drop_all_zero: bool = True,
    sleep: float = 0.05,
) -> Dict[str, Any]:
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw_dir = Path(out_dir) / "raw_ecos"
    if save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs(specs_path)

    collected: Dict[str, pd.DataFrame] = {}
    logs: List[Dict[str, Any]] = []

    for _, row in specs.iterrows():
        stat = str(row["STAT_CODE"])
        item = str(row["ITEM_CODE"])
        cyc  = str(row["CYCLE"]).lower()
        alias = str(row["ALIAS"]).strip() if "ALIAS" in row and pd.notna(row["ALIAS"]) else ""
        key = alias if alias else f"{stat}.{item}.{cyc}"
        key_safe = _safe_name(key)

        try:
            s = fetch_one_series(
                stat_code=stat, item_code=item, cycle=cyc,
                start_date=from_date, end_date=to_date, lang=lang, sleep_sec=sleep
            )
            n = len(s)
            status = "ok" if n > 0 else "empty"

            if n > 0:
                colname = key_safe
                s = s.rename(columns={"value": colname})
                if save_raw:
                    s.to_csv(raw_dir / f"{key_safe}.csv", index=False, encoding="utf-8-sig")
                collected[colname] = s[["date", colname]]

            logs.append({
                "key": key, "stat_code": stat, "item_code": item, "cycle": cyc,
                "rows": n, "status": status, "note": ""
            })

        except Exception as e:
            logs.append({
                "key": key, "stat_code": stat, "item_code": item, "cycle": cyc,
                "rows": 0, "status": "error", "note": str(e)
            })

    if not collected:
        raise RuntimeError("수집된 시리즈가 없습니다. specs, 기간, ECOS 키/권한을 확인하세요.")

    # 와이드 병합
    try:
        wide = merge_wide(list(collected.values()), on="date")
    except Exception:
        wide = None
        for _, df in collected.items():
            if wide is None:
                wide = df.copy()
            else:
                wide = pd.merge(wide, df, on="date", how="outer")
        wide = wide.sort_values("date").reset_index(drop=True)

    # 모두 0 칼럼 제거
    dropped_cols: List[str] = []
    if drop_all_zero:
        keep = ["date"]
        for c in wide.columns:
            if c == "date":
                continue
            if _all_zero(wide[c]):
                dropped_cols.append(c)
            else:
                keep.append(c)
        wide = wide[keep]

    f, t = _compact(from_date), _compact(to_date)
    wide_path = Path(out_dir) / f"wide_{f}_{t}.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    report = {
        "specs_path": str(specs_path),
        "out_dir": out_dir,
        "from_date": _datestr(from_date),
        "to_date": _datestr(to_date),
        "lang": lang,
        "save_raw": bool(save_raw),
        "drop_all_zero": bool(drop_all_zero),
        "dropped_all_zero_columns": dropped_cols,
        "n_series_requested": int(len(specs)),
        "n_series_collected": int(sum(1 for x in logs if x["status"] == "ok")),
        "wide_csv": str(wide_path),
        "created_at": _now_stamp(),
        "log": logs,
    }

    rep_path = Path(out_dir) / f"ecos_report_{_now_stamp()}.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


# ---------------------------
# CLI
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ECOS tool (find/get/info/build)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # find
    pf = sub.add_parser("find", help="키워드로 series_specs 카탈로그 검색")
    pf.add_argument("--specs", required=True, help="series_specs.csv 경로")
    pf.add_argument("--q", required=True, help="검색어(예: '수출 미국 M')")
    pf.add_argument("--limit", type=int, default=10)
    pf.add_argument("--fuzzy", type=int, default=1)
    pf.add_argument("--format", choices=["table", "json"], default="table")
    pf.set_defaults(func=cmd_find)

    # info
    pi = sub.add_parser("info", help="지표 키의 메타/설명/기간 및 샘플 확인")
    pi.add_argument("--specs", required=True)
    pi.add_argument("--key", required=True, help="ALIAS 또는 STAT.ITEM.CYCLE")
    pi.add_argument("--lang", choices=["kr", "en"], default="kr")
    pi.add_argument("--sample", type=int, default=0, help="샘플 행수(0이면 미표시)")
    pi.add_argument("--sample_from", default="2020-01-01")
    pi.add_argument("--sample_to", default=dt.date.today().strftime("%Y-%m-%d"))
    pi.add_argument("--sleep", type=float, default=0.05)
    pi.set_defaults(func=cmd_info)

    # get
    pg = sub.add_parser("get", help="ECOS에서 특정 지표를 즉시 조회하여 표준 출력")
    pg.add_argument("--specs", required=True)
    pg.add_argument("--key", required=True, help="ALIAS 또는 STAT.ITEM.CYCLE")
    pg.add_argument("--from_date", required=True)
    pg.add_argument("--to_date", required=True)
    pg.add_argument("--lang", choices=["kr", "en"], default="kr")
    pg.add_argument("--daily", type=int, default=1, help="일별 확장 여부(1/0)")
    pg.add_argument("--format", choices=["table","json","csv","parquet"], default="json")
    pg.add_argument("--save", type=int, default=0, help="파일 저장 여부(1/0)")
    pg.add_argument("--out_dir", default="", help="--save=1인 경우 저장 디렉터리")
    pg.add_argument("--sleep", type=float, default=0.05)
    pg.set_defaults(func=cmd_get)

    # build (하위호환)
    pb = sub.add_parser("build", help="스펙 전수 수집→와이드 병합 CSV 저장 (하위호환)")
    pb.add_argument("--specs", required=True)
    pb.add_argument("--out_dir", required=True)
    pb.add_argument("--from_date", required=True)
    pb.add_argument("--to_date", required=True)
    pb.add_argument("--lang", choices=["kr","en"], default="kr")
    pb.add_argument("--save_raw", type=int, default=1)
    pb.add_argument("--drop_all_zero", type=int, default=1)
    pb.add_argument("--sleep", type=float, default=0.05)
    pb.set_defaults(func=cmd_build)

    return p


def main():
    p = build_arg_parser()
    args = p.parse_args()
    return_code = args.func(args)
    sys.exit(return_code)



if __name__ == "__main__":
    main()
