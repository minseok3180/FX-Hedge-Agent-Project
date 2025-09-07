#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, hashlib, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ==============================
# 공통 유틸
# ==============================
SPLIT_PAT = re.compile(r"(?<=[.!?。！？])\s+|\n+")

def sent_split(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    t = re.sub(r"\s+", " ", text).strip()
    parts = [p.strip() for p in SPLIT_PAT.split(t) if p.strip()]
    return parts if parts else ([t] if t else [])

def chunk_sentences(sents: List[str], min_chars=800, max_chars=1200) -> List[str]:
    chunks, cur = [], ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = f"{cur} {s}"
        else:
            if len(cur) < min_chars and len(s) < max_chars:
                cur = f"{cur} {s}"
            else:
                chunks.append(cur.strip()); cur = s
        if len(cur) > max_chars:
            chunks.append(cur[:max_chars].strip()); cur = cur[max_chars:].strip()
    if cur: chunks.append(cur.strip())
    return [c for c in chunks if c]

def make_id(prefix: str, key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"

def get_collection(client: chromadb.Client, name: str):
    return client.get_or_create_collection(
        name=name,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")
    )


def _sanitize_value(v):
    # 허용: str/int/float/bool/None → 그대로
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    # numpy 스칼라
    if isinstance(v, (np.generic,)):
        return v.item()
    # 리스트/튜플/셋 → JSON 문자열로 저장(또는 ", " join 원하면 아래 변경)
    if isinstance(v, (list, tuple, set)):
        try:
            return json.dumps(list(v), ensure_ascii=False)
        except Exception:
            return ", ".join([str(x) for x in v])
    # dict → JSON 문자열
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    # 그 외 객체 → 문자열로
    return str(v)

def _sanitize_meta(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return {"_meta_raw": _sanitize_value(meta)}
    return {k: _sanitize_value(v) for k, v in meta.items()}


def upsert_batch(col, ids, docs, metas):
    # 1) 길이 정합 체크
    assert len(ids) == len(docs) == len(metas), "ids/docs/metas length mismatch"

    # 2) 배치 내부 중복/결측 정리
    seen = set()
    ids_u, docs_u, metas_u = [], [], []
    dropped_dup = 0
    dropped_empty = 0

    for i, (id_, doc, meta) in enumerate(zip(ids, docs, metas)):
        # 빈 문서 제거
        if doc is None or (isinstance(doc, str) and doc.strip() == ""):
            dropped_empty += 1
            continue
        # 배치 내 중복 제거(첫 건만 유지)
        if id_ in seen:
            dropped_dup += 1
            continue
        seen.add(id_)
        ids_u.append(id_)
        docs_u.append(doc)
        metas_u.append(_sanitize_meta(meta))

    # 3) 아무 것도 없으면 스킵
    if not ids_u:
        print("[upsert_batch] nothing to upsert "
              f"(dropped_empty={dropped_empty}, dropped_dup={dropped_dup})")
        return 0

    # 4) 업서트
    col.upsert(ids=ids_u, documents=docs_u, metadatas=metas_u)
    print(f"[upsert_batch] upserted={len(ids_u)}  "
          f"(dropped_empty={dropped_empty}, dropped_dup={dropped_dup})")
    return len(ids_u)


def yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")

def to_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

# ==============================
# 비정형(뉴스)
# ==============================
def load_daily_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("news_articles", []) or [], data.get("summary", {}) or {}

def load_articles_csv(path: Path):
    return pd.read_csv(path).to_dict(orient="records")

def load_summary_csv(path: Path):
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if not df.empty else {}

def article_to_docs(rec: Dict[str, Any], source_file: str):
    date = str(rec.get("date") or "")
    title = str(rec.get("title") or "")
    body  = str(rec.get("content") or "")
    url   = str(rec.get("url") or "")
    skey  = str(rec.get("search_keyword") or "")
    mkeys = str(rec.get("matched_keywords") or "")
    ptime = str(rec.get("publish_time") or "")

    base_text = f"{title}\n\n{body}".strip() if title else body
    chunks = chunk_sentences(sent_split(base_text), 800, 1200)
    if not chunks: return [], [], []

    doc_id_base = url or f"{date}_{title[:64]}"
    ids = [make_id("news", f"{doc_id_base}#{i}") for i in range(len(chunks))]
    metas = []
    for i in range(len(chunks)):
        metas.append({
            "type": "news",
            "date": date,
            "publish_time": ptime,
            "url": url,
            "search_keyword": skey,
            "matched_keywords": mkeys,
            "source_file": source_file,
            "chunk_id": i
        })
    return ids, chunks, metas

def summary_to_doc(date: str, summary: Dict[str, Any], source_file: str):
    if not summary: return None
    lines = [f"- {k}: {v}" for k, v in summary.items()]
    text  = "일간 뉴스 요약\n" + "\n".join(lines)
    _id   = make_id("news_daily_summary", f"{date}|{source_file}")
    meta  = {"type":"news_summary", "date": date, "source_file": source_file}
    return _id, text, meta

def ingest_news(collection, root: Path, date_str: str) -> int:
    """
    기준일(date_str)의 전일 뉴스를 적재.
    파일명 규칙:
      daily_data_YYYYMMDD.json
      news_articles_YYYYMMDD.csv
      news_summary_YYYYMMDD.csv
    """
    base_date = to_date(date_str)
    prev_date = base_date - timedelta(days=1)
    ds_prev   = yyyymmdd(prev_date)

    json_path = root / f"daily_data_{ds_prev}.json"
    arts_csv  = root / f"news_articles_{ds_prev}.csv"
    summ_csv  = root / f"news_summary_{ds_prev}.csv"

    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    sources = []

    if json_path.exists():
        arts, summ = load_daily_json(json_path)
        records.extend(arts); summary = summ; sources.append(str(json_path))
    if arts_csv.exists():
        records.extend(load_articles_csv(arts_csv)); sources.append(str(arts_csv))
    if (not summary) and summ_csv.exists():
        summary = load_summary_csv(summ_csv); sources.append(str(summ_csv))

    if not records and not summary:
        return 0

    seen, uniq = set(), []
    for r in records:
        key = (str(r.get("url","")), str(r.get("title","")))
        if key in seen: continue
        seen.add(key); uniq.append(r)

    all_ids, all_docs, all_meta = [], [], []
    srcs = ",".join(sources) if sources else ""
    for r in uniq:
        ids, docs, metas = article_to_docs(r, source_file=srcs)
        all_ids += ids; all_docs += docs; all_meta += metas

    if summary:
        sdoc = summary_to_doc(prev_date.strftime("%Y-%m-%d"), summary, source_file=srcs)
        if sdoc:
            sid, stext, smeta = sdoc
            all_ids.append(sid); all_docs.append(stext); all_meta.append(smeta)

    upsert_batch(collection, all_ids, all_docs, all_meta)
    return len(all_ids)

# ==============================
# 정형(하루 요약)
# ==============================
PREFERRED_FEATURES = [
    "usdkrw", "usdkrw_close", "usdkrw(target)", "lr", "lr_ema10",
    "usdjpy_lr", "usdcny_lr", "dxy", "dxy_close", "^TNX_close", "^VIX_close",
    "UUP_close", "lvl_z20", "lvl_rsi14", "lr_vol10"
]

def safe_float(v):
    try: return float(v)
    except: return None

def row_to_structured_text(row: pd.Series, all_cols: List[str]) -> str:
    present = [c for c in PREFERRED_FEATURES if c in all_cols]
    lines = []

    target_candidates = [c for c in ["usdkrw(target)", "usdkrw_close", "usdkrw"] if c in all_cols]
    tgt_val = None
    for c in target_candidates:
        val = safe_float(row.get(c))
        if val is not None:
            tgt_val = (c, val)
            break
    if tgt_val:
        lines.append(f"{tgt_val[0]}={tgt_val[1]:.4f}")

    for c in ["lr", "lr_ema10", "lvl_z20", "lvl_rsi14", "lr_vol10"]:
        if c in all_cols and pd.notna(row.get(c)):
            lines.append(f"{c}={safe_float(row.get(c)):.6f}")

    for c in ["usdjpy_lr","usdcny_lr","UUP_close","^TNX_close","^VIX_close","dxy_close","dxy"]:
        if c in all_cols and pd.notna(row.get(c)):
            val = safe_float(row.get(c))
            if val is not None:
                lines.append(f"{c}={val:.6f}")

    numeric_cols = [c for c in all_cols if c != "date" and np.issubdtype(type(row.get(c)), np.number)]
    extra = []
    for c in numeric_cols:
        v = safe_float(row.get(c))
        if v is None: continue
        extra.append((c, abs(v)))
    extra_sorted = [c for c,_ in sorted(extra, key=lambda x: x[1], reverse=True) if c not in present][:8]
    extras = []
    for c in extra_sorted:
        val = safe_float(row.get(c))
        if val is not None:
            extras.append(f"{c}={val:.6f}")
    if extras:
        lines.append("extras=" + ", ".join(extras))

    content = "정형(시장) 일간 요약\n" + "; ".join(lines)
    return content

def ingest_structured(collection, wide_csv: Path, date_str: str) -> int:
    if not wide_csv.exists(): return 0
    df = pd.read_csv(wide_csv)
    if "date" not in df.columns:
        raise ValueError("wide CSV에 'date' 칼럼이 필요합니다.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    row = df.loc[df["date"] == date_str]
    if row.empty: return 0
    row = row.iloc[0]

    content = row_to_structured_text(row, df.columns.tolist())
    _id = make_id("market", f"{date_str}|{wide_csv.name}")
    meta = {
        "type": "structured",
        "date": date_str,
        "source_csv_path": str(wide_csv),
        "columns": [c for c in df.columns if c != "date"]
    }
    upsert_batch(collection, [_id], [content], [meta])
    return 1

# ==============================
# 예측(모델 & 결과; CSV 버전)
# ==============================
def read_forecast_csv(path: Path) -> Dict[str, Any]:
    """
    예상 컬럼:
      - 필수: date
      - 우선: predicted, predicted_lr
      - 대안: yhat, y_pred, forecast 등 (숫자형 1개 선택)
    반환 형식:
      {
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "horizon_days": int,
        "predicted": [{"date":..., "predicted":..., "predicted_lr":...}, ...]
      }
    """
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("forecast CSV에 'date' 칼럼이 필요합니다.")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 예측값 컬럼 결정
    pred_col = None
    for c in ["predicted", "yhat", "y_pred", "forecast", "yhat_mean"]:
        if c in df.columns:
            pred_col = c
            break
    if pred_col is None:
        # date 외 숫자형 1개 자동 선택
        num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("예측값으로 사용할 숫자형 컬럼을 찾지 못했습니다.")
        pred_col = num_cols[0]

    # 예측 lr 컬럼(선택)
    lr_col = None
    for c in ["predicted_lr", "lr_pred", "yhat_lr"]:
        if c in df.columns:
            lr_col = c
            break

    recs = []
    for _, r in df.iterrows():
        recs.append({
            "date": r["date"],
            "predicted": None if pd.isna(r[pred_col]) else float(r[pred_col]),
            "predicted_lr": None if (lr_col is None or pd.isna(r.get(lr_col))) else float(r.get(lr_col))
        })

    dates = pd.to_datetime(df["date"])
    out = {
        "start_date": dates.min().strftime("%Y-%m-%d"),
        "end_date": dates.max().strftime("%Y-%m-%d"),
        "horizon_days": int(df["date"].nunique()),
        "predicted": recs
    }
    return out

def forecast_batch_doc_from_csv(fj: Dict[str, Any], src: str, model_script: Optional[Path]) -> Tuple[str, str, Dict[str, Any]]:
    start_date = str(fj.get("start_date") or "")
    end_date   = str(fj.get("end_date") or "")
    horizon    = int(fj.get("horizon_days") or 0)

    lines = [
        f"예측범위 {start_date}~{end_date} (h={horizon})",
        "외생채널 수: N/A(CSV)",  # CSV만으로는 알 수 없음
        "정책: N/A(CSV)"
    ]
    text = "USD/KRW 주간 예측 배치 요약\n" + " | ".join([s for s in lines if s])

    meta: Dict[str, Any] = {
        "type": "forecast",
        "start_date": start_date,
        "end_date": end_date,
        "horizon_days": horizon,
        "source_file": src,
        "from": "csv"
    }
    if model_script and model_script.exists():
        code_bytes = model_script.read_bytes()
        code_hash = hashlib.sha256(code_bytes).hexdigest()[:16]
        meta.update({
            "model_script": model_script.name,
            "model_script_sha256_16": code_hash
        })
    _id = make_id("forecast_batch", f"{start_date}|{end_date}|{src}")
    return _id, text, meta

def forecast_day_docs(fj: Dict[str, Any], src: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    per = fj.get("predicted") or []
    ids, docs, metas = [], [], []
    for r in per:
        d = str(r.get("date") or "")
        p = r.get("predicted")
        lr = r.get("predicted_lr")
        text = f"{d} 예측: {p}  (predicted_lr={lr})"
        _id = make_id("forecast_day", f"{d}|{src}")
        meta = {"type":"forecast_day", "date": d, "predicted": p, "predicted_lr": lr, "source_file": src}
        ids.append(_id); docs.append(text); metas.append(meta)
    return ids, docs, metas

def ingest_forecast_csv(collection, root: Path,
                        forecast_csv: Optional[Path],
                        model_script: Optional[Path],
                        per_day: bool,
                        date_str: str) -> int:
    """
    기본 동작:
      - forecast_csv 미지정 시, root에서 predicted_YYYYMMDD.csv를 기준일(date_str)로 탐색
        예) date_str=2025-09-03 -> predicted_20250903.csv
    """
    if forecast_csv is None:
        ds = yyyymmdd(to_date(date_str))
        cand = root / f"predicted_{ds}.csv"
        if cand.exists():
            forecast_csv = cand

    if forecast_csv is None or not forecast_csv.exists():
        return 0

    fj = read_forecast_csv(forecast_csv)
    src = str(forecast_csv)

    # 배치 요약 1건
    bid, btext, bmeta = forecast_batch_doc_from_csv(fj, src, model_script)
    ids, docs, metas = [bid], [btext], [bmeta]

    # (옵션) 하루 단위 문서
    if per_day:
        dids, ddocs, dmetas = forecast_day_docs(fj, src)
        ids += dids; docs += ddocs; metas += dmetas

    upsert_batch(collection, ids, docs, metas)
    return len(ids)

# ==============================
# 메인
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="루트 경로")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD 기준 일자(뉴스=전일, 정형=당일)")
    ap.add_argument("--chroma_dir", required=True, help="Chroma persist 디렉토리")
    ap.add_argument("--collection_news", default="news_unstructured")
    ap.add_argument("--collection_struct", default="market_structured")
    ap.add_argument("--collection_forecast", default="model_forecast")
    ap.add_argument("--wide_glob", default="wide_*.csv", help="정형 CSV 글롭 패턴")
    ap.add_argument("--forecast_csv", default=None, help="예측 CSV 경로(미지정 시 predicted_YYYYMMDD.csv 자동 탐색)")
    ap.add_argument("--model_script", default="rag_weekly_predict.py", help="모델 스크립트 파일명(루트 기준)")
    ap.add_argument("--per_day_forecast", type=int, default=1, help="일자별 예측 문서 생성 여부(1/0)")
    args = ap.parse_args()

    root = Path(args.root)

    # 최신 wide_* 자동 선택
    wide_candidates = sorted(root.glob(args.wide_glob))
    wide_csv = wide_candidates[-1] if wide_candidates else None

    client = chromadb.PersistentClient(path=args.chroma_dir)
    col_news = get_collection(client, args.collection_news)
    col_struct = get_collection(client, args.collection_struct)
    col_fore = get_collection(client, args.collection_forecast)

    # 1) 뉴스: 기준일의 전일
    n_news = ingest_news(col_news, root, args.date)

    # 2) 정형(하루 요약): 기준일 행
    n_struct = 0
    if wide_csv is not None:
        n_struct = ingest_structured(col_struct, wide_csv, args.date)

    # 3) 예측(배치 + 옵션: 일자별): predicted_YYYYMMDD.csv 사용
    fcsv = Path(args.forecast_csv) if args.forecast_csv else None
    mscript = (root / args.model_script) if args.model_script else None
    n_fore = ingest_forecast_csv(col_fore, root, fcsv, mscript, bool(args.per_day_forecast), args.date)

    print(f"news docs upserted: {n_news}")
    print(f"structured docs upserted: {n_struct}")
    print(f"forecast docs upserted: {n_fore}")
    print(f"Chroma path: {args.chroma_dir}")
    print(f"collections: news={args.collection_news}, structured={args.collection_struct}, forecast={args.collection_forecast}")

if __name__ == "__main__":
    main()
