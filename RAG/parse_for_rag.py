# parse_for_rag.py
# 입력 파일들을 RAG 업서트에 바로 쓸 수 있는 표준 dict list로 파싱
# 출력: ./build/rag_docs_{stamp}.jsonl

from pathlib import Path
import json, pandas as pd

def iter_market_docs(wide_csv: Path):
    df = pd.read_csv(wide_csv)
    for _, r in df.iterrows():
        date = str(r["date"])
        payload = {k: v for k, v in r.items() if k != "date"}
        doc = f"{date}: " + ", ".join([f"{k}={payload[k]}" for k in payload.keys()])
        yield {
            "collection": "market",
            "id_hint": f"market-{date}",
            "document": doc,
            "metadata": {"type":"timeseries","source": wide_csv.name, "date": date}
        }

def iter_news_docs(news_json: Path=None, news_csv: Path=None, news_summary: Path=None):
    if news_json and news_json.exists():
        obj = json.loads(news_json.read_text(encoding="utf-8"))
        for art in obj.get("news_articles", []):
            base = f"{art.get('title','')}\n\n{art.get('content','')}"
            yield {
                "collection":"news",
                "id_hint": f"news-{art.get('date',obj.get('collection_date',''))}-{hash(base)}",
                "document": base,
                "metadata":{
                    "type":"news","source":news_json.name,"date":art.get("date", obj.get("collection_date","")),
                    "url": art.get("url",""), "title": art.get("title","")[:180],
                    "kws": art.get("matched_keywords","")
                }
            }
    if news_csv and news_csv.exists():
        df = pd.read_csv(news_csv)
        for _, r in df.iterrows():
            base = f"{r.get('title','')}\n\n{r.get('content','')}"
            yield {
                "collection":"news",
                "id_hint": f"news-{r.get('date','')}-{hash(base)}",
                "document": base,
                "metadata":{
                    "type":"news","source":news_csv.name,"date": r.get("date",""),
                    "url": r.get("url",""), "title": str(r.get("title",""))[:180],
                    "kws": r.get("matched_keywords","")
                }
            }
    if news_summary and news_summary.exists():
        df = pd.read_csv(news_summary)
        if len(df):
            rec = df.iloc[0].to_dict()
            yield {
                "collection":"news",
                "id_hint": f"news-summary-{rec.get('date','')}",
                "document": json.dumps(rec, ensure_ascii=False),
                "metadata": {"type":"news_summary","source": news_summary.name, "date": rec.get("date","")}
            }

def iter_pred_docs(pred_csv: Path):
    if not pred_csv.exists():
        return
    df = pd.read_csv(pred_csv)
    for _, r in df.iterrows():
        date = str(r["date"])
        txt = f"Date {date} | predicted={r.get('predicted')}, predicted_lr={r.get('predicted_lr')}"
        yield {
            "collection":"own_hedge",
            "id_hint": f"pred-{date}",
            "document": txt,
            "metadata": {"type":"prediction","source": pred_csv.name, "date": date}
        }

def build_docs(fx_csv, news_ds="20250825"):
    wide_csv = Path(fx_csv)
    njson = Path(f"news_data/daily_data_{news_ds}.json")
    ncsv  = Path(f"news_data/news_articles_{news_ds}.csv")
    nsum  = Path(f"news_data/news_summary_{news_ds}.csv")
    pcsv  = Path("prediction_model/timexer_results.csv")

    docs = []
    docs += list(iter_market_docs(wide_csv))
    docs += list(iter_news_docs(njson, ncsv, nsum))
    docs += list(iter_pred_docs(pcsv))

    out_dir = Path("build"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"rag_docs_{news_ds}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"작성: {out_path} (docs={len(docs)})")
    return out_path

if __name__ == "__main__":
    build_docs("fx_data/wide_20200101_20250825.csv", "20250825")
