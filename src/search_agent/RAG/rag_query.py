#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
import chromadb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma_store")
    ap.add_argument("--collection", required=True,
                    choices=["news_unstructured","market_structured","model_forecast"])
    ap.add_argument("--query", default=None, help="텍스트 질의(임베딩 검색)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--where", default=None, help='메타 필터(JSON). 예: {"date":"2025-09-03"}')
    ap.add_argument("--ids", nargs="*", help="특정 ID 직접 조회")
    ap.add_argument("--limit", type=int, default=None, help="ids 없이 단순 get 시 개수")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.db)
    col = client.get_collection(args.collection)

    where = json.loads(args.where) if args.where else None

    if args.ids:
        out = col.get(ids=args.ids)
    elif args.query:
        out = col.query(query_texts=[args.query], n_results=args.n, where=where)
    else:
        out = col.get(where=where, limit=args.limit or args.n)

    # 보기 좋게 출력
    def short(x, k=120):
        if not isinstance(x, str): return str(x)
        return x if len(x)<=k else x[:k]+"..."
    if "documents" in out:
        rows = zip(out.get("ids", []),
                   out.get("documents", []),
                   out.get("metadatas", []),
                   out.get("distances", [[]])[0] if "distances" in out else [])
        for rid, doc, meta, dist in rows:
            print("="*80)
            print("ID:", rid)
            print("META:", meta)
            if dist != []: print("DIST:", dist)
            print("DOC:", short(doc))
    else:
        print(out)

if __name__ == "__main__":
    main()
