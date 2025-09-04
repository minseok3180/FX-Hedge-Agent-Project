# rag_store.py
# 사용:
#   python rag_store.py --jsonl build/rag_docs_20250825.jsonl

import argparse, hashlib, json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

def make_id(hint: str, doc: str) -> str:
    return hashlib.md5(f"{hint}-{doc[:200]}".encode("utf-8")).hexdigest()[:24]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.db, settings=Settings(allow_reset=False))
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.model)

    cols = {
        "market": client.get_or_create_collection("market", embedding_function=emb, metadata={"hnsw:space":"cosine"}),
        "news": client.get_or_create_collection("news", embedding_function=emb, metadata={"hnsw:space":"cosine"}),
        "own_hedge": client.get_or_create_collection("own_hedge", embedding_function=emb, metadata={"hnsw:space":"cosine"}),
        "history": client.get_or_create_collection("history", embedding_function=emb, metadata={"hnsw:space":"cosine"}),
    }

    ids, docs, metas, colnames = [], [], [], []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cid = make_id(d.get("id_hint",""), d["document"])
            ids.append(cid); docs.append(d["document"]); metas.append(d["metadata"]); colnames.append(d["collection"])

    # 컬렉션별로 나눠 업서트
    from collections import defaultdict
    bucket = defaultdict(list)
    for cid, doc, meta, cname in zip(ids, docs, metas, colnames):
        bucket[cname].append((cid, doc, meta))
    for cname, rows in bucket.items():
        col = cols[cname]
        col.upsert(
            ids=[r[0] for r in rows],
            documents=[r[1] for r in rows],
            metadatas=[r[2] for r in rows],
        )
        print(f"upsert: {cname} -> {len(rows)}건")

if __name__ == "__main__":
    main()
