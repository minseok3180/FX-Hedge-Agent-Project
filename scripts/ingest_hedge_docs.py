# scripts/ingest_hedge_docs.py
import re, hashlib, requests
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
import trafilatura
from bs4 import BeautifulSoup
import tiktoken
from openai import OpenAI

from src.tools.qdrant_client import QdrantTool

DATA_DIR = Path("/app/data/hedge_docs")
PDF_DIR  = DATA_DIR / "pdfs"
URLS_TXT = DATA_DIR / "urls.txt"

EMB_MODEL = "text-embedding-3-large"
MAX_TOKENS = 512
OVERLAP = 64

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def read_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path.as_posix())
    pages = [p.get_text("text") for p in doc]
    return clean_text("\n".join(pages))

def fetch_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        txt = trafilatura.extract(downloaded, include_tables=True, include_formatting=True) or ""
        if txt.strip():
            return clean_text(txt)
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return clean_text(soup.get_text(" "))

def chunk_text(text: str, max_tokens=MAX_TOKENS, overlap=OVERLAP, enc_name="cl100k_base") -> List[str]:
    enc = tiktoken.get_encoding(enc_name)
    toks = enc.encode(text)
    i, chunks = 0, []
    while i < len(toks):
        piece = toks[i:i+max_tokens]
        chunks.append(enc.decode(piece))
        i += max_tokens - overlap
    return [c.strip() for c in chunks if c.strip()]

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    client = OpenAI()
    q = QdrantTool()
    q.create_collection()

    payloads: List[Dict] = []
    chunks_all: List[str] = []

    # 1) PDFs
    if PDF_DIR.exists():
        for pdf in sorted(PDF_DIR.glob("*.pdf")):
            text = read_pdf(pdf)
            chunks = chunk_text(text)
            for idx, ch in enumerate(chunks):
                payloads.append({
                    "source_type": "pdf",
                    "source": pdf.name,
                    "url": None,
                    "title": pdf.stem,
                    "chunk_id": idx,
                    "sha1": sha1(ch),
                    "text": ch,
                })
            chunks_all.extend(chunks)

    # 2) URLs
    if URLS_TXT.exists():
        for url in URLS_TXT.read_text(encoding="utf-8").splitlines():
            u = url.strip()
            if not u:
                continue
            try:
                text = fetch_url(u)
            except Exception:
                continue
            chunks = chunk_text(text)
            for idx, ch in enumerate(chunks):
                payloads.append({
                    "source_type": "url",
                    "source": u,
                    "url": u,
                    "title": u,
                    "chunk_id": idx,
                    "sha1": sha1(ch),
                    "text": ch,
                })
            chunks_all.extend(chunks)

    # 3) 중복 제거
    seen, dedup_chunks, dedup_payloads = set(), [], []
    for ch, pl in zip(chunks_all, payloads):
        if pl["sha1"] in seen:
            continue
        seen.add(pl["sha1"])
        dedup_chunks.append(ch)
        dedup_payloads.append(pl)

    # 4) 임베딩 + 업서트
    vectors = []
    B = 64
    for i in range(0, len(dedup_chunks), B):
        batch = dedup_chunks[i:i+B]
        vectors.extend(embed_texts(client, batch))
    q.upsert_points(vectors=vectors, payloads=dedup_payloads)
    print(f"Upserted {len(dedup_chunks)} chunks into '{q.collection}'.")

if __name__ == "__main__":
    main()
