# src/tools/ingest_hedge_docs.py
# 원격 Qdrant에 PDF / URL 문서를 임베딩해서 업서트하는 스크립트

import sys
from pathlib import Path

# 현재 디렉토리를 sys.path에서 제거하여 로컬 qdrant_client.py와 충돌 방지
script_dir = Path(__file__).parent
if str(script_dir) in sys.path:
    sys.path.remove(str(script_dir))

# 프로젝트 루트를 sys.path에 추가
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import re
import hashlib
import requests
from typing import List, Dict
import time

import fitz  # PyMuPDF
import trafilatura
from bs4 import BeautifulSoup
import tiktoken
from openai import OpenAI
import os
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트에서)
# override=True: 기존 환경 변수가 있어도 .env 파일의 값으로 덮어씀
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"Loaded .env file from: {env_path}")
else:
    # .env 파일이 없어도 환경 변수는 사용 가능
    load_dotenv(override=True)

# Qdrant 클라이언트 import (현재 디렉토리를 제거했으므로 패키지가 정상적으로 import됨)
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# -------------------------------------------------------------------
# 경로 / 설정
# -------------------------------------------------------------------

# 프로젝트 루트 기준으로 데이터 디렉토리 설정
DATA_DIR = project_root / "data" / "hedge_docs"
PDF_DIR = DATA_DIR / "pdfs"
URLS_TXT = DATA_DIR / "urls.txt"

# Qdrant 설정 (환경 변수 또는 .env 파일에서 읽기)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "Hedge_Expert")

# 디버깅: 로드된 Qdrant 설정 확인
print(f"Qdrant settings from .env/environment:")
print(f"  QDRANT_HOST: {QDRANT_HOST}")
print(f"  QDRANT_PORT: {QDRANT_PORT}")
print(f"  QDRANT_COLLECTION: {COLLECTION_NAME}")
if QDRANT_API_KEY:
    print(f"  QDRANT_API_KEY: {'*' * min(len(QDRANT_API_KEY), 10)}...")
else:
    print(f"  QDRANT_API_KEY: (not set)")

# OpenAI 설정 (환경 변수 또는 .env 파일에서 읽기)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# OpenAI 임베딩 모델 (환경 변수에서 읽기, 기본값: text-embedding-3-large)
# 모델별 차원 매핑
EMBEDDING_MODEL_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

EMB_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
EMB_DIM = EMBEDDING_MODEL_DIMS.get(EMB_MODEL, 3072)

if EMB_MODEL not in EMBEDDING_MODEL_DIMS:
    print(f"Warning: Unknown embedding model '{EMB_MODEL}'. Using default dimension 3072.")
    print(f"Supported models: {', '.join(EMBEDDING_MODEL_DIMS.keys())}")

print(f"Using embedding model: {EMB_MODEL} (dimension: {EMB_DIM})")

# 청크 설정
MAX_TOKENS = 512
OVERLAP = 64

if not OPENAI_API_KEY:
    env_file_hint = f" (.env 파일 경로: {env_path})" if env_path.exists() else ""
    raise ValueError(
        f"OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.{env_file_hint}\n"
        "환경 변수를 설정하거나 프로젝트 루트에 .env 파일을 생성하세요.\n"
        ".env 파일 예시:\n"
        "  OPENAI_API_KEY=your-api-key-here\n"
        "  QDRANT_HOST=localhost\n"
        "  QDRANT_PORT=6333"
    )


# -------------------------------------------------------------------
# 유틸 함수들
# -------------------------------------------------------------------

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def read_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path.as_posix())
    pages = [p.get_text("text") for p in doc]
    return clean_text("\n".join(pages))


def fetch_url(url: str) -> str:
    # 1차 시도: trafilatura
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        txt = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_formatting=True,
        ) or ""
        if txt.strip():
            return clean_text(txt)

    # 2차 시도: 직접 HTML 파싱
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return clean_text(soup.get_text(" "))


def chunk_text(
    text: str,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP,
    enc_name: str = "cl100k_base",
) -> List[str]:
    enc = tiktoken.get_encoding(enc_name)
    toks = enc.encode(text)
    i, chunks = 0, []
    while i < len(toks):
        piece = toks[i : i + max_tokens]
        chunks.append(enc.decode(piece))
        i += max_tokens - overlap
    return [c.strip() for c in chunks if c.strip()]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    try:
        resp = client.embeddings.create(model=EMB_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "PermissionDenied" in error_msg or "not allowed" in error_msg.lower():
            raise ValueError(
                f"모델 '{EMB_MODEL}'에 대한 접근 권한이 없습니다.\n"
                f"사용 가능한 모델: {', '.join(EMBEDDING_MODEL_DIMS.keys())}\n"
                f".env 파일에서 OPENAI_EMBEDDING_MODEL을 다른 모델로 변경하세요.\n"
                f"예: OPENAI_EMBEDDING_MODEL=text-embedding-3-small (1536 차원)\n"
                f"또는: OPENAI_EMBEDDING_MODEL=text-embedding-ada-002 (1536 차원)\n"
                f"\n주의: 모델을 변경하면 Qdrant 컬렉션의 벡터 차원도 변경해야 합니다!"
            ) from e
        raise


# -------------------------------------------------------------------
# 메인 로직
# -------------------------------------------------------------------

def main():
    # 전체 시작 시간 기록
    start_time = time.time()
    print("=" * 60)
    print("문서 임베딩 및 Qdrant 업로드 시작")
    print("=" * 60)
    
    # OpenAI, Qdrant 클라이언트 준비
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Qdrant 클라이언트 초기화 (환경 변수 또는 .env 파일에서 읽은 값 사용)
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    if QDRANT_API_KEY:
        qdrant_client = QdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            api_key=QDRANT_API_KEY
        )
    else:
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )
    
    # 1) 컬렉션 생성 (이미 존재해도 에러만 찍고 넘어감)
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(
                size=EMB_DIM,
                distance=qm.Distance.COSINE
            )
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")

    payloads: List[Dict] = []
    chunks_all: List[str] = []

    # 2) PDF 처리
    if PDF_DIR.exists():
        for pdf in sorted(PDF_DIR.glob("*.pdf")):
            text = read_pdf(pdf)
            chunks = chunk_text(text)
            for idx, ch in enumerate(chunks):
                payloads.append(
                    {
                        "source_type": "pdf",
                        "source": pdf.name,
                        "url": None,
                        "title": pdf.stem,
                        "chunk_id": idx,
                        "sha1": sha1(ch),
                        "text": ch,
                    }
                )
            chunks_all.extend(chunks)

    # 3) URL 처리
    if URLS_TXT.exists():
        for url in URLS_TXT.read_text(encoding="utf-8").splitlines():
            u = url.strip()
            if not u:
                continue
            try:
                text = fetch_url(u)
            except Exception:
                # 문제 있는 URL은 그냥 스킵
                continue

            chunks = chunk_text(text)
            for idx, ch in enumerate(chunks):
                payloads.append(
                    {
                        "source_type": "url",
                        "source": u,
                        "url": u,
                        "title": u,
                        "chunk_id": idx,
                        "sha1": sha1(ch),
                        "text": ch,
                    }
                )
            chunks_all.extend(chunks)

    # 4) sha1 기준 중복 제거
    seen = set()
    dedup_chunks: List[str] = []
    dedup_payloads: List[Dict] = []

    for ch, pl in zip(chunks_all, payloads):
        if pl["sha1"] in seen:
            continue
        seen.add(pl["sha1"])
        dedup_chunks.append(ch)
        dedup_payloads.append(pl)

    # 5) 임베딩 생성
    vectors: List[List[float]] = []
    B = 64
    for i in range(0, len(dedup_chunks), B):
        batch = dedup_chunks[i : i + B]
        vectors.extend(embed_texts(client, batch))

    # 6) Qdrant 업서트 (배치 단위로 나눠서 업로드)
    # Qdrant의 페이로드 제한(32MB)을 고려하여 작은 배치로 나눠서 업로드
    UPSERT_BATCH_SIZE = 100  # 한 번에 업로드할 포인트 수 (페이로드 크기에 따라 조정)
    
    total_points = len(vectors)
    print(f"\n총 {total_points}개의 포인트를 배치 크기 {UPSERT_BATCH_SIZE}로 나눠서 업로드합니다...")
    
    upsert_start_time = time.time()
    for batch_idx in range(0, total_points, UPSERT_BATCH_SIZE):
        batch_end = min(batch_idx + UPSERT_BATCH_SIZE, total_points)
        batch_points = []
        
        for idx in range(batch_idx, batch_end):
            batch_points.append(
                qm.PointStruct(
                    id=idx,
                    vector=vectors[idx],
                    payload=dedup_payloads[idx]
                )
            )
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch_points
        )
        
        print(f"  배치 {batch_idx // UPSERT_BATCH_SIZE + 1}/{(total_points + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE}: "
              f"{batch_idx + 1}~{batch_end}번 포인트 업로드 완료 ({batch_end}/{total_points})")
    
    upsert_elapsed = time.time() - upsert_start_time
    print(f"\n총 {total_points}개의 청크를 '{COLLECTION_NAME}' 컬렉션에 업로드 완료 (업로드 소요 시간: {upsert_elapsed:.2f}초)")
    
    # 전체 소요 시간 계산 및 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int((elapsed_time % 1) * 1000)
    
    print("=" * 60)
    print("작업 완료!")
    print(f"총 소요 시간: {hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d} ({elapsed_time:.3f}초)")
    print(f"처리된 청크 수: {len(dedup_chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

