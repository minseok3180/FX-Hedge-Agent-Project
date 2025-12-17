#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
원격 Qdrant에 PDF / URL 문서를 임베딩해서 업서트하는 스크립트.

Note: Qdrant 연결은 vdb.py의 _get_qdrant_client()를 재사용한다.
"""

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
import asyncio
from typing import List, Dict, Any
import time

import fitz  # PyMuPDF
import trafilatura
from bs4 import BeautifulSoup
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

from qdrant_client.models import VectorParams, Distance

# .env 파일 로드 (프로젝트 루트에서)
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"Loaded .env file from: {env_path}")
else:
    load_dotenv(override=True)

# vdb.py의 Qdrant 클라이언트 및 함수 재사용
from src.tools.vdb import _get_qdrant_client, vdb_upsert_points, vdb_create_collection
from src.utils.settings import settings
from src.utils.logger import get_logger

logger = get_logger("ingest-docs")

# -------------------------------------------------------------------
# 경로 / 설정
# -------------------------------------------------------------------

# 프로젝트 루트 기준으로 데이터 디렉토리 설정
DATA_DIR = project_root / "data" / "hedge_docs"
PDF_DIR = DATA_DIR / "pdfs"
URLS_TXT = DATA_DIR / "urls.txt"

# Qdrant 컬렉션명 (환경 변수 또는 기본값)
COLLECTION_NAME = settings.qdrant_collection if hasattr(settings, 'qdrant_collection') else "Hedge_Expert"

# OpenAI 임베딩 모델
EMBEDDING_MODEL_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

EMB_MODEL = settings.openai_embedding_model if hasattr(settings, 'openai_embedding_model') else "text-embedding-3-large"
EMB_DIM = EMBEDDING_MODEL_DIMS.get(EMB_MODEL, 3072)

if EMB_MODEL not in EMBEDDING_MODEL_DIMS:
    logger.warning(f"Unknown embedding model '{EMB_MODEL}'. Using default dimension 3072.")

logger.info(f"Using embedding model: {EMB_MODEL} (dimension: {EMB_DIM})")

# 청크 설정
MAX_TOKENS = 512
OVERLAP = 64


# -------------------------------------------------------------------
# 유틸 함수들
# -------------------------------------------------------------------

def clean_text(s: str) -> str:
    """텍스트 공백 정리"""
    return re.sub(r"\s+", " ", s or "").strip()


def read_pdf(pdf_path: Path) -> str:
    """PDF 파일에서 텍스트 추출"""
    doc = fitz.open(pdf_path.as_posix())
    pages = [p.get_text("text") for p in doc]
    return clean_text("\n".join(pages))


def fetch_url(url: str) -> str:
    """URL에서 텍스트 추출"""
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
    """텍스트를 토큰 기반으로 청크 분할"""
    enc = tiktoken.get_encoding(enc_name)
    toks = enc.encode(text)
    i, chunks = 0, []
    while i < len(toks):
        piece = toks[i : i + max_tokens]
        chunks.append(enc.decode(piece))
        i += max_tokens - overlap
    return [c.strip() for c in chunks if c.strip()]


def sha1(s: str) -> str:
    """SHA1 해시 생성"""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """OpenAI API로 텍스트 임베딩 생성"""
    try:
        resp = client.embeddings.create(model=EMB_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "PermissionDenied" in error_msg or "not allowed" in error_msg.lower():
            raise ValueError(
                f"모델 '{EMB_MODEL}'에 대한 접근 권한이 없습니다.\n"
                f"사용 가능한 모델: {', '.join(EMBEDDING_MODEL_DIMS.keys())}\n"
                f".env 파일에서 OPENAI_EMBEDDING_MODEL을 다른 모델로 변경하세요."
            ) from e
        raise


# -------------------------------------------------------------------
# 메인 로직
# -------------------------------------------------------------------

async def ingest_documents(collection_name: str = COLLECTION_NAME) -> Dict[str, Any]:
    """
    PDF/URL 문서를 임베딩하여 Qdrant에 업로드한다.
    
    vdb.py의 _get_qdrant_client()와 vdb_upsert_points()를 재사용한다.
    
    Args:
        collection_name: Qdrant 컬렉션명
        
    Returns:
        업로드 결과 딕셔너리
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("문서 임베딩 및 Qdrant 업로드 시작")
    logger.info("=" * 60)
    
    # OpenAI 클라이언트 준비
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    # vdb.py의 Qdrant 클라이언트 재사용
    logger.info("Qdrant 클라이언트 연결 중... (vdb.py 재사용)")
    qdrant_client = _get_qdrant_client()
    
    # 1) 컬렉션 생성 (이미 존재해도 에러만 찍고 넘어감)
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name not in collection_names:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMB_DIM,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection '{collection_name}'")
    else:
        logger.info(f"Collection '{collection_name}' already exists")

    payloads: List[Dict] = []
    chunks_all: List[str] = []

    # 2) PDF 처리
    if PDF_DIR.exists():
        for pdf in sorted(PDF_DIR.glob("*.pdf")):
            logger.info(f"Processing PDF: {pdf.name}")
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
                logger.info(f"Processing URL: {u}")
                text = fetch_url(u)
            except Exception as e:
                logger.warning(f"URL 처리 실패 (스킵): {u} - {e}")
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

    logger.info(f"총 청크 수: {len(chunks_all)}, 중복 제거 후: {len(dedup_chunks)}")

    if not dedup_chunks:
        logger.warning("업로드할 청크가 없습니다.")
        return {
            "success": True,
            "collection_name": collection_name,
            "total_chunks": 0,
            "elapsed_seconds": time.time() - start_time
        }

    # 5) 임베딩 생성
    logger.info("임베딩 생성 중...")
    vectors: List[List[float]] = []
    BATCH_SIZE = 64
    for i in range(0, len(dedup_chunks), BATCH_SIZE):
        batch = dedup_chunks[i : i + BATCH_SIZE]
        vectors.extend(embed_texts(openai_client, batch))
        logger.debug(f"  임베딩 진행: {min(i + BATCH_SIZE, len(dedup_chunks))}/{len(dedup_chunks)}")

    # 6) vdb.py의 함수를 사용하여 업서트 (배치 단위)
    UPSERT_BATCH_SIZE = 100
    total_points = len(vectors)
    logger.info(f"\n총 {total_points}개의 포인트를 배치 크기 {UPSERT_BATCH_SIZE}로 나눠서 업로드합니다...")
    
    upsert_start_time = time.time()
    for batch_idx in range(0, total_points, UPSERT_BATCH_SIZE):
        batch_end = min(batch_idx + UPSERT_BATCH_SIZE, total_points)
        batch_points = []
        
        for idx in range(batch_idx, batch_end):
            batch_points.append({
                "id": idx,
                "vector": vectors[idx],
                "payload": dedup_payloads[idx]
            })
        
        # vdb_upsert_points 사용 (async 함수이므로 await)
        await vdb_upsert_points.ainvoke({
            "points": batch_points,
            "collection_name": collection_name
        })
        
        logger.info(
            f"  배치 {batch_idx // UPSERT_BATCH_SIZE + 1}/"
            f"{(total_points + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE}: "
            f"{batch_idx + 1}~{batch_end}번 포인트 업로드 완료 ({batch_end}/{total_points})"
        )
    
    upsert_elapsed = time.time() - upsert_start_time
    total_elapsed = time.time() - start_time
    
    logger.info(f"\n총 {total_points}개의 청크를 '{collection_name}' 컬렉션에 업로드 완료")
    logger.info(f"업로드 소요 시간: {upsert_elapsed:.2f}초")
    logger.info("=" * 60)
    logger.info("작업 완료!")
    logger.info(f"총 소요 시간: {total_elapsed:.2f}초")
    logger.info(f"처리된 청크 수: {len(dedup_chunks)}")
    logger.info("=" * 60)
    
    return {
        "success": True,
        "collection_name": collection_name,
        "total_chunks": len(dedup_chunks),
        "elapsed_seconds": total_elapsed,
        "upsert_elapsed_seconds": upsert_elapsed
    }


def main():
    """CLI 진입점"""
    result = asyncio.run(ingest_documents())
    print(f"\n결과: {result}")


if __name__ == "__main__":
    main()
