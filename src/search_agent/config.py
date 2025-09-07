#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# 프로젝트 루트 추정
ROOT = Path(__file__).resolve().parents[2]

# RAG 디렉토리(네 구조 상 search_agent 안에 존재)
RAG_DIR = Path(__file__).resolve().parent / "RAG"
CHROMA_DIR = ROOT / "RAG/chroma_store"    # 벡터 DB 경로(없으면 생성됨)
DATA_DIR = ROOT / "data"              # 수집 산출물 모아둘 곳(선택)

# ECOS 인증키 환경변수명
ENV_ECOS_KEY = "ECOS_KEY"

# 날짜 파싱 기본 lookback
DEFAULT_LOOKBACK_DAYS = 7

# YF 별칭 → 티커
TICKER_ALIAS = {
    "usdkrw": "USDKRW=X",
    "usdjpy": "JPY=X",
    "usdcny": "CNY=X",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "dowjones": "^DJI",
    "vix": "^VIX",
    "wti": "CL=F",
    "gold": "GC=F",
    "us10y": "^TNX",
}
