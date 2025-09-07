#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECOS 카탈로그 빌더 CLI
- 사용법:
    export PYTHONPATH=src
    export ECOS_KEY=YOUR_KEY
    python -m search_agent.adapters.build_ecos_catalog --cache_dir src/search_agent/RAG/fx_data/.ecos_cache --lang kr --max_tables 200
"""

import os, argparse
from pathlib import Path
from .ecos_catalog import EcosCatalog

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="src/search_agent/RAG/fx_data/.ecos_cache")
    ap.add_argument("--lang", default="kr", choices=["kr", "en"])
    ap.add_argument("--max_tables", type=int, default=None, help="테이블 개수 제한(스모크용)")
    ap.add_argument("--page_size", type=int, default=10000)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--stop_after_no_data", type=int, default=50)
    args = ap.parse_args()

    api_key = os.getenv("ECOS_KEY")
    if not api_key:
        raise SystemExit("환경변수 ECOS_KEY가 비어 있습니다.")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cat = EcosCatalog(cache_dir=cache_dir, lang=args.lang)
    t, i = cat.build(
        api_key=api_key,
        max_tables=args.max_tables,
        sleep=args.sleep,
        page_size=args.page_size,
        stop_after_consecutive_no_data=args.stop_after_no_data,
    )
    print(f"[done] tables={len(t):,}, items={len(i):,}")
    print(f"[files] {cache_dir}/tables_{args.lang}.csv , {cache_dir}/items_{args.lang}.csv")

if __name__ == "__main__":
    main()
