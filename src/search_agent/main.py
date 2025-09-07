#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from search_agent.agent.engine import SearchAgent

def main():
    p = argparse.ArgumentParser(description="Search Agent (News Tool + FX Tool)")
    p.add_argument("--q", required=True, help="자연어 질의")
    args = p.parse_args()

    agent = SearchAgent()
    out = agent.run(args.q)

    try:
        # pandas 객체면 표로 출력
        import pandas as pd
        if isinstance(out, (pd.Series, pd.DataFrame)):
            print(out.to_string())
        else:
            print(out)
    except Exception:
        print(out)

if __name__ == "__main__":
    main()
