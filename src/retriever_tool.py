from langchain.tools import tool


@tool("web_search")
def web_search_tool(query: str) -> str:
    """
    간단한 웹 검색. 네트워크 불가 시 빈 문자열.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            res = ddgs.text(query, max_results=5)
            lines = []
            for r in res:
                lines.append(
                    f"title: {r.get('title','')}\nbody: {r.get('body','')}\nurl: {r.get('href','')}"
                )
            return "\n---\n".join(lines)
    except Exception:
        return ""


@tool("rl_simulate")
def rl_simulate_tool(params: str) -> str:
    """
    RL 기반 간단 시뮬레이션 더미. 파라미터 문자열을 받아 결과 요약 반환.
    """
    return "RL 시뮬레이션 결과 요약: 더미 PnL +1.2%, 변동성 0.8%, 샤프 1.5 (파라미터: " + params + ")"


