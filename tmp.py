from search_agent.tools.forecast_tool import ForecastTool
tool = ForecastTool(chroma_path="./RAG/chroma_store", collection_name="model_forecast")
rows = tool.query_forecast("2025-09-04", "2025-09-10", top_k=7)
for r in rows:
    print(r["date"], r["forecast"][:120], "...")
