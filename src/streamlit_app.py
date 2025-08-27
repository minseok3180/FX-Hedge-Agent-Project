import streamlit as st
from src.config import Config
from src.rag_parsing import get_retriever
from src.retriever_tool import web_search_tool, rl_simulate_tool
from src.agentgraph import load_llm, build_agent_graph


st.set_page_config(page_title="FX Hedge Agent", page_icon="💱")
st.title("FX Hedge Agent (MVP)")


@st.cache_resource
def bootstrap():
    cfg = Config()
    retriever = get_retriever(cfg.vector_db_dir, cfg.embedding_model)
    llm = load_llm(cfg.hf_model_name, cfg.max_new_tokens, cfg.temperature)
    tools = {"web_search": web_search_tool, "rl_simulate": rl_simulate_tool}
    graph = build_agent_graph(llm, retriever, tools)
    return cfg, graph


cfg, graph = bootstrap()

user_q = st.chat_input("질문을 입력하세요 (예: 오늘 환율 헤지 방향?)")
if "history" not in st.session_state:
    st.session_state["history"] = []

if user_q:
    st.session_state["history"].append(("user", user_q))
    result = graph.invoke({"question": user_q})
    st.session_state["history"].append(("assistant", result.get("answer", "(no answer)")))

for role, msg in st.session_state["history"]:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.write(msg)


