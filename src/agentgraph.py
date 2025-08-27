from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from .agentstate import AgentState


def load_llm(model_name: str, max_new_tokens: int = 256, temperature: float = 0.2) -> Runnable:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        pad_token_id=tok.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen)


def build_agent_graph(llm: Runnable, retriever, tools: dict):
    g = StateGraph(AgentState)

    def node_intent(state: AgentState):
        return {"question": state["question"]}

    def node_retrieve(state: AgentState):
        q = state["question"]
        docs = retriever.get_relevant_documents(q)
        texts = [d.page_content[:1000] for d in docs]
        return {"retrieved": texts}

    def node_decide_tool(state: AgentState):
        q = state["question"]
        choice = "none"
        if any(k in q for k in ["검색", "뉴스", "웹"]):
            choice = "web_search"
        elif "RL" in q or "시뮬레이션" in q:
            choice = "rl_simulate"
        return {"tool_choice": choice}

    def node_run_tool(state: AgentState):
        choice = state.get("tool_choice", "none")
        q = state["question"]
        if choice == "web_search":
            return {"tool_result": tools["web_search"].invoke(q)}
        elif choice == "rl_simulate":
            return {"tool_result": tools["rl_simulate"].invoke("baseline")}
        else:
            return {"tool_result": ""}

    def node_answer(state: AgentState):
        sys = "당신은 외환 헤지 의사결정을 돕는 전문가입니다. 간결하고 근거를 요약해 답변하세요."
        retrieved = "\n\n".join(state.get("retrieved", []))
        tool = state.get("tool_result", "")
        prompt = f"""[컨텍스트]
{retrieved}

[도구결과]
{tool}

[질문]
{state['question']}

[지시]
- 위 컨텍스트와 도구결과를 근거로 답변
- 필요시 간단한 헤지 제안(방향/근거/리스크)
"""
        res = llm.invoke([SystemMessage(content=sys), HumanMessage(content=prompt)])
        text = res if isinstance(res, str) else getattr(res, "content", str(res))
        return {"answer": text}

    g.add_node("intent", node_intent)
    g.add_node("retrieve", node_retrieve)
    g.add_node("decide_tool", node_decide_tool)
    g.add_node("run_tool", node_run_tool)
    g.add_node("answer", node_answer)

    g.set_entry_point("intent")
    g.add_edge("intent", "retrieve")
    g.add_edge("retrieve", "decide_tool")
    g.add_edge("decide_tool", "run_tool")
    g.add_edge("run_tool", "answer")
    g.add_edge("answer", END)

    return g.compile()


