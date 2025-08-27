from typing import TypedDict, List


class AgentState(TypedDict, total=False):
    question: str
    retrieved: List[str]
    tool_choice: str
    tool_result: str
    answer: str


