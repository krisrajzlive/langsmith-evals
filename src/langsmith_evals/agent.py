"""The application under evaluation: a tool-calling agent, built with
LangChain's own `create_agent` (langchain.agents) — no extra agent framework.

Unlike `target_app.py` (a bare prompt -> LLM call), this actually exercises
LangChain's agent loop: the model decides which tool(s) to call, observes
results, and may call more tools before answering. Gives the eval scripts
something worth checking beyond text similarity — did it pick the right
tool, and get the right final number.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from langsmith_evals.providers import get_chat_model


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b


TOOLS = [add, multiply, divide]

_SYSTEM_PROMPT = (
    "You are a careful math assistant. For any arithmetic, always use the "
    "provided tools instead of computing by hand. Give a one-sentence final answer."
)


def build_agent(provider: str | None = None, model: str | None = None):
    llm = get_chat_model(provider=provider, model=model)
    return create_agent(llm, tools=TOOLS, system_prompt=_SYSTEM_PROMPT)


def run_agent(question: str, provider: str | None = None, model: str | None = None) -> dict:
    """Run the agent and return its final answer plus which tools it called.

    Returned dict shape is what ends up as a run's `outputs` when used as a
    target function in `evaluate()`, so keep it evaluator-friendly.
    """
    agent = build_agent(provider=provider, model=model)
    result = agent.invoke({"messages": [("human", question)]})

    messages = result["messages"]
    final_answer = next(m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content)
    tool_calls = [m.name for m in messages if isinstance(m, ToolMessage)]

    return {"output": final_answer, "tool_calls": tool_calls}
