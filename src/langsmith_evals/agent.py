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


# @tool's docstring isn't just documentation here — LangChain sends it to the
# model as the tool's description, so it's what the LLM reads to decide when
# to call this tool. Keep it accurate; a misleading docstring misleads the model too.


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b + 5  # NOTE: this is wrong (off by +5) — flagged earlier, left as-is per your edit


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
    # create_agent() compiles a graph that loops: call the model -> if it
    # requested tool calls, run them -> feed results back to the model -> repeat
    # until it answers without calling a tool.
    llm = get_chat_model(provider=provider, model=model)
    return create_agent(llm, tools=TOOLS, system_prompt=_SYSTEM_PROMPT)


def run_agent(question: str, provider: str | None = None, model: str | None = None) -> dict:
    """Run the agent and return its final answer plus which tools it called.

    Returned dict shape is what ends up as a run's `outputs` when used as a
    target function in `evaluate()`, so keep it evaluator-friendly.
    """
    agent = build_agent(provider=provider, model=model)
    # create_agent's state shape: {"messages": [HumanMessage, AIMessage, ToolMessage, ...]}
    # — the full conversation, including every intermediate tool call and result.
    result = agent.invoke({"messages": [("human", question)]})

    messages = result["messages"]
    # Walk backwards to find the last AI message with content — that's the final
    # answer, as opposed to an earlier AIMessage that only requested a tool call
    # (those have empty .content).
    final_answer = next(m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content)
    # Every ToolMessage in the transcript is one tool call's result; its .name
    # is which tool produced it, so this recovers the full call sequence.
    tool_calls = [m.name for m in messages if isinstance(m, ToolMessage)]
    # ...and its .content is what that tool actually returned — kept separately
    # so evaluators can check the final answer against the *real* tool output,
    # not just "was some tool called."
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]

    return {"output": final_answer, "tool_calls": tool_calls, "tool_results": tool_results}
