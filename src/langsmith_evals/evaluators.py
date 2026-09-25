"""Evaluators shared by the offline, online, and pairwise eval scripts.

Each evaluator follows the LangSmith `run_evaluator` contract: it takes a
`Run` (and optionally the golden `Example`) and returns a dict with `key`,
`score`, and an optional `comment`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Example, Run

from langsmith_evals.providers import get_chat_model


class Judgement(BaseModel):
    score: int = Field(description="1 (poor) to 5 (excellent)")
    reasoning: str = Field(description="One sentence justifying the score")


_JUDGE_SYSTEM_PROMPT = """You are grading an AI assistant's answer against a reference answer.
Score the candidate answer's correctness and completeness relative to the reference on a 1-5 scale.
5 = fully correct and complete, 1 = wrong or irrelevant."""


def output_text_from(outputs: dict) -> str:
    """Pull the final answer text out of an outputs dict, whatever shape it's in.

    Two shapes show up across the scripts/apps here:
    - `evaluate()`-driven runs over `target_app.answer_question` / `agent.run_agent`:
      `{"output": "some string"}` (the target function's own return value).
    - a bare LangChain Runnable/agent traced directly (e.g. online eval's --live
      mode): either `{"output": AIMessage(...)}` (target_app) or
      `{"messages": [HumanMessage(...), AIMessage(...), ToolMessage(...), ...]}` (agent).
    """
    outputs = outputs or {}

    if "messages" in outputs:
        # Walk backwards for the last AI message with real content — an earlier
        # AIMessage may only carry a tool-call request with empty .content.
        for message in reversed(outputs["messages"]):
            if type(message).__name__ == "AIMessage" and getattr(message, "content", ""):
                return message.content
        return ""

    value = outputs.get("output", "")
    if hasattr(value, "content"):
        return value.content
    return value if isinstance(value, str) else str(value)


def tool_calls_from(outputs: dict) -> list[str]:
    """Names of tools invoked, across the same output shapes as `output_text_from`."""
    outputs = outputs or {}

    if "messages" in outputs:
        return [m.name for m in outputs["messages"] if type(m).__name__ == "ToolMessage"]

    # evaluate()-driven runs over run_agent() already carry tool_calls directly;
    # target_app's plain chain has none, so this returns [] for it.
    return outputs.get("tool_calls", [])


def tool_results_from(outputs: dict) -> list[str]:
    """Raw content each tool call actually returned, same order as `tool_calls_from`,
    across the same output shapes as `output_text_from`. Lets an evaluator check the
    final answer against what a tool *really* returned, not just that one was called."""
    outputs = outputs or {}

    if "messages" in outputs:
        return [m.content for m in outputs["messages"] if type(m).__name__ == "ToolMessage"]

    return outputs.get("tool_results", [])


# Thin Run-shaped wrappers so the evaluators below (which get a `Run`, per the
# run_evaluator contract) can share the same dict-based extraction logic above.
def _extract_output_text(run: Run) -> str:
    return output_text_from(run.outputs)


def _extract_tool_calls(run: Run) -> list[str]:
    return tool_calls_from(run.outputs)


def _judge(question: str, reference: str, candidate: str) -> Judgement:
    # with_structured_output forces the model's reply into the Judgement schema
    # (score + reasoning) instead of free-form text we'd have to parse ourselves.
    llm = get_chat_model().with_structured_output(Judgement)
    return llm.invoke(
        [
            ("system", _JUDGE_SYSTEM_PROMPT),
            (
                "human",
                f"Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {candidate}",
            ),
        ]
    )


@run_evaluator
def correctness(run: Run, example: Example | None = None) -> EvaluationResult:
    """LLM-as-judge: candidate answer vs. the dataset's reference answer, scaled to 0-1."""
    question = run.inputs.get("question", "")
    candidate = _extract_output_text(run)
    # `example` is only present in offline/dataset-driven evals; live/online runs
    # have no reference answer, so this evaluator isn't used there (see eval_online.py).
    reference = (example.outputs or {}).get("answer", "") if example else ""

    judgement = _judge(question=question, reference=reference, candidate=candidate)
    # Judge scores 1-5; EvaluationResult expects 0-1, hence the /5.
    return EvaluationResult(key="correctness", score=judgement.score / 5, comment=judgement.reasoning)


@run_evaluator
def conciseness(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic: reward answers under ~60 words, no LLM call needed."""
    candidate = _extract_output_text(run)
    word_count = len(candidate.split())
    # Full score up to 60 words, then linearly penalized — 0 once it's 160+ words over.
    score = 1.0 if word_count <= 60 else max(0.0, 1 - (word_count - 60) / 100)
    return EvaluationResult(key="conciseness", score=score, comment=f"{word_count} words")


@run_evaluator
def non_empty(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic: did the app produce any output at all."""
    candidate = _extract_output_text(run)
    return EvaluationResult(key="non_empty", score=1.0 if candidate.strip() else 0.0)


@run_evaluator
def used_tools(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic (agent only): did it actually call a tool instead of guessing by hand."""
    tool_calls = _extract_tool_calls(run)
    return EvaluationResult(key="used_tools", score=1.0 if tool_calls else 0.0, comment=", ".join(tool_calls) or None)


@run_evaluator
def tool_choice_correctness(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic (agent only): did the set of tools called match the dataset's `expected_tools`.

    Scored as overlap (expected tools that were actually called / expected tools),
    so an agent that also retries a tool after a bad call isn't unfairly penalized.
    """
    expected = set((example.outputs or {}).get("expected_tools", [])) if example else set()
    if not expected:
        # No expected_tools on this example (or no example at all, e.g. online eval)
        # -- score=None means "not applicable", not "failed".
        return EvaluationResult(key="tool_choice_correctness", score=None, comment="no expected_tools on example")

    actual = set(_extract_tool_calls(run))
    # Fraction of the expected tools that were actually called, not exact-match —
    # calling extra tools (e.g. retrying after an error) doesn't get penalized.
    score = len(expected & actual) / len(expected)
    return EvaluationResult(
        key="tool_choice_correctness",
        score=score,
        comment=f"expected {sorted(expected)}, got {sorted(actual)}",
    )


@run_evaluator
def tool_faithfulness(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic (agent only): does the final answer actually reflect what the
    *last* tool call returned, rather than the model silently overriding a bad
    (or good) tool result with its own mental math.

    This is the one evaluator here that can catch a broken tool even when the
    model "gets lucky" and answers correctly anyway — `correctness` alone
    can't: it only checks the final answer against the reference, and has no
    idea whether the agent actually relied on its tools to get there.
    """
    tool_results = tool_results_from(run.outputs)
    if not tool_results:
        return EvaluationResult(key="tool_faithfulness", score=None, comment="no tool calls to check")

    last_result = tool_results[-1]
    try:
        value = float(last_result)
    except (TypeError, ValueError):
        return EvaluationResult(key="tool_faithfulness", score=None, comment=f"tool result not numeric: {last_result!r}")

    candidate = _extract_output_text(run)
    # Accept either "25.0" or "25" showing up in the answer text — models
    # normally drop the trailing ".0" for a whole number.
    as_float_str = str(value)
    as_int_str = str(int(value)) if value == int(value) else None
    is_faithful = as_float_str in candidate or (as_int_str is not None and as_int_str in candidate)

    return EvaluationResult(
        key="tool_faithfulness",
        score=1.0 if is_faithful else 0.0,
        comment=f"last tool call returned {value}, final answer was {candidate!r}",
    )
