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


def _extract_output_text(run: Run) -> str:
    """Pull the answer text out of a run's outputs, whatever shape they're in.

    `evaluate()`-driven runs have outputs like `{"output": "some string"}`
    (the target function's own return value). Runs captured straight off a
    LangChain `Runnable` (e.g. in the online eval script) instead carry the
    last step's raw return value, e.g. `{"output": AIMessage(...)}`.
    """
    value = (run.outputs or {}).get("output", "")
    if hasattr(value, "content"):
        return value.content
    return value if isinstance(value, str) else str(value)


def _judge(question: str, reference: str, candidate: str) -> Judgement:
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
    reference = (example.outputs or {}).get("answer", "") if example else ""

    judgement = _judge(question=question, reference=reference, candidate=candidate)
    return EvaluationResult(key="correctness", score=judgement.score / 5, comment=judgement.reasoning)


@run_evaluator
def conciseness(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic: reward answers under ~60 words, no LLM call needed."""
    candidate = _extract_output_text(run)
    word_count = len(candidate.split())
    score = 1.0 if word_count <= 60 else max(0.0, 1 - (word_count - 60) / 100)
    return EvaluationResult(key="conciseness", score=score, comment=f"{word_count} words")


@run_evaluator
def non_empty(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic: did the app produce any output at all."""
    candidate = _extract_output_text(run)
    return EvaluationResult(key="non_empty", score=1.0 if candidate.strip() else 0.0)
