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
    candidate = (run.outputs or {}).get("output", "")
    reference = (example.outputs or {}).get("answer", "") if example else ""

    judgement = _judge(question=question, reference=reference, candidate=candidate)
    return EvaluationResult(key="correctness", score=judgement.score / 5, comment=judgement.reasoning)


@run_evaluator
def conciseness(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic: reward answers under ~60 words, no LLM call needed."""
    candidate = (run.outputs or {}).get("output", "")
    word_count = len(candidate.split())
    score = 1.0 if word_count <= 60 else max(0.0, 1 - (word_count - 60) / 100)
    return EvaluationResult(key="conciseness", score=score, comment=f"{word_count} words")


@run_evaluator
def non_empty(run: Run, example: Example | None = None) -> EvaluationResult:
    """Heuristic: did the app produce any output at all."""
    candidate = (run.outputs or {}).get("output", "")
    return EvaluationResult(key="non_empty", score=1.0 if candidate.strip() else 0.0)
