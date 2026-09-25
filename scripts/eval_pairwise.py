"""Pairwise (comparative) evaluation: run two variants of the target app over
the same dataset, then have an LLM judge pick a winner example-by-example.

Useful for A/B-testing prompts, models, or providers against each other.

Usage:
    uv run python scripts/eval_pairwise.py --a-provider openai --b-provider ollama --b-model llama3.1
"""

from __future__ import annotations

import argparse
import functools

from dotenv import load_dotenv
from langsmith.evaluation import evaluate, evaluate_comparative
from langsmith.evaluation.evaluator import ComparisonEvaluationResult
from langsmith.schemas import Example, Run

from langsmith_evals.providers import get_chat_model
from langsmith_evals.target_app import answer_question

load_dotenv()

DATASET_NAME = "qa-smoke-test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--a-provider", default="openai")
    parser.add_argument("--a-model", default=None)
    parser.add_argument("--b-provider", default="ollama")
    parser.add_argument("--b-model", default=None)
    return parser.parse_args()


def _target(inputs: dict, provider: str, model: str | None) -> dict:
    return {"output": answer_question(inputs["question"], provider=provider, model=model)}


def preference(runs: list[Run], example: Example | None = None) -> ComparisonEvaluationResult:
    """LLM judge picks the better of the two candidate answers (1.0 winner / 0.0 loser)."""
    question = example.inputs.get("question", "") if example else ""
    reference = (example.outputs or {}).get("answer", "") if example else ""

    candidates = "\n".join(f"[{i}] (run {run.id}): {(run.outputs or {}).get('output', '')}" for i, run in enumerate(runs))

    llm = get_chat_model()
    verdict = llm.invoke(
        [
            (
                "system",
                "You are comparing two candidate answers to a question against a reference answer. "
                "Reply with only the index (0 or 1) of the better candidate.",
            ),
            ("human", f"Question: {question}\nReference: {reference}\n\nCandidates:\n{candidates}"),
        ]
    )
    winner_index = 0 if "0" in verdict.content else 1
    scores = {run.id: (1.0 if i == winner_index else 0.0) for i, run in enumerate(runs)}
    return ComparisonEvaluationResult(key="preference", scores=scores)


def main() -> None:
    args = parse_args()

    experiment_a = evaluate(
        functools.partial(_target, provider=args.a_provider, model=args.a_model),
        data=args.dataset,
        experiment_prefix=f"pairwise-a-{args.a_provider}",
    )
    experiment_b = evaluate(
        functools.partial(_target, provider=args.b_provider, model=args.b_model),
        data=args.dataset,
        experiment_prefix=f"pairwise-b-{args.b_provider}",
    )

    results = evaluate_comparative(
        (experiment_a.experiment_name, experiment_b.experiment_name),
        evaluators=[preference],
    )
    print(results)


if __name__ == "__main__":
    main()
