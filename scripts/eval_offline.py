"""Offline evaluation: run the target app over a fixed LangSmith dataset and score it.

This is the classic "batch eval" workflow — run once, get a scored experiment
in LangSmith you can compare against future runs.

Usage:
    uv run python scripts/create_dataset.py   # once, to seed the dataset
    uv run python scripts/eval_offline.py
    uv run python scripts/eval_offline.py --provider ollama --model llama3.1
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langsmith.evaluation import evaluate

from langsmith_evals.evaluators import conciseness, correctness, non_empty
from langsmith_evals.target_app import answer_question

load_dotenv()

DATASET_NAME = "qa-smoke-test"


def parse_args() -> argparse.Namespace:
    # --provider/--model select the chain's LLM; --dataset overrides which LangSmith
    # dataset to evaluate against (defaults to the one create_dataset.py seeds).
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="openai | ollama | huggingface")
    parser.add_argument("--model", default=None, help="Provider-specific model name")
    parser.add_argument("--dataset", default=DATASET_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def target(inputs: dict) -> dict:
        # `inputs` comes from LangSmith: one dataset row's `inputs` field per call.
        # `args.provider`/`args.model` come from the CLI flags: fixed for the whole run.
        question = inputs["question"]
        answer = answer_question(question, provider=args.provider, model=args.model)
        return {"output": answer}

    # Runs `target` once per dataset example, scores each result with every
    # evaluator, and logs everything as a named experiment in LangSmith.
    results = evaluate(
        target,
        data=args.dataset,
        evaluators=[correctness, conciseness, non_empty],
        experiment_prefix=f"offline-{args.provider or 'default'}",
        metadata={"provider": args.provider, "model": args.model},
    )
    print(results)


if __name__ == "__main__":
    main()
