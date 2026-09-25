"""Offline evaluation of the tool-calling agent (src/langsmith_evals/agent.py).

Same batch-eval workflow as eval_offline.py, but against the tool-calling
agent instead of the bare prompt->LLM chain, with evaluators that also check
tool usage: did it call a tool at all, and did it call the *expected* tool(s).

Usage:
    uv run python scripts/create_agent_dataset.py   # once, to seed the dataset
    uv run python scripts/eval_agent_offline.py
    uv run python scripts/eval_agent_offline.py --provider ollama --model llama3.1
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langsmith.evaluation import evaluate

from langsmith_evals.agent import run_agent
from langsmith_evals.evaluators import correctness, tool_choice_correctness, tool_faithfulness, used_tools

load_dotenv()

DATASET_NAME = "agent-math-tasks"


def parse_args() -> argparse.Namespace:
    # --provider/--model select the agent's LLM; --dataset overrides which LangSmith
    # dataset to evaluate against (defaults to the one create_agent_dataset.py seeds).
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="openai | ollama | huggingface")
    parser.add_argument("--model", default=None, help="Provider-specific model name")
    parser.add_argument("--dataset", default=DATASET_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def target(inputs: dict) -> dict:
        # run_agent() already returns {"output": ..., "tool_calls": [...]}, the shape
        # evaluate() expects, so unlike eval_offline.py's target() no wrapping is needed.
        return run_agent(inputs["question"], provider=args.provider, model=args.model)

    # Runs `target` once per dataset example, scores each result with every
    # evaluator, and logs everything as a named experiment in LangSmith.
    results = evaluate(
        target,
        data=args.dataset,
        evaluators=[correctness, used_tools, tool_choice_correctness, tool_faithfulness],
        experiment_prefix=f"agent-offline-{args.provider or 'default'}",
        metadata={"provider": args.provider, "model": args.model, "target": "agent"},
    )
    print(results)


if __name__ == "__main__":
    main()
