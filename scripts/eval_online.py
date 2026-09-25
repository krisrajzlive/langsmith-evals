"""Online evaluation: score real (production-style) traces after the fact and
attach the scores back as feedback on the run, instead of running a fixed
dataset offline.

Two ways to feed it runs:
  1. --live: invoke the target app on some ad-hoc questions right now, capture
     the resulting run IDs, and score those (simulates traffic).
  2. --project + --minutes: pull the most recent runs already logged to a
     LangSmith project (real production traffic) and score those instead.

Because there's no golden reference for live traffic, only reference-free
evaluators are used here (no `correctness`, which needs a dataset example).

Usage:
    uv run python scripts/eval_online.py --live
    uv run python scripts/eval_online.py --project my-prod-project --minutes 60
"""

from __future__ import annotations

import argparse
import datetime

from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from langsmith.schemas import Run

from langsmith_evals.evaluators import conciseness, non_empty
from langsmith_evals.target_app import answer_question

load_dotenv()

LIVE_QUESTIONS = [
    "What's a quick way to explain recursion to a beginner?",
    "Give me one tip for writing better commit messages.",
]

EVALUATORS = [conciseness, non_empty]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Invoke the app now and score those runs")
    parser.add_argument("--project", default=None, help="LangSmith project to pull recent runs from")
    parser.add_argument("--minutes", type=int, default=60, help="Look back window when using --project")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def _score_and_log(client: Client, run: Run) -> None:
    for evaluator in EVALUATORS:
        result = evaluator.evaluate_run(run)
        client.create_feedback(
            run_id=run.id,
            key=result.key,
            score=result.score,
            comment=result.comment,
        )
        print(f"run {run.id}: {result.key}={result.score} ({result.comment or ''})")


def get_live_runs(provider: str | None, model: str | None) -> list[Run]:
    client = Client()
    runs: list[Run] = []
    for question in LIVE_QUESTIONS:
        with collect_runs() as cb:
            answer_question(question, provider=provider, model=model)
        root_run_id = cb.traced_runs[0].id
        runs.append(client.read_run(root_run_id))
    return runs


def get_recent_project_runs(client: Client, project_name: str, minutes: int) -> list[Run]:
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
    return list(
        client.list_runs(
            project_name=project_name,
            run_type="chain",
            is_root=True,
            start_time=since,
        )
    )


def main() -> None:
    args = parse_args()
    client = Client()

    if args.live:
        runs = get_live_runs(args.provider, args.model)
    elif args.project:
        runs = get_recent_project_runs(client, args.project, args.minutes)
    else:
        raise SystemExit("Pass --live or --project <name>")

    if not runs:
        print("No runs found to score.")
        return

    for run in runs:
        _score_and_log(client, run)


if __name__ == "__main__":
    main()
