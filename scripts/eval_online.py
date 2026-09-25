"""Online evaluation: score real (production-style) traces after the fact and
attach the scores back as feedback on the run, instead of running a fixed
dataset offline.

Two ways to feed it runs:
  1. --live: invoke the target app on some ad-hoc questions right now, capture
     the resulting run IDs, and score those (simulates traffic).
  2. --project + --minutes: pull the most recent runs already logged to a
     LangSmith project (real production traffic) and score those instead.

--target picks what's under evaluation: the bare prompt->LLM chain (`chain`,
default) or the tool-calling agent (`agent`) — the agent is the more
realistic choice for "online" monitoring, since production traffic is
usually an agent.

Because there's no golden reference for live traffic, only reference-free
evaluators are used here (no `correctness`, which needs a dataset example).

Usage:
    uv run python scripts/eval_online.py --live
    uv run python scripts/eval_online.py --live --target agent
    uv run python scripts/eval_online.py --project my-prod-project --minutes 60
"""

from __future__ import annotations

import argparse
import datetime

from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from langsmith.schemas import Run

from langsmith_evals.agent import run_agent
from langsmith_evals.evaluators import conciseness, non_empty, tool_faithfulness, used_tools
from langsmith_evals.target_app import answer_question

load_dotenv()

LIVE_QUESTIONS = {
    "chain": [
        "What's a quick way to explain recursion to a beginner?",
        "Give me one tip for writing better commit messages.",
    ],
    "agent": [
        "What is 12 plus 8?",
        "What is (7 times 9) divided by 3?",
    ],
}

EVALUATORS = {
    "chain": [conciseness, non_empty],
    "agent": [conciseness, non_empty, used_tools, tool_faithfulness],
}

TARGET_CALLS = {
    "chain": answer_question,
    "agent": run_agent,
}


def parse_args() -> argparse.Namespace:
    # Exactly one of --live / --project is expected (enforced in main(), not
    # here, since argparse can't express "one of these two" cleanly).
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Invoke the app now and score those runs")
    parser.add_argument("--project", default=None, help="LangSmith project to pull recent runs from")
    parser.add_argument("--minutes", type=int, default=60, help="Look back window when using --project")
    parser.add_argument("--target", choices=["chain", "agent"], default="chain")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def _score_and_log(client: Client, run: Run, session_id, evaluators) -> None:
    # Runs each evaluator against this one run and immediately writes its score
    # back to LangSmith as feedback attached to that run (visible in the UI).
    for evaluator in evaluators:
        result = evaluator.evaluate_run(run)
        client.create_feedback(
            run_id=run.id,
            key=result.key,
            score=result.score,
            comment=result.comment,
            session_id=session_id,
        )
        print(f"run {run.id}: {result.key}={result.score} ({result.comment or ''})")


def _resolve_session_id(client: Client, run: Run):
    # "session_id" is LangSmith's internal name for a project's id. Runs fetched
    # via list_runs() already carry it; runs captured locally off a live
    # invocation don't, so fall back to looking it up by project name instead.
    session_id = getattr(run, "session_id", None)
    if session_id:
        return session_id
    session_name = getattr(run, "session_name", None)
    return client.read_project(project_name=session_name).id if session_name else None


def get_live_runs(target: str, provider: str | None, model: str | None) -> list[Run]:
    """Invoke the app and return its run trees directly (already populated with
    outputs locally — no need to round-trip through the API, which lags behind
    real time since ingestion is asynchronous)."""
    target_fn = TARGET_CALLS[target]

    runs: list[Run] = []
    for question in LIVE_QUESTIONS[target]:
        with collect_runs() as cb:
            target_fn(question, provider=provider, model=model)
        # collect_runs() gathers every top-level run started in the context, in
        # completion order. For an agent that's its internal LLM/tool calls
        # *and* the outer graph run; the graph run finishes last, so [-1]
        # is the one whose outputs are the full {"messages": [...]} state.
        runs.append(cb.traced_runs[-1])

    # create_feedback() below needs the run to already exist server-side.
    wait_for_all_tracers()
    return runs


def get_recent_project_runs(client: Client, project_name: str, minutes: int) -> list[Run]:
    """Fetch real, already-logged runs from a LangSmith project instead of
    generating traffic ourselves — the actual "monitor production" path.
    `is_root=True` skips internal/child runs (tool calls, sub-steps) and
    returns only the top-level invocation of each trace."""
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

    # Get the list of runs to score, from whichever source was requested.
    if args.live:
        runs = get_live_runs(args.target, args.provider, args.model)
    elif args.project:
        runs = get_recent_project_runs(client, args.project, args.minutes)
    else:
        raise SystemExit("Pass --live or --project <name>")

    if not runs:
        print("No runs found to score.")
        return

    # Same evaluators, applied to every run, regardless of where it came from.
    evaluators = EVALUATORS[args.target]
    for run in runs:
        _score_and_log(client, run, _resolve_session_id(client, run), evaluators)


if __name__ == "__main__":
    main()
