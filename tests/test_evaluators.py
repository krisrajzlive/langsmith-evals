"""Unit tests for the heuristic (non-LLM) evaluators. These run offline, no API keys needed."""

import datetime

from langsmith.schemas import Run

from langsmith_evals.evaluators import conciseness, non_empty


def _make_run(output_text: str) -> Run:
    # Minimal fake Run in the {"output": "..."} shape (matches an evaluate()-driven
    # run over target_app.answer_question), just enough for Run's required fields.
    return Run(
        id="00000000-0000-0000-0000-000000000000",
        name="test-run",
        run_type="chain",
        start_time=datetime.datetime.now(datetime.timezone.utc),
        inputs={"question": "irrelevant"},
        outputs={"output": output_text},
    )


def test_non_empty_scores_zero_for_blank_output():
    result = non_empty.evaluate_run(_make_run(""))
    assert result.score == 0.0


def test_non_empty_scores_one_for_real_output():
    result = non_empty.evaluate_run(_make_run("Paris."))
    assert result.score == 1.0


def test_conciseness_full_score_under_word_limit():
    result = conciseness.evaluate_run(_make_run("A short answer."))
    assert result.score == 1.0


def test_conciseness_penalizes_long_output():
    long_answer = " ".join(["word"] * 200)
    result = conciseness.evaluate_run(_make_run(long_answer))
    assert result.score < 1.0
