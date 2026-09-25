# langsmith-evals

Evaluation scripts for LLM apps built with LangChain, scored and tracked in
LangSmith. Supports OpenAI, Ollama, and Hugging Face as interchangeable
providers for both the app under test and the LLM-as-judge.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Fill in `.env`:
- `LANGSMITH_API_KEY` — required for all scripts (they log to LangSmith).
- `PROVIDER` — default provider for the target app + judge: `openai`, `ollama`, or `huggingface`.
- `OPENAI_API_KEY` / `HUGGINGFACEHUB_API_TOKEN` — as needed for your provider.
- `OLLAMA_BASE_URL` — only if your Ollama server isn't `http://localhost:11434`.

## Project layout

```
src/langsmith_evals/
  providers.py     # get_chat_model(provider, model) for openai/ollama/huggingface
  target_app.py     # the app under test — a minimal QA chain
  evaluators.py     # shared evaluators: correctness (LLM judge), conciseness, non_empty
scripts/
  create_dataset.py   # seed a LangSmith dataset with QA examples
  eval_offline.py      # batch eval over a fixed dataset (offline)
  eval_online.py       # score live/production traces after the fact (online)
  eval_pairwise.py     # compare two providers/models head-to-head
tests/
  test_evaluators.py   # unit tests for the non-LLM evaluators (no API calls)
```

## Offline evaluation

Run the target app over a fixed dataset and score every example — the
standard "batch eval" workflow, producing a scored experiment in LangSmith.

```bash
uv run python scripts/create_dataset.py     # once, to seed the dataset
uv run python scripts/eval_offline.py
uv run python scripts/eval_offline.py --provider ollama --model llama3.1
```

## Online evaluation

Score real traces after the fact instead of a fixed dataset — either traffic
you generate right now (`--live`) or recent runs already logged to a
LangSmith project (`--project`). Reference-free evaluators only (no dataset
example to compare against); results are attached back to each run as
LangSmith feedback.

```bash
uv run python scripts/eval_online.py --live
uv run python scripts/eval_online.py --project my-prod-project --minutes 60
```

## Pairwise (comparative) evaluation

Run two variants (e.g. different providers/models/prompts) over the same
dataset, then have an LLM judge pick a winner example-by-example.

```bash
uv run python scripts/eval_pairwise.py --a-provider openai --b-provider ollama --b-model llama3.1
```

## Tests

```bash
uv run pytest
```

## Notes on providers

- `openai` and most `ollama` models (e.g. `llama3.1`) support the structured
  output the `correctness` judge relies on. `huggingface` models vary — if the
  judge step fails on a HF model, point `PROVIDER`/judge calls at `openai` or
  `ollama` while keeping the target app on `huggingface`.
