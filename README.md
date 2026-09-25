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

## Running the scripts

1. One-time setup: `uv sync`, then `cp .env.example .env` and fill in keys (see below).
2. Seed datasets (once each — safe to re-run, they skip examples that already exist):
   ```bash
   uv run python scripts/create_dataset.py         # qa-smoke-test, for `chain`
   uv run python scripts/create_agent_dataset.py   # agent-math-tasks, for `agent`
   ```
3. Then run whichever of these you need, in any order:

   | Script | Needs a dataset first? | What it does |
   |---|---|---|
   | `eval_offline.py` | yes (`qa-smoke-test`) | batch-scores the plain chain |
   | `eval_agent_offline.py` | yes (`agent-math-tasks`) | batch-scores the agent, including tool usage |
   | `eval_online.py --live` | no | scores traces generated right now |
   | `eval_online.py --project <name>` | no | scores traces already logged to a LangSmith project |
   | `eval_pairwise.py` | yes (matches `--a-target`) | A/B two variants (providers/models/targets) |

   The only hard rule is dataset-before-offline/pairwise on that dataset; `eval_online.py` needs no dataset at all.

## Apps under test

Two targets, both plain LangChain, evaluated the same way:

- **`chain`** ([target_app.py](src/langsmith_evals/target_app.py)) — a bare
  `ChatPromptTemplate | chat_model` Runnable. No tools, single LLM call.
- **`agent`** ([agent.py](src/langsmith_evals/agent.py)) — a tool-calling
  agent built with LangChain's own `create_agent` (`langchain.agents`, no
  extra framework) and calculator tools (`add`/`multiply`/`divide`) built
  with LangChain's `@tool` decorator. Exercises the actual agent loop: tool
  selection, tool execution, and (sometimes) self-correction after a bad
  tool call.

## Project layout

```
src/langsmith_evals/
  providers.py     # get_chat_model(provider, model) for openai/ollama/huggingface
  target_app.py     # target 1: bare prompt -> LLM chain
  agent.py          # target 2: LangChain create_agent with calculator tools
  evaluators.py     # shared evaluators: correctness (LLM judge), conciseness,
                     #   non_empty, used_tools, tool_choice_correctness
scripts/
  create_dataset.py         # seed the QA dataset (for `chain`)
  create_agent_dataset.py   # seed the arithmetic dataset (for `agent`)
  eval_offline.py           # batch eval of `chain` over its dataset
  eval_agent_offline.py     # batch eval of `agent` over the arithmetic dataset
  eval_online.py            # score live/production traces (--target chain|agent)
  eval_pairwise.py          # compare two variants (providers, models, or targets) head-to-head
tests/
  test_evaluators.py   # unit tests for the non-LLM evaluators (no API calls)
```

## Offline evaluation

Run a target app over a fixed dataset and score every example — the standard
"batch eval" workflow, producing a scored experiment in LangSmith. The agent
evaluators additionally check whether it called a tool at all (`used_tools`)
and whether it called the *expected* tool(s) for that task (`tool_choice_correctness`).

```bash
uv run python scripts/create_dataset.py           # once, seeds qa-smoke-test (chain)
uv run python scripts/create_agent_dataset.py      # once, seeds agent-math-tasks (agent)

uv run python scripts/eval_offline.py
uv run python scripts/eval_offline.py --provider ollama --model llama3.1
uv run python scripts/eval_agent_offline.py
```

## Online evaluation

Score real traces after the fact instead of a fixed dataset — either traffic
you generate right now (`--live`) or recent runs already logged to a
LangSmith project (`--project`). `--target` picks which app's traces to score
(`chain` default, or `agent`). Reference-free evaluators only (no dataset
example to compare against); results are attached back to each run as
LangSmith feedback.

```bash
uv run python scripts/eval_online.py --live
uv run python scripts/eval_online.py --live --target agent
uv run python scripts/eval_online.py --project my-prod-project --minutes 60
```

## Pairwise (comparative) evaluation

Run two variants over the same dataset, then have an LLM judge pick a winner
example-by-example. Vary provider/model for a given target, or compare the
chain against the agent directly on the same tasks:

```bash
uv run python scripts/eval_pairwise.py --a-provider openai --b-provider ollama --b-model llama3.1
uv run python scripts/eval_pairwise.py --a-target agent --a-provider openai --b-target agent --b-provider ollama --b-model llama3.1
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
