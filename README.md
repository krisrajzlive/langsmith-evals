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
  evaluators.py     # shared evaluators (see "Evaluators" below)
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

## Offline vs. online vs. pairwise evaluation

These three answer different questions, and use different evaluators because
of it:

- **Offline** ([eval_offline.py](scripts/eval_offline.py),
  [eval_agent_offline.py](scripts/eval_agent_offline.py)) — runs the app
  against a **fixed, curated dataset** with known-good reference answers.
  Deliberately controlled: same questions every time, so you can compare
  experiments apples-to-apples across code/prompt/model changes ("did this
  change make things better or worse on the same N questions?"). Because
  there's a reference answer, evaluators can compare *against* it
  (`correctness`, `tool_choice_correctness`). Think: a regression test suite.

- **Online** ([eval_online.py](scripts/eval_online.py)) — scores **real
  traces** instead: either traffic you generate right now (`--live`) or
  actual runs already logged to a LangSmith project (`--project`). No
  dataset, no reference answers — you don't control what a user asks, so
  there's nothing to compare against. Only reference-free evaluators apply
  (`conciseness`, `non_empty`, `used_tools`, `tool_faithfulness`). Results
  are attached back to the run as LangSmith *feedback*, not a new
  experiment. Think: production monitoring/observability.

- **Pairwise** ([eval_pairwise.py](scripts/eval_pairwise.py)) — runs **two
  variants** (different providers, models, or targets) over the same
  dataset, then has an LLM judge pick a winner example-by-example. Useful
  for A/B decisions ("is `deep_agent` actually better than `agent`?", "is
  `gpt-4o-mini` good enough vs. a bigger model?") where you want a head-to-head
  comparison rather than two separate absolute scores.

## Evaluators

All defined in [evaluators.py](src/langsmith_evals/evaluators.py):

| Evaluator | Used on | LLM call? | What it checks |
|---|---|---|---|
| `correctness` | any | yes (judge) | Final answer vs. the dataset's reference answer, scored 1-5 -> 0-1. |
| `conciseness` | any | no | Penalizes answers over ~60 words. |
| `non_empty` | any | no | Did the app produce any output at all. |
| `used_tools` | agent only | no | Did it call a tool at all, instead of guessing by hand. |
| `tool_choice_correctness` | agent only | no | Did it call the *expected* tool(s) for that task (dataset's `expected_tools`). |
| `tool_faithfulness` | agent only | no | Does the final answer match what the **last tool call actually returned** — independent of whether that answer happens to be objectively right. |

`tool_faithfulness` is the odd one out and worth calling out: it's the only
evaluator here that can catch a broken tool *even when the model gets the
right answer anyway*. We hit this for real during development — the `add`
tool was deliberately broken (`return a + b + 5`), and `correctness` kept
scoring 1.0 because the model quietly distrusted the wrong tool output and
answered from its own (correct) mental math instead. `tool_faithfulness`
compares the final answer against the tool's actual return value, not the
"real" answer, so it caught it where every other evaluator passed. See
[agent.py](src/langsmith_evals/agent.py) — `run_agent()` returns both
`tool_calls` (names) and `tool_results` (actual returned values) so this
evaluator has something to check against.

**Tracing a low score back to the actual tool call:** open the failing row
in the LangSmith experiment link the script prints, click into it, and the
run's trace tree (`LangGraph -> model -> tools -> <tool name>`) shows each
tool span's real `inputs`/`outputs`. That's the fastest way to see *why* an
evaluator flagged something, not just that it did.

## Offline evaluation

```bash
uv run python scripts/create_dataset.py           # once, seeds qa-smoke-test (chain)
uv run python scripts/create_agent_dataset.py      # once, seeds agent-math-tasks (agent)

uv run python scripts/eval_offline.py
uv run python scripts/eval_offline.py --provider ollama --model llama3.1
uv run python scripts/eval_agent_offline.py
```

## Online evaluation

```bash
uv run python scripts/eval_online.py --live
uv run python scripts/eval_online.py --live --target agent
uv run python scripts/eval_online.py --project my-prod-project --minutes 60
```

## Pairwise (comparative) evaluation

Vary provider/model for a given target, or compare the chain against the
agent directly on the same tasks. Every dataset-`evaluate()` call under the
hood makes its own LLM calls — with two variants plus the judge, that's
**3 separate LLM calls per example**, not 1: variant A's answer, variant B's
answer, and the judge's verdict. The judge itself (`preference()` in
[eval_pairwise.py](scripts/eval_pairwise.py)) always uses `.env`'s default
provider/model — it's not wired to `--a-*`/`--b-*`, so it stays a neutral
third party regardless of what you're comparing.

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
- `ollama` requires a local Ollama server actually running (`ollama serve`,
  or the desktop app) and reachable at `OLLAMA_BASE_URL`
  (`http://localhost:11434` by default) — a connection-refused error here
  means nothing is listening on that port, not a LangSmith/API quota issue.
