"""Create (or update) the LangSmith dataset used to evaluate the tool-calling agent.

Unlike `create_dataset.py` (plain factual QA), these tasks require arithmetic
that the agent should hand off to its `add`/`multiply`/`divide` tools rather
than compute itself — so evaluators can check both the final number and
whether it actually used the tools.

Usage:
    uv run python scripts/create_agent_dataset.py
"""

from __future__ import annotations

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "agent-math-tasks"

EXAMPLES = [
    {
        "question": "What is 12 plus 8?",
        "answer": "20",
        "expected_tools": ["add"],
    },
    {
        "question": "What is 7 multiplied by 9?",
        "answer": "63",
        "expected_tools": ["multiply"],
    },
    {
        "question": "What is 144 divided by 12?",
        "answer": "12",
        "expected_tools": ["divide"],
    },
    {
        "question": "What is (12 + 8) multiplied by 3?",
        "answer": "60",
        "expected_tools": ["add", "multiply"],
    },
    {
        "question": "If I have 100 dollars and split it evenly between 4 people, then each person gets how much, times 2?",
        "answer": "50",
        "expected_tools": ["divide", "multiply"],
    },
]


def main() -> None:
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset {DATASET_NAME!r} already exists, reusing it.")
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Arithmetic tasks the agent should solve via its calculator tools, not by hand.",
        )
        print(f"Created dataset {DATASET_NAME!r} ({dataset.id}).")

    existing = list(client.list_examples(dataset_id=dataset.id))
    existing_questions = {ex.inputs.get("question") for ex in existing}
    new_examples = [ex for ex in EXAMPLES if ex["question"] not in existing_questions]

    if not new_examples:
        print("No new examples to add.")
        return

    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in new_examples],
        outputs=[{"answer": ex["answer"], "expected_tools": ex["expected_tools"]} for ex in new_examples],
        dataset_id=dataset.id,
    )
    print(f"Added {len(new_examples)} example(s) to {DATASET_NAME!r}.")


if __name__ == "__main__":
    main()
