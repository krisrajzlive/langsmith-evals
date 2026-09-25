"""Create (or update) the LangSmith dataset used by the offline/pairwise eval scripts.

Usage:
    uv run python scripts/create_dataset.py
"""

from __future__ import annotations

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "qa-smoke-test"

EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
    },
    {
        "question": "What does CPU stand for?",
        "answer": "CPU stands for Central Processing Unit.",
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "answer": "William Shakespeare wrote 'Romeo and Juliet'.",
    },
    {
        "question": "What is the boiling point of water in Celsius at sea level?",
        "answer": "Water boils at 100 degrees Celsius at sea level.",
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter is the largest planet in our solar system.",
    },
]


def main() -> None:
    # Reuse the dataset if it already exists instead of erroring or duplicating it.
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset {DATASET_NAME!r} already exists, reusing it.")
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Small factual QA set used to smoke-test the eval pipelines.",
        )
        print(f"Created dataset {DATASET_NAME!r} ({dataset.id}).")

    # Only add examples that aren't already in the dataset, so re-running this
    # script is safe (no duplicate rows) instead of appending EXAMPLES every time.
    existing = list(client.list_examples(dataset_id=dataset.id))
    existing_questions = {ex.inputs.get("question") for ex in existing}
    new_examples = [ex for ex in EXAMPLES if ex["question"] not in existing_questions]

    if not new_examples:
        print("No new examples to add.")
        return

    # inputs/outputs are parallel lists here: index i of each belongs to the same example.
    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in new_examples],
        outputs=[{"answer": ex["answer"]} for ex in new_examples],
        dataset_id=dataset.id,
    )
    print(f"Added {len(new_examples)} example(s) to {DATASET_NAME!r}.")


if __name__ == "__main__":
    main()
