import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "raw_annotated.jsonl"
)


def check_dataset():

    record_count = 0
    entity_counts = Counter()

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(line)

            record_count += 1

            for entity in record["entities"]:

                entity_counts[entity["label"]] += 1

    print("\n========== DATASET SUMMARY ==========\n")

    print(f"Total documents: {record_count}")

    print("\nPII entity counts:")

    for label, count in entity_counts.items():

        print(f"{label:20} {count}")

    print("\nTotal PII entities:", sum(entity_counts.values()))


if __name__ == "__main__":
    check_dataset()