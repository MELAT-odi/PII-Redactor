
import json
import random
from pathlib import Path

from datasets import load_dataset


project_root = Path(__file__).resolve().parent.parent
synthetic_dir = project_root / "data" / "synthetic"
processed_dir = project_root / "data" / "processed"

random_seed = 42

dataset_files = {
    split_name: (
        "https://huggingface.co/datasets/masakhane/masakhaner/"
        "resolve/refs%2Fconvert%2Fparquet/amh/"
        f"{split_name}/0000.parquet"
    )
    for split_name in ["train", "validation", "test"]
}

label_mapping = {
    "O": "O",
    "B-PER": "B-PERSON_NAME",
    "I-PER": "I-PERSON_NAME",
    "B-LOC": "B-ADDRESS_PLACE",
    "I-LOC": "I-ADDRESS_PLACE",
    "B-ORG": "B-ORG",
    "I-ORG": "I-ORG",
    "B-DATE": "B-DATE",
    "I-DATE": "I-DATE",
}


def read_jsonl(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(file_path, records):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def convert_masakhaner_split(split_data, split_name):
    converted_records = []

    tag_names = split_data.features["ner_tags"].feature.names

    for record in split_data:
        labels = [
            label_mapping[tag_names[tag_id]]
            for tag_id in record["ner_tags"]
        ]

        if len(record["tokens"]) != len(labels):
            raise ValueError(
                f"Token-label mismatch in MasakhaNER record "
                f"{record['id']}"
            )

        converted_records.append({
            "id": f"masakhaner_{split_name}_{record['id']}",
            "tokens": record["tokens"],
            "labels": labels,
        })

    return converted_records


def validate_records(records, split_name):
    for record in records:
        if len(record["tokens"]) != len(record["labels"]):
            raise ValueError(
                f"Token-label mismatch in {split_name}: {record['id']}"
            )


def main():
    print("Loading synthetic training data...")

    synthetic_train = read_jsonl(
        synthetic_dir / "train.jsonl"
    )

    print("Loading MasakhaNER...")

    masakhaner = load_dataset(
        "parquet",
        data_files=dataset_files,
    )

    print("Converting MasakhaNER labels...")

    masakha_train = convert_masakhaner_split(
        masakhaner["train"], "train"
    )
    masakha_validation = convert_masakhaner_split(
        masakhaner["validation"], "validation"
    )
    masakha_test = convert_masakhaner_split(
        masakhaner["test"], "test"
    )

    # Shuffle only the existing synthetic training examples.
    rng = random.Random(random_seed)
    rng.shuffle(synthetic_train)

    validation_size = round(len(synthetic_train) * 0.10)

    synthetic_validation = synthetic_train[:validation_size]
    synthetic_train = synthetic_train[validation_size:]

    # Combine matching splits only.
    combined_train = synthetic_train + masakha_train
    combined_validation = (
        synthetic_validation + masakha_validation
    )

    rng.shuffle(combined_train)
    rng.shuffle(combined_validation)

    validate_records(combined_train, "training")
    validate_records(combined_validation, "validation")
    validate_records(masakha_test, "MasakhaNER test")

    # Save new processed files without overwriting existing datasets.
    write_jsonl(
        processed_dir / "train.jsonl",
        combined_train,
    )
    write_jsonl(
        processed_dir / "validation.jsonl",
        combined_validation,
    )
    write_jsonl(
        processed_dir / "masakhaner_test.jsonl",
        masakha_test,
    )

    print("\nDataset preparation completed.")
    print(f"Combined training examples: {len(combined_train)}")
    print(f"Combined validation examples: {len(combined_validation)}")
    print(f"MasakhaNER test examples: {len(masakha_test)}")
    print("\nSaved files:")
    print("data/processed/train.jsonl")
    print("data/processed/validation.jsonl")
    print("data/processed/masakhaner_test.jsonl")
    print("\nExisting synthetic test and external evaluation files")
    print("were not modified.")


if __name__ == "__main__":
    main()
