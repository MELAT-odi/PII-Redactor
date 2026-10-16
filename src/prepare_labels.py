import json
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "synthetic"

split_files = {
    "train": data_dir / "train.jsonl",
    "validation": data_dir / "validation.jsonl",
    "test": data_dir / "test.jsonl",
}


def load_records(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def collect_labels(records):
    labels = set()

    for record in records:
        labels.update(record["labels"])

    return labels


def validate_records(records, split_name):
    for record in records:
        tokens = record["tokens"]
        labels = record["labels"]

        if len(tokens) != len(labels):
            raise ValueError(
                f"Token/label mismatch in {split_name}, "
                f"record {record['id']}: "
                f"{len(tokens)} tokens, {len(labels)} labels"
            )


def main():
    datasets = {}

    for split_name, file_path in split_files.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {split_name} dataset: {file_path}"
            )

        datasets[split_name] = load_records(file_path)
        validate_records(datasets[split_name], split_name)

    label_list = sorted(collect_labels(datasets["train"]))

    label_to_id = {
        label: index
        for index, label in enumerate(label_list)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    for split_name in ["validation", "test"]:
        unseen_labels = (
            collect_labels(datasets[split_name]) - set(label_list)
        )

        if unseen_labels:
            raise ValueError(
                f"Labels in {split_name} missing from training: "
                f"{sorted(unseen_labels)}"
            )

    print(f"Training records:   {len(datasets['train'])}")
    print(f"Validation records: {len(datasets['validation'])}")
    print(f"Test records:       {len(datasets['test'])}")
    print(f"Number of labels:   {len(label_list)}")

    print("\nLabel mapping:")
    for label, index in label_to_id.items():
        print(f"{index:2d} -> {label}")

    print("\nAll splits have valid token/label lengths.")
    print("Validation and test labels are covered by the training mapping.")


if __name__ == "__main__":
    main()
