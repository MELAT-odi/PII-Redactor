import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

train_file = project_root/"data"/"synthetic"/"train.jsonl"
test_file = project_root/"data"/"synthetic"/"test.jsonl"

def load_records(file_path):
    records=[]

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            records.append(
                json.loads(line)
            )

    return records

def collect_labels(records):
    labels= set() # to recomve duplication and contain unique values

    for record in records:
        for label in record["labels"]:
            labels.add(label)

    return labels

def validate_records(records):

    for record in records:

        tokens= record["tokens"]
        labels= record["labels"]

        if len(tokens) != len(labels):
            raise ValueError(
                f"Token/Label mismatch in "
                f"{record["id"]}"
                f"{len(tokens)} tokens"
                f"{len(labels)} labels"
            )

def main():

    train_records= load_records(
        train_file
    )

    test_records= load_records(
        test_file
    )

    validate_records(
        train_records
    )

    validate_records(
        test_records
    )

    train_labels= collect_labels(
        train_records
    )

    test_labels=collect_labels(
        test_records
    )

    all_labels =train_labels |test_labels
    label_list =sorted(
        all_labels

    )

    label_to_id ={
        label: index
        for index, label in enumerate(label_list)
    }

    id_to_label={
        index: label
        for label, index in label_to_id.items()
    }

    print(
        f"Training records:"
        f"{len(train_records)}"
    )

    print(
        f"Test records: "
        f"{len(test_records)}"
    )

    print()

    print(
        f"Number of labels: "
        f"{len(label_list)}"
    )

    print()

    print("Labels:")

    for label in label_list:
        print(
            f"{label_to_id[label]:2d} -> {label}"
        )

    print()

    print("Label mapping:")
    print(label_to_id)

    print()

    print("ID mapping:")
    print(id_to_label)


if __name__ == "__main__":
    main()
