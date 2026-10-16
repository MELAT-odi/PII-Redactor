import json
from collections import Counter
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "synthetic"

valid_entity_types = {
    "ADDRESS_PLACE",
    "BANK_ACCOUNT",
    "DATE_OF_BIRTH",
    "EMAIL",
    "NATIONAL_ID",
    "PERSON_NAME",
    "PHONE",
    "TIN",
}

valid_labels = {"O"}
for entity_type in valid_entity_types:
    valid_labels.add(f"B-{entity_type}")
    valid_labels.add(f"I-{entity_type}")


def validate_file(file_path):
    label_counts = Counter()
    record_count = 0
    errors = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record_count += 1
            record = json.loads(line)
            tokens = record.get("tokens", [])
            labels = record.get("labels", [])
            record_id = record.get("id", f"line {line_number}")

            if len(tokens) != len(labels):
                errors.append(
                    f"{record_id}: {len(tokens)} tokens but "
                    f"{len(labels)} labels"
                )
                continue

            previous_label = "O"

            for token, label in zip(tokens, labels):
                label_counts[label] += 1

                if label not in valid_labels:
                    errors.append(
                        f"{record_id}: invalid label {label!r}"
                    )

                if label.startswith("I-"):
                    entity_type = label[2:]
                    if previous_label not in {
                        f"B-{entity_type}",
                        f"I-{entity_type}",
                    }:
                        errors.append(
                            f"{record_id}: unexpected {label!r} "
                            f"before token {token!r}"
                        )

                previous_label = label

    print(f"\nFile: {file_path.name}")
    print(f"Records: {record_count}")
    print("Label counts:")

    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    if errors:
        print(f"\nFAILED: {len(errors)} issue(s) found.")
        for error in errors[:20]:
            print(f"  - {error}")
        return False

    print("PASSED: token counts, labels, and BIO transitions are valid.")
    return True


def main():
    all_passed = True

    for split in ["train", "validation", "test"]:
        file_path = data_dir / f"{split}.jsonl"
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            all_passed = False
            continue

        if not validate_file(file_path):
            all_passed = False

    if not all_passed:
        raise SystemExit("\nValidation failed. Fix the issues before training.")

    print("\nAll three datasets passed validation.")


if __name__ == "__main__":
    main()
