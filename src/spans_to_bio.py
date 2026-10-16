import json
import re
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "synthetic"
input_file = data_dir / "raw_annotated.jsonl"

output_files = {
    "train": data_dir / "train.jsonl",
    "validation": data_dir / "validation.jsonl",
    "test": data_dir / "test.jsonl",
}

token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text):
    return [
        (match.group(), match.start(), match.end())
        for match in token_pattern.finditer(text)
    ]


def convert_record(record):
    text = record["text"]
    entities = sorted(record["entities"], key=lambda item: item["start"])
    tokens_with_offsets = tokenize_with_offsets(text)

    tokens = []
    labels = []
    entity_index = 0
    previous_end = 0

    for entity in entities:
        start = entity["start"]
        end = entity["end"]

        if text[start:end] != entity["text"]:
            raise ValueError(
                f"{record['id']}: entity text does not match its offsets"
            )

        if start < previous_end:
            raise ValueError(f"{record['id']}: overlapping entities")

        previous_end = end

    for token, token_start, token_end in tokens_with_offsets:
        while (
            entity_index < len(entities)
            and entities[entity_index]["end"] <= token_start
        ):
            entity_index += 1

        label = "O"

        if entity_index < len(entities):
            entity = entities[entity_index]
            entity_start = entity["start"]
            entity_end = entity["end"]

            overlaps = (
                token_start < entity_end
                and token_end > entity_start
            )

            if overlaps:
                if token_start < entity_start or token_end > entity_end:
                    raise ValueError(
                        f"{record['id']}: token crosses an entity boundary: "
                        f"{token!r}"
                    )

                prefix = (
                    "B-"
                    if token_start == entity_start
                    else "I-"
                )
                label = prefix + entity["label"]

        tokens.append(token)
        labels.append(label)

    return {
        "id": record["id"],
        "split": record["split"],
        "tokens": tokens,
        "labels": labels,
    }


def main():
    split_records = {
        "train": [],
        "validation": [],
        "test": [],
    }

    with input_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            split = record.get("split")

            if split not in split_records:
                raise ValueError(
                    f"Line {line_number}: invalid split {split!r}"
                )

            split_records[split].append(convert_record(record))

    for split, records in split_records.items():
        output_file = output_files[split]

        with output_file.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        print(f"{split}: {len(records)} records -> {output_file}")

    print("BIO conversion completed successfully.")


if __name__ == "__main__":
    main()