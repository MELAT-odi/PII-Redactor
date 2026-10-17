import json
import re
from collections import Counter
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
data_file = project_root / "data" / "synthetic" / "raw_annotated.jsonl"

target_labels = {
    "NATIONAL_ID",
    "TIN",
    "BANK_ACCOUNT",
    "PHONE",
}

context_patterns = {
    "fayda_amharic": r"ፋይዳ",
    "national_id_amharic": r"ብሔራዊ\s*መታወቂያ|መታወቂያ\s*ቁጥር",
    "id_amharic": r"መታወቂያ|መለያ\s*ቁጥር",
    "national_id_english": r"national\s*id|national\s*identity",
    "fayda_english": r"fayda|fin\b",
    "id_english": r"\bid\s*(?:number|no\.?|#)\b|\bidentity\s*number\b",
}

length_counts = {
    label: Counter()
    for label in target_labels
}
split_counts = Counter()
context_counts = Counter()
bad_spans = []
examples = Counter()
digit_pattern = re.compile(r"\d")


def get_split(record):
    return record.get("split", "unspecified")


def main():
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    total_records = 0
    total_entities = Counter()

    with data_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            total_records += 1

            text = record.get("text", "")
            split = get_split(record)

            for entity in record.get("entities", []):
                label = entity.get("label")
                if label not in target_labels:
                    continue

                total_entities[label] += 1
                value = entity.get("text", "")
                start = entity.get("start")
                end = entity.get("end")

                if isinstance(start, int) and isinstance(end, int):
                    if text[start:end] != value:
                        bad_spans.append(
                            (line_number, label, "span_text_mismatch")
                        )

                    left = text[max(0, start - 60):start]
                    right = text[end:end + 40]
                    context = f"{left} {right}"
                else:
                    context = text

                digit_count = len(digit_pattern.findall(value))
                length_counts[label][digit_count] += 1

                if label == "NATIONAL_ID":
                    split_counts[split] += 1

                    for cue_name, pattern in context_patterns.items():
                        if re.search(pattern, context, flags=re.IGNORECASE):
                            context_counts[cue_name] += 1

                    # Print only lengths and labels, never identifier values.
                    examples[(split, digit_count)] += 1

    print(f"Records audited: {total_records}")
    print("\nEntity totals:")
    for label in sorted(target_labels):
        print(f"  {label}: {total_entities[label]}")

    print("\nDigit-length distribution:")
    for label in sorted(target_labels):
        print(f"  {label}: {dict(sorted(length_counts[label].items()))}")

    print("\nNATIONAL_ID counts by split:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    print("\nNATIONAL_ID context cues (cues may overlap):")
    for cue_name in sorted(context_patterns):
        print(f"  {cue_name}: {context_counts[cue_name]}")

    print("\nNATIONAL_ID split and digit-length counts:")
    for (split, length), count in sorted(examples.items()):
        print(f"  split={split}, digits={length}: {count}")

    print(f"\nInvalid entity spans: {len(bad_spans)}")
    for item in bad_spans[:10]:
        print(f"  line={item[0]}, label={item[1]}, issue={item[2]}")


if __name__ == "__main__":
    main()