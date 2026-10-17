
import json
from pathlib import Path

from rule_detector import detect_national_ids


project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "synthetic" / "raw_annotated.jsonl"

true_positives = 0
false_positives = 0
false_negatives = 0
other_entity_overlaps = 0
records_checked = 0

with dataset_path.open("r", encoding="utf-8") as dataset_file:
    for line in dataset_file:
        record = json.loads(line)
        text = record["text"]
        records_checked += 1

        gold_national_ids = {
            (entity["start"], entity["end"])
            for entity in record["entities"]
            if entity["label"] == "NATIONAL_ID"
        }

        other_entities = [
            entity
            for entity in record["entities"]
            if entity["label"] != "NATIONAL_ID"
        ]

        predictions = {
            (item["start"], item["end"])
            for item in detect_national_ids(text)
        }

        true_positives += len(predictions & gold_national_ids)
        false_negatives += len(gold_national_ids - predictions)

        for start, end in predictions - gold_national_ids:
            false_positives += 1

            if any(
                start < entity["end"] and end > entity["start"]
                for entity in other_entities
            ):
                other_entity_overlaps += 1


precision = (
    true_positives / (true_positives + false_positives)
    if true_positives + false_positives else 0
)

recall = (
    true_positives / (true_positives + false_negatives)
    if true_positives + false_negatives else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print(f"Records checked: {records_checked}")
print(f"True positives: {true_positives}")
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")
print(f"False positives overlapping other entities: {other_entity_overlaps}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
