
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


project_root = Path(__file__).resolve().parent.parent
test_file = project_root / "data" / "synthetic" / "test.jsonl"
model_dir = project_root / "models" / "ner_model"


def load_test_data(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


def load_saved_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForTokenClassification.from_pretrained(
        model_path
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    id_to_label = {
        int(key): value
        for key, value in model.config.id2label.items()
    }

    return tokenizer, model, device, id_to_label


def predict_word_labels(tokens, tokenizer, model, device, id_to_label):
    encoded_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        return_tensors="pt"
    )

    word_ids = encoded_inputs.word_ids(batch_index=0)

    model_inputs = {
        key: value.to(device)
        for key, value in encoded_inputs.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    predicted_ids = outputs.logits[0].argmax(dim=-1).tolist()

    predicted_labels = []
    previous_word_id = None

    for word_id, predicted_id in zip(word_ids, predicted_ids):
        if word_id is None:
            continue

        if word_id != previous_word_id:
            predicted_labels.append(id_to_label[predicted_id])

        previous_word_id = word_id

    return predicted_labels


def evaluate_model():
    print("Loading test data...")
    test_records = load_test_data(test_file)

    print("Loading saved model...")
    tokenizer, model, device, id_to_label = load_saved_model(
        model_dir
    )

    print(f"Device: {device}")
    print(f"Test documents: {len(test_records)}")
    print("\nEvaluating model...")

    true_labels = []
    predicted_labels = []
    skipped_records = 0

    for record in test_records:
        tokens = record["tokens"]
        actual_labels = record["labels"]

        predicted = predict_word_labels(
            tokens,
            tokenizer,
            model,
            device,
            id_to_label
        )

        if len(predicted) != len(actual_labels):
            skipped_records += 1
            continue

        true_labels.append(actual_labels)
        predicted_labels.append(predicted)

    if not true_labels:
        raise ValueError(
            "No records could be evaluated. "
            "Check token and label alignment."
        )

    print("\n=== Overall Entity-Level Metrics ===")
    print(f"Precision: {precision_score(true_labels, predicted_labels):.4f}")
    print(f"Recall:    {recall_score(true_labels, predicted_labels):.4f}")
    print(f"F1-score:  {f1_score(true_labels, predicted_labels):.4f}")

    print("\n=== Per-Category Report ===")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            digits=4,
            zero_division=0
        )
    )

    print(f"Evaluated documents: {len(true_labels)}")
    print(f"Skipped documents: {skipped_records}")


if __name__ == "__main__":
    evaluate_model()

    

