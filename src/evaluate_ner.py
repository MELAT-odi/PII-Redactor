import json
from pathlib import Path

import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)


project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "synthetic"
model_dir = project_root / "models" / "colab_best_model"

evaluation_files = {
    "Held-out test set": data_dir / "test.jsonl",
}


def load_records(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_saved_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model directory not found: {model_path}"
        )

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


def predict_word_labels(
    tokens, tokenizer, model, device, id_to_label
):
    encoded_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
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


def evaluate_dataset(
    dataset_name,
    file_path,
    tokenizer,
    model,
    device,
    id_to_label,
):
    if not file_path.exists():
        print(f"\nSkipping {dataset_name}: file not found.")
        print(f"Expected file: {file_path}")
        return

    records = load_records(file_path)

    true_labels = []
    predicted_labels = []
    skipped_records = 0
    error_count = 0
    error_examples = []

    for record in records:
        tokens = record["tokens"]
        actual_labels = record["labels"]

        if len(tokens) != len(actual_labels):
            skipped_records += 1
            continue

        predicted = predict_word_labels(
            tokens,
            tokenizer,
            model,
            device,
            id_to_label,
        )

        if len(predicted) != len(actual_labels):
            skipped_records += 1
            continue

        true_labels.append(actual_labels)
        predicted_labels.append(predicted)

        if predicted != actual_labels:
            error_count += 1

            if len(error_examples) < 5:
                error_examples.append({
                    "id": record.get("id", "unknown"),
                    "tokens": tokens,
                    "correct": actual_labels,
                    "predicted": predicted,
                })

    print(f"\n{'=' * 60}")
    print(dataset_name)
    print(f"{'=' * 60}")
    print(f"Records in file:       {len(records)}")
    print(f"Evaluated records:     {len(true_labels)}")
    print(f"Skipped records:       {skipped_records}")
    print(f"Records with errors:   {error_count}")

    if not true_labels:
        print("No records could be evaluated.")
        return

    print("\nOverall entity-level metrics")
    print(f"Precision: {precision_score(true_labels, predicted_labels):.4f}")
    print(f"Recall:    {recall_score(true_labels, predicted_labels):.4f}")
    print(f"F1-score:  {f1_score(true_labels, predicted_labels):.4f}")

    print("\nPer-entity-category report")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            digits=4,
            zero_division=0,
        )
    )

    print("\nFirst few prediction errors")
    if not error_examples:
        print("No token-label mismatches found.")
    else:
        for example in error_examples:
            print(f"\nRecord: {example['id']}")
            print("Tokens:    ", example["tokens"])
            print("Correct:   ", example["correct"])
            print("Predicted: ", example["predicted"])


def main():
    print("Loading saved model...")

    tokenizer, model, device, id_to_label = load_saved_model(
        model_dir
    )

    print(f"Device: {device}")
    print(f"Model directory: {model_dir}")

    for dataset_name, file_path in evaluation_files.items():
        evaluate_dataset(
            dataset_name,
            file_path,
            tokenizer,
            model,
            device,
            id_to_label,
        )


if __name__ == "__main__":
    main()