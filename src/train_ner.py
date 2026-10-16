import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "synthetic"
model_dir = project_root / "models" / "ner_model"

model_name = "xlm-roberta-base"


def load_records(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_split(split_name):
    file_path = data_dir / f"{split_name}.jsonl"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing {split_name} dataset: {file_path}"
        )

    records = load_records(file_path)

    for record in records:
        if len(record["tokens"]) != len(record["labels"]):
            raise ValueError(
                f"Token/label mismatch in {split_name}, "
                f"record {record['id']}"
            )

    return records


def main():
    train_records = load_split("train")
    validation_records = load_split("validation")
    test_records = load_split("test")

    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print(f"Test records: {len(test_records)}")

    label_list = sorted({
        label
        for record in train_records
        for label in record["labels"]
    })

    label_to_id = {
        label: index
        for index, label in enumerate(label_list)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    for split_name, records in [
        ("validation", validation_records),
        ("test", test_records),
    ]:
        unseen_labels = {
            label
            for record in records
            for label in record["labels"]
            if label not in label_to_id
        }

        if unseen_labels:
            raise ValueError(
                f"Unknown labels in {split_name}: "
                f"{sorted(unseen_labels)}"
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=256,
        )

        aligned_labels = []

        for batch_index, labels in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=batch_index
            )

            previous_word_id = None
            label_ids = []

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label_to_id[labels[word_id]])
                else:
                    label_ids.append(-100)

                previous_word_id = word_id

            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels

        return tokenized_inputs

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(validation_records),
        "test": Dataset.from_list(test_records),
    })

    tokenized_dataset = dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )

    print("\nTokenization completed.")
    print(f"Number of labels: {len(label_list)}")
    print("Validation set will be used during training.")
    print("Test set will remain unused until final evaluation.")

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    def compute_metrics(prediction):
        logits, label_ids = prediction
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for predicted_ids, expected_ids in zip(
            predictions, label_ids
        ):
            record_predictions = []
            record_labels = []

            for predicted_id, expected_id in zip(
                predicted_ids, expected_ids
            ):
                if expected_id == -100:
                    continue

                record_predictions.append(
                    id_to_label[int(predicted_id)]
                )
                record_labels.append(
                    id_to_label[int(expected_id)]
                )

            true_predictions.append(record_predictions)
            true_labels.append(record_labels)

        return {
            "precision": precision_score(
                true_labels, true_predictions
            ),
            "recall": recall_score(
                true_labels, true_predictions
            ),
            "f1": f1_score(
                true_labels, true_predictions
            ),
        }

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting model training...")
    trainer.train()

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print(f"\nBest model saved to: {model_dir}")
    print("Training completed successfully.")
    print("The test set has not been evaluated.")


if __name__ == "__main__":
    main()