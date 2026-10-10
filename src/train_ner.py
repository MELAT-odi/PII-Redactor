import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import(
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)


project_root= Path(__file__).resolve().parent.parent

train_file = project_root/"data" / "synthetic" / "train.jsonl"
test_file = project_root / "data" / "synthetic" / "test.jsonl"

model_name = "xlm-roberta-base"

output_dir =project_root/"models"/"ner_model"

def load_records(file_path):
    records = []

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

def load_dataset():

    train_records = load_records(
        train_file
    )

    test_records = load_records(
        test_file
    )

    dataset = DatasetDict({
        "train": Dataset.from_list(
            train_records
        ),
        "test": Dataset.from_list(
            test_records
        )
    })

    return dataset

def create_label_mapping(dataset):

    label_set = set()

    for record in dataset["train"]:

        for label in record["labels"]:
            label_set.add(label)

    label_list = sorted(
        label_set
    )

    label_to_id = {
        label: index
        for index, label in enumerate(
            label_list
        )
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    return label_list, label_to_id, id_to_label


def tokenize_and_align_labels(
    examples,
    tokenizer,
    label_to_id
):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    all_labels = []

    for index, labels in enumerate(
        examples["labels"]
    ):

        word_ids = tokenized_inputs.word_ids(
            batch_index=index
        )

        previous_word_id = None

        label_ids = []

        for word_id in word_ids:

            if word_id is None:

                label_ids.append(-100)

            elif word_id != previous_word_id:

                label_ids.append(
                    label_to_id[
                        labels[word_id]
                    ]
                )

            else:

                label_ids.append(-100)

            previous_word_id = word_id

        all_labels.append(
            label_ids
        )

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


def create_model(
        label_list,
        label_to_id,
        id_to_label
):

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        label2id=label_to_id,
        id2label=id_to_label
    )

    return model

def create_data_collator(tokenizer):
    data_collator =DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    return data_collator

def create_training_arguments():

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none"
    )

    return training_arguments

def create_trainer(
    model,
    tokenizer,
    training_arguments,
    tokenized_dataset,
    data_collator
):

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator
    )

    return trainer

def main():

    print("Loading dataset...")

    dataset = load_dataset()

    label_list, label_to_id, id_to_label = (
        create_label_mapping(dataset)
    )

    print(
        f"Training records: "
        f"{len(dataset['train'])}"
    )

    print(
        f"Test records: "
        f"{len(dataset['test'])}"
    )

    print(
        f"Number of labels: "
        f"{len(label_list)}"
    )

    print()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    print("Tokenizing dataset...")

    tokenized_dataset = dataset.map(
        lambda examples:
            tokenize_and_align_labels(
                examples,
                tokenizer,
                label_to_id
            ),
        batched=True
    )

    print("Loading model...")

    model = create_model(
        label_list,
        label_to_id,
        id_to_label
    )

    data_collator = create_data_collator(
        tokenizer
    )

    training_arguments = create_training_arguments()

    trainer = create_trainer(
        model,
        tokenizer,
        training_arguments,
        tokenized_dataset,
        data_collator
    )

    print()
    print("Starting training...")

    trainer.train()

    print()
    print("Training completed.")

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    print(
        f"Model saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()