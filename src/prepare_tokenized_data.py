import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


project_root = Path(__file__).resolve().parent.parent

train_file = project_root / "data" / "synthetic" / "train.jsonl"
test_file = project_root / "data" / "synthetic" / "test.jsonl"


model_name = "xlm-roberta-base"


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


def main():

    dataset = load_dataset()

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    tokenized_dataset = dataset.map(
        lambda examples:
            tokenize_and_align_labels(
                examples,
                tokenizer,
                label_to_id
            ),
        batched=True
    )

    print(
        "Tokenization completed."
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Training records: "
        f"{len(tokenized_dataset['train'])}"
    )

    print(
        f"Test records: "
        f"{len(tokenized_dataset['test'])}"
    )

    print(
        f"Number of labels: "
        f"{len(label_list)}"
    )

    print()

    print(
        "Original tokens:"
    )

    print(
        dataset["train"][0]["tokens"]
    )

    print()

    print(
        "Transformer tokens:"
    )

    print(
        tokenizer.convert_ids_to_tokens(
            tokenized_dataset["train"][0][
                "input_ids"
            ]
        )
    )

    print()

    print(
        "Aligned labels:"
    )

    print(
        tokenized_dataset["train"][0][
            "labels"
        ]
    )


if __name__ == "__main__":
    main()