import json
import re
from pathlib import Path


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "raw_annotated.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "bio_dataset.jsonl"
)


# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

TOKEN_PATTERN = re.compile(
    r"\S+|[።፣፤፥፦፧፨,.;:!?()\[\]{}]"
)


def tokenize_with_spans(text):
    """
    Split text into tokens while keeping each token's
    original character start and end positions.
    """

    tokens = []

    for match in TOKEN_PATTERN.finditer(text):

        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    return tokens


# ---------------------------------------------------------
# BIO CONVERSION
# ---------------------------------------------------------

def convert_record(record):

    text = record["text"]
    entities = record["entities"]

    tokens = tokenize_with_spans(text)

    labels = ["O"] * len(tokens)

    for entity in entities:

        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_label = entity["label"]

        matching_tokens = []

        for i, token in enumerate(tokens):

            token_start = token["start"]
            token_end = token["end"]

            # Token overlaps entity
            if (
                token_start < entity_end
                and token_end > entity_start
            ):
                matching_tokens.append(i)

        if not matching_tokens:

            raise ValueError(
                f"Entity could not be matched:\n"
                f"{entity}\n"
                f"Text: {text}"
            )

        # First token = B-
        first_token = matching_tokens[0]

        labels[first_token] = f"B-{entity_label}"

        # Remaining tokens = I-
        for token_index in matching_tokens[1:]:

            labels[token_index] = f"I-{entity_label}"

    return {
        "id": record["id"],
        "tokens": [
            token["token"]
            for token in tokens
        ],
        "labels": labels
    }


# ---------------------------------------------------------
# DATASET CONVERSION
# ---------------------------------------------------------

def convert_dataset():

    output_records = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(line)

            converted = convert_record(record)

            output_records.append(converted)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for record in output_records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Converted {len(output_records)} records."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    convert_dataset()