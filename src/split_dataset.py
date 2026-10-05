import json
import random
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

input_file = project_root/"data"/"synthetic"/"bio_dataset.jsonl"

train_file = project_root /"data"/ "synthetic"/ "train.jsonl"
test_file = project_root/ "data"/ "synthetic" /"test.jsonl"

def load_records(file_path):
    records =[]

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

def save_records(records, file_path):

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                +"\n"
            )

def split_dataset(records, train_ratio=0.8):

    random.seed(42)

    shuffled_records = records.copy()

    random.shuffle(
        shuffled_records
    )

    train_size = int(
        len(shuffled_records)* train_ratio
    )

    train_records =shuffled_records[
        :train_size
    ]

    test_records = shuffled_records[
        train_size:
    ]

    return train_records, test_records

def main():

    records =load_records(
        input_file
    )

    train_records,test_records =split_dataset(
        records
    )

    save_records(
        train_records,
        train_file
    )

    save_records(
        test_records,
        test_file
    )

    print("Dataset split completed.")
    print()
    print(f"Total records: {len(records)}")
    print(f"Training records: {len(train_records)}")
    print(f"Test records: {len(test_records)}")
    print()
    print(f"Training file: {train_file}")
    print(f"Test file: {test_file}")


if __name__ == "__main__":
    main()   

