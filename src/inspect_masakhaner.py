from datasets import load_dataset

def main():
    print("Loading MasakhaNER Amharic dataset...")

    dataset_files = {
        split_name: (
            "https://huggingface.co/datasets/masakhane/masakhaner/"
            "resolve/refs%2Fconvert%2Fparquet/amh/"
            f"{split_name}/0000.parquet"
        )
        for split_name in ["train", "validation", "test"]
    }

    dataset = load_dataset(
        "parquet",
        data_files=dataset_files
    )

    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Number of examples: {len(split_data)}")
        print(f"Columns: {split_data.column_names}")

        if len(split_data) > 0:
            example = split_data[0]
            print("\nFirst example:")
            print("Tokens:", example["tokens"])
            print("NER tags:", example["ner_tags"])

if __name__ == "__main__":
    main()