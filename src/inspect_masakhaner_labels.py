
from datasets import load_dataset, ClassLabel

def main():
    dataset_files = {
        split_name: (
            "https://huggingface.co/datasets/masakhane/masakhaner/"
            "resolve/refs%2Fconvert%2Fparquet/amh/"
            f"{split_name}/0000.parquet"
        )
        for split_name in ["train", "validation", "test"]
    }

    dataset = load_dataset("parquet", data_files=dataset_files)

    print("Dataset features:")
    print(dataset["train"].features)

    tag_feature = dataset["train"].features["ner_tags"]

    if isinstance(tag_feature.feature, ClassLabel):
        print("\nLabel mapping:")
        for label_id, label_name in enumerate(tag_feature.feature.names):
            print(f"{label_id}: {label_name}")
    else:
        print("\nCould not automatically identify the label names.")
        print("NER tag feature:", tag_feature)

if __name__ == "__main__":
    main()
