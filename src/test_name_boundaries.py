
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from spans_to_bio import convert_record


def test_name_boundary(text, name):
    start = text.index(name)

    record = {
        "id": "boundary_test",
        "text": text,
        "entities": [
            {
                "start": start,
                "end": start + len(name),
                "label": "PERSON_NAME",
                "text": name,
            }
        ],
    }

    result = convert_record(record)

    print(f"\nText: {text}")
    print(f"Name: {name}")

    for token, label in zip(result["tokens"], result["labels"]):
        print(f"  {token!r}: {label}")


# Case 1: የ is part of the name.
test_name_boundary("የሺ ተስፋዬ", "የሺ ተስፋዬ")

# Case 2: የ is outside the annotated name.
test_name_boundary("የሳራ ተስፋዬ", "ሳራ ተስፋዬ")
