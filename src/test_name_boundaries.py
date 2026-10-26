import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from spans_to_bio import convert_record


def test_prefix_is_part_of_name():
    text = "የሺ ተስፋዬ"
    name = "የሺ ተስፋዬ"
    start = text.index(name)

    record = {
        "id": "boundary_test_1",
        "split": "test",
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

    assert result["tokens"] == ["የሺ", "ተስፋዬ"]
    assert result["labels"] == ["B-PERSON_NAME", "I-PERSON_NAME"]


def test_prefix_is_outside_name():
    text = "የሳራ ተስፋዬ"
    name = "ሳራ ተስፋዬ"
    start = text.index(name)

    record = {
        "id": "boundary_test_2",
        "split": "test",
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

    assert result["tokens"] == ["የ", "ሳራ", "ተስፋዬ"]
    assert result["labels"] == [
        "O",
        "B-PERSON_NAME",
        "I-PERSON_NAME",
    ]
