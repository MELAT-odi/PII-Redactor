
from rule_detector import detect_national_ids


def test_detects_twelve_digit_fin():
    text = "የፋይዳ ቁጥር 123456789012 ነው።"

    results = detect_national_ids(text)

    assert len(results) == 1
    assert results[0]["value"] == "123456789012"
    assert results[0]["label"] == "NATIONAL_ID"


def test_detects_grouped_fin():
    text = "FIN: 1234-5678-9012"

    results = detect_national_ids(text)

    assert len(results) == 1
    assert results[0]["value"] == "1234-5678-9012"


def test_does_not_detect_ten_digit_number():
    assert detect_national_ids("Phone: 0912345678") == []


def test_does_not_detect_thirteen_digit_number():
    assert detect_national_ids("1234567890123") == []


def test_does_not_detect_twelve_digits_inside_longer_number():
    assert detect_national_ids("991234567890123") == []


def test_returns_correct_offsets():
    text = "FIN: 123456789012"

    result = detect_national_ids(text)[0]

    assert text[result["start"]:result["end"]] == result["value"]


if __name__ == "__main__":
    tests = [
        test_detects_twelve_digit_fin,
        test_detects_grouped_fin,
        test_does_not_detect_ten_digit_number,
        test_does_not_detect_thirteen_digit_number,
        test_does_not_detect_twelve_digits_inside_longer_number,
        test_returns_correct_offsets,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"\nAll {len(tests)} tests passed.")
