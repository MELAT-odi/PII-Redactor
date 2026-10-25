from src.reversible_redaction import (
    generate_encryption_key,
    reversible_redact,
    restore_text,
)


def test_reversible_redaction_restores_original_text():
    original_text = "Call me at 0912345678."
    detections = [
        {
            "start": 11,
            "end": 21,
            "label": "PHONE",
        }
    ]

    encryption_key = generate_encryption_key()

    redacted_text, vault = reversible_redact(
        original_text,
        detections,
        encryption_key,
    )

    restored_text = restore_text(
        redacted_text,
        vault,
        encryption_key,
    )

    assert "[PHONE_1]" in redacted_text
    assert "0912345678" not in redacted_text
    assert restored_text == original_text
