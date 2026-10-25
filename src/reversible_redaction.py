from pathlib import Path
import json

from cryptography.fernet import Fernet


def generate_encryption_key():
    """Generate a new Fernet encryption key."""
    return Fernet.generate_key()


def reversible_redact(text, detections, encryption_key):
    """Replace PII with numbered placeholders and encrypt original values."""
    fernet = Fernet(encryption_key)

    accepted = []

    for entity in sorted(
        detections,
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
        ),
    ):
        start = entity["start"]
        end = entity["end"]

        if start < 0 or end > len(text) or start >= end:
            continue

        overlaps = any(
            start < existing["end"]
            and end > existing["start"]
            for existing in accepted
        )

        if not overlaps:
            accepted.append(entity)

    accepted.sort(key=lambda item: item["start"])

    redacted_text = text
    vault = {}
    entity_counts = {}

    for entity in reversed(accepted):
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]
        original_value = text[start:end]

        entity_counts[label] = entity_counts.get(label, 0) + 1
        placeholder = f"[{label}_{entity_counts[label]}]"

        encrypted_value = fernet.encrypt(
            original_value.encode("utf-8")
        ).decode("utf-8")

        vault[placeholder] = {
            "label": label,
            "encrypted_value": encrypted_value,
        }

        redacted_text = (
            redacted_text[:start]
            + placeholder
            + redacted_text[end:]
        )

    return redacted_text, vault


def restore_text(redacted_text, vault, encryption_key):
    """Restore original values using the encrypted vault."""
    fernet = Fernet(encryption_key)
    restored_text = redacted_text

    for placeholder, item in vault.items():
        original_value = fernet.decrypt(
            item["encrypted_value"].encode("utf-8")
        ).decode("utf-8")

        restored_text = restored_text.replace(
            placeholder,
            original_value,
        )

    return restored_text


def save_vault(vault, vault_path):
    """Save encrypted PII values to a JSON vault file."""
    path = Path(vault_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(vault, file, ensure_ascii=False, indent=2)


def load_vault(vault_path):
    """Load an encrypted PII vault from a JSON file."""
    with Path(vault_path).open("r", encoding="utf-8") as file:
        return json.load(file)