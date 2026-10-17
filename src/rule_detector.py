
import re


# Match exactly 12 digits, optionally separated by spaces or hyphens.
national_id_pattern = re.compile(
    r"(?<![0-9])(?:[0-9][ -]?){11}[0-9](?![0-9])"
)


def detect_national_ids(text):
    """
    Find potential 12-digit FIN values in text.

    Returns a list of dictionaries with the detected value
    and its character offsets in the original text.
    """
    detections = []

    for match in national_id_pattern.finditer(text):
        detections.append({
            "label": "NATIONAL_ID",
            "value": match.group(),
            "start": match.start(),
            "end": match.end(),
        })

    return detections
