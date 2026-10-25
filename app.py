
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


project_root = Path(__file__).resolve().parent
model_dir = project_root / "models" / "colab_best_model"


st.set_page_config(
    page_title="Amharic PII Detector",
    page_icon="🔐",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model folder not found: {model_dir}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir)
    )

    model.to(device)
    model.eval()

    id_to_label = {
        int(key): value
        for key, value in model.config.id2label.items()
    }

    return tokenizer, model, device, id_to_label


def detect_phone_and_birth_date(text):
    detections = []

    # Match Ethiopian mobile numbers, including Amharic prefixes.
    phone_pattern = re.compile(
        r"(?<!\d)(?:\+251[\s-]?9\d{2}[\s-]?\d{3}[\s-]?\d{3}|09\d{8})(?!\d)"
    )

    for match in phone_pattern.finditer(text):
        detections.append({
            "label": "PHONE",
            "value": match.group(),
            "start": match.start(),
            "end": match.end(),
        })

    # Match dates following an explicit Amharic birth-date label.
    birth_date_pattern = re.compile(
        r"የትውልድ\s*ቀን\w*\s*[:：]?\s*"
        r"(\d{1,2}\s+[\u1200-\u137F]+\s+\d{4})"
    )

    for match in birth_date_pattern.finditer(text):
        detections.append({
            "label": "DATE_OF_BIRTH",
            "value": match.group(1),
            "start": match.start(1),
            "end": match.end(1),
        })

    return detections


def detect_pii(text, tokenizer, model, device, id_to_label):
    # Preserve the original character positions for every word.
    words = []

    for match in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE):
        words.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
        })

    if not words:
        return []

    if len(words) > 180:
        raise ValueError(
            "Please shorten the text to 180 tokens or fewer "
            "for this demo."
        )

    word_texts = [word["text"] for word in words]

    encoded = tokenizer(
        word_texts,
        is_split_into_words=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    word_ids = encoded.word_ids(batch_index=0)

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    predicted_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    # Use the prediction for the first subword of each word.
    word_predictions = {}

    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_predictions:
            word_predictions[word_id] = id_to_label[
                predicted_ids[token_index]
            ]

    detections = []
    current_entity = None

    def save_entity():
        if current_entity is not None:
            detections.append(current_entity.copy())

    for word_index, word in enumerate(words):
        label = word_predictions.get(word_index, "O")

        if label == "O" or "-" not in label:
            save_entity()
            current_entity = None
            continue

        prefix, category = label.split("-", 1)

        if prefix == "B" or (
            current_entity is not None
            and current_entity["label"] != category
        ):
            save_entity()

            current_entity = {
                "label": category,
                "start": word["start"],
                "end": word["end"],
            }

        elif prefix == "I" and current_entity is None:
            current_entity = {
                "label": category,
                "start": word["start"],
                "end": word["end"],
            }

        else:
            current_entity["end"] = word["end"]

    save_entity()

    for entity in detections:
        entity["value"] = text[
            entity["start"]:entity["end"]
        ]

    return detections


def filter_false_positives(text, detections):
    filtered_detections = []

    for entity in detections:
        start = entity["start"]
        label = entity["label"]

        context_before = text[max(0, start - 60):start]

        # Avoid labeling a clearly identified receipt number as TIN.
        is_receipt_number = (
            label == "TIN"
            and "ደረሰኝ ቁጥር" in context_before
        )

        # Avoid labeling a known event/confirmation date as a DOB.
        is_event_date = (
            label == "DATE_OF_BIRTH"
            and (
                "ማረጋገጫው ቀን" in context_before
                or "ማረጋገጫ ቀን" in context_before
            )
        )
                # Avoid labeling shipment tracking numbers as bank accounts.
        is_tracking_number = (
            label == "BANK_ACCOUNT"
            and (
                "መከታተያ ቁጥር" in context_before
                or "መከታተያ ቁጥር" in text[
                    entity["end"]:entity["end"] + 30
                ]
            )
        )

        if is_receipt_number or is_event_date or is_tracking_number:
            continue

        filtered_detections.append(entity)

    return filtered_detections


def redact_text(text, detections):
    # Prefer the longest detection when multiple detections overlap.
    accepted = []

    for entity in sorted(
        detections,
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
        ),
    ):
        overlaps = any(
            entity["start"] < existing["end"]
            and entity["end"] > existing["start"]
            for existing in accepted
        )

        if not overlaps:
            accepted.append(entity)

    # Replace right-to-left so original offsets remain valid.
    redacted = text

    for entity in sorted(
        accepted,
        key=lambda item: item["start"],
        reverse=True,
    ):
        redacted = (
            redacted[:entity["start"]]
            + f'[{entity["label"]}]'
            + redacted[entity["end"]:]
        )

    accepted.sort(key=lambda item: item["start"])

    return redacted, accepted


# -------------------- Streamlit interface --------------------

st.title("🔐 Amharic PII Detector and Redactor")

st.write(
    "Detect personally identifiable information (PII) in text "
    "and redact detected entities."
)

st.info(
    "This demo combines a trained NER model with rules for "
    "Ethiopian mobile numbers and explicitly labeled Amharic birth dates. "
    "Detection is not perfect. Some personal information may be missed, "
    "and some text may be incorrectly identified. Use synthetic examples."
)

try:
    tokenizer, model, device, id_to_label = load_model()
    st.success(f"Model loaded successfully on {device}.")
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


sample_text = (
    "ስም: አበበ ከበደ። "
    "ስልክ: 0912345678። "
    "የትውልድ ቀኔ 14 ሚያዝያ 1991 ነው። "
    "ኢሜይል: abebe@example.com"
)

text = st.text_area(
    "Enter text to scan",
    value=sample_text,
    height=200,
    placeholder="Type or paste Amharic or English text here...",
)

if st.button("Detect and redact", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")

    else:
        try:
            with st.spinner("Detecting PII..."):
                # Run the trained NER model.
                detections = detect_pii(
                    text,
                    tokenizer,
                    model,
                    device,
                    id_to_label,
                )

                # Add the targeted rule-based detections.
                rule_detections = detect_phone_and_birth_date(text)
                detections.extend(rule_detections)

                # Remove known contextual false positives.
                detections = filter_false_positives(
                    text,
                    detections,
                )

                # Redact detected spans.
                redacted_text, accepted = redact_text(
                    text,
                    detections,
                )

            st.subheader("Redacted text")

            st.text_area(
                "Irreversibly redacted output",
                value=redacted_text,
                height=180,
            )

            st.caption(
                "This mode replaces detected values with category "
                "placeholders and does not keep a restoration mapping."
            )

            st.subheader(f"Detected entities: {len(accepted)}")

            if accepted:
                st.dataframe(
                    [
                        {
                            "Category": item["label"],
                            "Detected text": item["value"],
                            "Start": item["start"],
                            "End": item["end"],
                        }
                        for item in accepted
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(
                    "No entities were detected. This does not guarantee "
                    "that the text contains no PII."
                )

        except Exception as error:
            st.error(f"Detection failed: {error}")
