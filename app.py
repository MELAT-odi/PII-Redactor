import json
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.reversible_redaction import (
    generate_encryption_key,
    reversible_redact,
)


# -------------------- Project configuration --------------------

model_name = "Melat-odi/amharic-pii-detector"

st.set_page_config(
    page_title="Amharic PII Detector",
    page_icon="🔐",
    layout="wide",
)


# -------------------- Load trained model --------------------

@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name
    )

    model.to(device)
    model.eval()

    id_to_label = {
        int(key): value
        for key, value in model.config.id2label.items()
    }

    return tokenizer, model, device, id_to_label


tokenizer, model, device, id_to_label = load_model()

# -------------------- Rule-based detection --------------------

def detect_phone_and_birth_date(text):
    detections = []

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


# -------------------- Model-based PII detection --------------------

def detect_pii(text, tokenizer, model, device, id_to_label):
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


# -------------------- Filter known false positives --------------------

def filter_false_positives(text, detections):
    filtered_detections = []

    for entity in detections:
        start = entity["start"]
        label = entity["label"]

        context_before = text[max(0, start - 60):start]

        is_receipt_number = (
            label == "TIN"
            and "ደረሰኝ ቁጥር" in context_before
        )

        is_event_date = (
            label == "DATE_OF_BIRTH"
            and (
                "ማረጋገጫው ቀን" in context_before
                or "ማረጋገጫ ቀን" in context_before
            )
        )

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


# -------------------- Irreversible redaction --------------------

def redact_text(text, detections):
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
            accepted.append(entity.copy())

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


# -------------------- Initialize session state --------------------

if "accepted_entities" not in st.session_state:
    st.session_state["accepted_entities"] = []

if "redacted_text" not in st.session_state:
    st.session_state["redacted_text"] = ""

if "redaction_completed" not in st.session_state:
    st.session_state["redaction_completed"] = False

if "reversible_vault" not in st.session_state:
    st.session_state["reversible_vault"] = None

if "reversible_key" not in st.session_state:
    st.session_state["reversible_key"] = None


# -------------------- Main interface --------------------

st.title("🔐 Amharic PII Detector and Redactor")

st.write(
    "Detect personally identifiable information (PII) "
    "in Amharic or English text and protect it through redaction."
)

st.info(
    "This Amharic PII Detector and Redactor uses a trained "
    "Named Entity Recognition (NER) model, supported by "
    "rule-based detection, to identify personally identifiable "
    "information (PII) in text.\n\n"
    "**Irreversible Redaction:** Replaces detected personal "
    "information with category labels, such as `[PHONE]`.\n\n"
    "**Reversible Redaction:** Replaces detected information "
    "with unique placeholders and stores the original values "
    "in an encrypted vault that can be restored using the "
    "corresponding key.\n\n"
    "**Important:** Detection is not guaranteed to be complete "
    "or error-free. Some sensitive information may be missed, "
    "and some text may be incorrectly identified. For "
    "demonstrations and testing, use synthetic or non-sensitive "
    "sample text."
)


# Load the model
try:

    st.success(f"Model loaded successfully on {device}.")
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# -------------------- Persistent input text --------------------

sample_text = (
    "ስም: አበበ ከበደ። "
    "ስልክ: 0912345678። "
    "የትውልድ ቀኔ 14 ሚያዝያ 1991 ነው። "
    "ኢሜይል: abebe@example.com"
)

if "saved_input_text" not in st.session_state:
    st.session_state["saved_input_text"] = sample_text

if "input_text" not in st.session_state:
    st.session_state["input_text"] = st.session_state["saved_input_text"]


def save_input_text():
    st.session_state["saved_input_text"] = st.session_state["input_text"]

    # Mark the old result as outdated when the input changes.
    st.session_state["redaction_completed"] = False


text = st.text_area(
    "Enter text to scan",
    height=200,
    placeholder="Type or paste Amharic or English text here...",
    key="input_text",
    on_change=save_input_text,
)


# 2. Redaction mode directly below the text box
redaction_mode = st.radio(
    "Choose redaction mode",
    ["Irreversible", "Reversible"],
    horizontal=True,
    key="redaction_mode",
)


# 3. Detect and redact button
if st.button("Detect and redact", type="primary", use_container_width=True):
    text = st.session_state["input_text"]
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Detecting and redacting PII..."):
                detections = detect_pii(
                    text,
                    tokenizer,
                    model,
                    device,
                    id_to_label,
                )

                detections.extend(
                    detect_phone_and_birth_date(text)
                )

                detections = filter_false_positives(
                    text,
                    detections,
                )

                # Resolve overlaps before either redaction mode.
                _, accepted = redact_text(text, detections)

                st.session_state["accepted_entities"] = accepted

                if redaction_mode == "Irreversible":
                    redacted_text, accepted = redact_text(
                        text,
                        accepted,
                    )

                    st.session_state["redacted_text"] = redacted_text
                    st.session_state["reversible_vault"] = None
                    st.session_state["reversible_key"] = None

                else:
                    encryption_key = generate_encryption_key()

                    redacted_text, vault = reversible_redact(
                        text,
                        accepted,
                        encryption_key,
                    )

                    st.session_state["redacted_text"] = redacted_text
                    st.session_state["reversible_vault"] = vault
                    st.session_state["reversible_key"] = encryption_key

                st.session_state["redaction_completed"] = True

            st.success("PII detection and redaction completed.")

        except Exception as error:
            st.session_state["redaction_completed"] = False
            st.error(f"Processing failed: {error}")


# -------------------- Redacted output --------------------

if st.session_state["redaction_completed"]:
    st.divider()
    st.subheader("Redacted Text")

    # The output is stored in session state, so it remains visible
    # when Streamlit reruns the application.
    st.text_area(
        "Protected output",
        value=st.session_state["redacted_text"],
        height=180,
        disabled=True,
    )



if st.session_state["reversible_vault"] is not None:
    st.divider()
    st.subheader("🔐 Encrypted Vault and Key")

    vault_json = json.dumps(
        st.session_state["reversible_vault"],
        ensure_ascii=False,
        indent=2,
    )

    encryption_key = st.session_state["reversible_key"].decode("utf-8")

    st.markdown("**Encrypted vault**")

    with st.expander("View encrypted vault", expanded=False):
        st.code(vault_json, language="json")

    st.download_button(
        "⬇️ Download encrypted vault",
        data=vault_json,
        file_name="encrypted_pii_vault.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("**Encryption key**")

    if "show_encryption_key" not in st.session_state:
        st.session_state["show_encryption_key"] = False

    st.text_input(
        "Copy the key from this field",
        value=encryption_key,
        type=(
            "default"
            if st.session_state["show_encryption_key"]
            else "password"
        ),
        disabled=False,
        key="encryption_key_display",
    )

    st.checkbox(
        "Show encryption key",
        key="show_encryption_key",
    )

    st.download_button(
        "⬇️ Download encryption key",
        data=encryption_key,
        file_name="pii_encryption_key.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.warning(
        "Keep the key separate from the vault. Anyone who obtains both "
        "may be able to recover the original personal information."
    )



    if not st.session_state["accepted_entities"]:
        st.warning(
            "No entities were detected. This does not guarantee "
            "that the text contains no PII."
        )

    st.caption(
        "Use the Restore Text page to restore protected text in reversible mode. "
        "The Model Evaluation page displays detected entities and test-set metrics."
    )
