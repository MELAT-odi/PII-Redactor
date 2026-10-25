import json
import streamlit as st

from src.reversible_redaction import restore_text

st.set_page_config(
    page_title="Restore Text",
    page_icon="🔑",
    layout="wide",
)

# Keep restoration inputs in session state across page navigation.
if "restore_saved_text" not in st.session_state:
    st.session_state["restore_saved_text"] = ""

if "restore_saved_vault" not in st.session_state:
    st.session_state["restore_saved_vault"] = ""

if "restore_saved_key" not in st.session_state:
    st.session_state["restore_saved_key"] = ""

st.title("Restore Original Text")
st.caption("Restore protected PII using the matching encrypted vault and key.")

redacted_text = st.text_area(
    "Redacted text",
    height=180,
    placeholder="Paste text containing placeholders such as [PHONE_1]...",
    value=st.session_state["restore_saved_text"],
    key="restore_text_widget",
)

vault_data = st.text_area(
    "Encrypted vault (JSON)",
    height=180,
    placeholder="Paste the encrypted vault JSON here...",
    value=st.session_state["restore_saved_vault"],
    key="restore_vault_widget",
)

encryption_key = st.text_input(
    "Encryption key",
    type="password",
    placeholder="Paste your encryption key here...",
    value=st.session_state["restore_saved_key"],
    key="restore_key_widget",
)

# Save the latest values for restoration.
st.session_state["restore_saved_text"] = redacted_text
st.session_state["restore_saved_vault"] = vault_data
st.session_state["restore_saved_key"] = encryption_key

if st.button("Restore Text", type="primary"):
    if not redacted_text.strip():
        st.error("Please enter the redacted text.")
    elif not vault_data.strip():
        st.error("Please provide the encrypted vault.")
    elif not encryption_key.strip():
        st.error("Please enter your encryption key.")
    else:
        try:
            parsed_vault = json.loads(vault_data)
            encrypted_vault = parsed_vault.get(
                "encrypted_vault", parsed_vault
            )

            restored_text = restore_text(
                redacted_text,
                encrypted_vault,
                encryption_key.encode("utf-8"),
            )

            st.success("Text restored successfully.")
            st.text_area(
                "Restored original text",
                value=restored_text,
                height=200,
            )
        except Exception:
            st.error(
                "Restoration failed. Check that the redacted text, "
                "encrypted vault, and encryption key match."
            )
