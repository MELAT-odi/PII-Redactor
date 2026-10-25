
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("Model Evaluation")
st.caption("Performance of the Amharic PII detection model on the test dataset.")

col1, col2, col3 = st.columns(3)

col1.metric("Precision", "84.97%")
col2.metric("Recall", "84.43%")
col3.metric("F1 Score", "84.70%")

st.info(
    "These are previously recorded test-set metrics, "
    "not measurements of the current input text."
)

st.divider()
st.subheader("Detected Entities")

entities = st.session_state.get("accepted_entities", [])

if entities:
    table_data = [
        {
            "Category": entity["label"],
            "Detected Text": entity["value"],
            "Start": entity["start"],
            "End": entity["end"],
        }
        for entity in entities
    ]

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write(
        "No detection results are available yet. "
        "Go to Detect & Redact and scan some text first."
    )
