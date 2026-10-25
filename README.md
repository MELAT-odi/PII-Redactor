# Amharic PII Detector and Redactor

An NLP-based application that detects and redacts Personally Identifiable Information (PII) in Amharic text using a fine-tuned XLM-RoBERTa model and rule-based detection.

**Try the live demo:** [Open the application](YOUR_DEPLOYED_DEMO_URL)

**Source code:** [GitHub Repository](https://github.com/MELAT-odi/PII-Redactor)

> The application is an educational MVP. Detection is not perfect, so review redacted text before sharing it.

## 1. Project Overview

Personal information can be exposed when text documents are shared, stored, or processed. This project explores how Natural Language Processing (NLP) can help identify and protect sensitive information in Amharic text.

The application combines a trained Named Entity Recognition (NER) model with rule-based detection and provides two redaction modes:

- **Irreversible redaction:** Replaces detected PII with category placeholders such as `[PERSON_NAME]` and `[PHONE]`.
- **Reversible redaction:** Replaces detected values with numbered placeholders and stores the original values in an encrypted vault so they can be restored using the appropriate encryption key.

### Example

**Input**

An Amharic sentence containing a person's name and phone number.

**Irreversibly redacted output**

The person's name is replaced with `[PERSON_NAME]`, and the phone number is replaced with `[PHONE]`.

The application is designed to demonstrate how automated PII detection and redaction can support privacy-aware text processing.

## 2. Objectives

- Detect sensitive information in Amharic text.
- Identify multiple PII categories using NLP and pattern-based rules.
- Provide irreversible and reversible redaction workflows.
- Evaluate NER performance using held-out test data.
- Provide an interactive interface for users to inspect detections and redacted text.

## 3. Supported PII Categories

| Category | Description |
|---|---|
| `PERSON_NAME` | Personal names |
| `PHONE` | Phone numbers |
| `EMAIL` | Email addresses |
| `ADDRESS_PLACE` | Place and address information |
| `DATE_OF_BIRTH` | Dates of birth |
| `NATIONAL_ID` | National identification numbers |
| `TIN` | Taxpayer Identification Numbers |
| `BANK_ACCOUNT` | Bank account numbers |

Detection performance varies by category. Some identifiers, particularly TIN and National ID, require further improvement.

## 4. Technologies Used

- **Python** — application logic and data processing
- **Streamlit** — interactive web interface
- **PyTorch** — model inference
- **Hugging Face Transformers** — XLM-RoBERTa token classification
- **Regular expressions** — structured PII detection
- **pytest** — automated testing
- **Git and GitHub** — version control

## 5. How It Works

1. The user enters Amharic text.
2. The detection pipeline processes the text using the trained NER model and supported rules.
3. Detected spans are assigned PII categories.
4. The user selects a redaction mode.
5. The application displays the redacted text.
6. In reversible mode, the original text can be restored using the encrypted vault and the correct key.

## 6. Redaction Modes

### Irreversible mode

Detected values are replaced with category-specific placeholders, for example:

- `[PERSON_NAME]`
- `[PHONE]`
- `[EMAIL]`

The original values are not intended to be recoverable from the redacted output alone.

### Reversible mode

Detected values are replaced with numbered placeholders, allowing the original text to be restored using the corresponding encrypted vault and encryption key.

**Security note:** Protect the encryption key and vault carefully. Anyone with the required restoration materials may be able to recover sensitive values. Do not use real confidential data in an unsecured demo.

## 7. Model Evaluation

The fine-tuned model was evaluated on a held-out synthetic test set.

| Metric | Score |
|---|---:|
| Precision | 84.97% |
| Recall | 84.43% |
| F1-score | 84.70% |
| Test records | 400 |

These are entity-level evaluation metrics for the current saved model and test set.

Performance differs by category. In the current evaluation, TIN detection had an F1-score of 0.00, while National ID detection had an F1-score of 0.20. These results indicate that further training-data improvements and model evaluation are necessary.

The dataset is synthetic, and these results should not be interpreted as proof of performance on real-world Amharic documents.

## 8. Project Structure

```text
pii_redactor/
├── app.py
├── pages/
│   ├── 1_Restore_Text.py
│   └── 2_Evaluation.py
├── src/
│   ├── generate_synthetic_data.py
│   ├── spans_to_bio.py
│   ├── split_dataset.py
│   ├── prepare_tokenized_data.py
│   ├── train_ner.py
│   ├── evaluate_ner.py
│   └── reversible_redaction.py
├── data/
│   └── synthetic/
├── models/
│   └── colab_best_model/
├── tests/
├── .streamlit/
├── requirements.txt
├── .gitignore
└── README.md
```

Additional project scripts and data files may be present in the repository.

## 9. Run Locally

### Prerequisites

- Python 3.12 recommended
- Git
- The trained model files required by the application

### Installation

Clone the repository:

```bash
git clone https://github.com/MELAT-odi/PII-Redactor.git
cd PII-Redactor
```

Create a virtual environment.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Ensure the trained model is available at `models/colab_best_model/` and contains the required tokenizer and model files.

### Start the application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal.

## 10. Testing

Run the automated tests from the project root:

```bash
python -m pytest tests -v
```

The available tests cover the functionality implemented in the repository. Review the test results before relying on the application for sensitive workflows.

## 11. Limitations

- The training data is synthetic and may not represent the full diversity of real-world Amharic text.
- TIN and National ID detection currently require improvement.
- Names, addresses, unusual number formats, and unseen patterns may be missed or incorrectly identified.
- Evaluation results on synthetic data may not generalize to real documents.
- Redaction quality depends on the accuracy of the detection pipeline.
- The application is an educational prototype, not a production-certified privacy solution.

A successful redaction operation does not guarantee that all PII has been removed.

## 12. Future Improvements

- Increase the diversity and quality of annotated training data.
- Improve detection of TIN, National ID, and address information.
- Expand evaluation with carefully annotated examples.
- Improve error handling and user feedback.
- Strengthen key management and security testing for reversible redaction.
- Evaluate performance on varied Amharic text sources.

## 13. Privacy and Responsible Use

Use synthetic information when testing the public demo. Avoid entering real personal, medical, financial, or confidential information into an unsecured deployment.

Review redacted text before sharing it, because missed detections may leave sensitive information exposed.

## 14. Repository

**Project:** Amharic PII Detector and Redactor  
**GitHub:** https://github.com/MELAT-odi/PII-Redactor
