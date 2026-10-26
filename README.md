# Amharic PII Detector and Redactor

An NLP-based application that detects and redacts Personally Identifiable Information (PII) in Amharic text using a fine-tuned XLM-RoBERTa model and rule-based detection.

**Live demo:** [Open the Amharic PII Detector](https://pii-redactor-j3mvgg9j5yj6hqhjdz2euz.streamlit.app/)

**Source code:** [GitHub Repository](https://github.com/MELAT-odi/PII-Redactor)

> **Note:** This is an educational MVP. Detection is not perfect, so review redacted text before sharing it.

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

The application demonstrates how automated PII detection and redaction can support privacy-aware text processing.

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

Detection performance varies by category. TIN and National ID detection currently require further improvement.

## 4. Technologies Used

- **Python** — application logic and data processing
- **Streamlit** — interactive web interface
- **PyTorch** — model inference
- **Hugging Face Transformers** — XLM-RoBERTa token classification
- **Regular expressions** — structured PII detection
- **Cryptography (Fernet)** — encryption for reversible redaction
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

### Irreversible Mode

Detected values are replaced with category-specific placeholders, for example:

- `[PERSON_NAME]`
- `[PHONE]`
- `[EMAIL]`

The original values are not intended to be recoverable from the redacted output alone.

### Reversible Mode

Detected values are replaced with numbered placeholders, allowing the original text to be restored using the corresponding encrypted vault and encryption key.

**Security note:** Protect the encryption key and vault carefully. Anyone with the required restoration materials may be able to recover sensitive values. Do not use real confidential data in an unsecured demo.

## 7. Model Evaluation

The fine-tuned model was evaluated on a held-out synthetic test set containing 400 records.

| Metric | Score |
|---|---:|
| Entity-level precision | 84.97% |
| Entity-level recall | 84.43% |
| Entity-level F1-score | 84.70% |
| Test records evaluated | 400 |

These metrics describe the performance of the currently evaluated saved model on the synthetic test set. The exact record match rate was 67.50%; this measures records whose complete predicted entity sequences matched the reference sequences and is different from entity-level F1-score.

### Performance by PII Category

| Category | F1-score |
|---|---:|
| `ADDRESS_PLACE` | 0.7324 |
| `BANK_ACCOUNT` | 0.7015 |
| `DATE_OF_BIRTH` | 1.0000 |
| `EMAIL` | 1.0000 |
| `NATIONAL_ID` | 0.2000 |
| `PERSON_NAME` | 0.9597 |
| `PHONE` | 0.9368 |
| `TIN` | 0.0000 |

The results show that performance differs across categories. In this evaluation, TIN detection had an F1-score of 0.00 and National ID detection had an F1-score of 0.20. These categories need additional data, error analysis, and model improvement.

The dataset is synthetic, so these results should not be interpreted as proof of performance on real-world Amharic documents.

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

This structure describes the main project components. Some local files and datasets may not be included in the Git repository.

## 9. Run Locally

### Prerequisites

- Python 3.12 recommended
- Git
- Internet access to install dependencies and obtain the trained model

### Step 1: Clone the Repository

```bash
git clone https://github.com/MELAT-odi/PII-Redactor.git
cd PII-Redactor
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Obtain the Trained Model

The trained model files are excluded from the Git repository by `.gitignore` because the model weights are large.

Before launching the application, obtain the trained model separately from the project's designated model repository or follow the model setup instructions provided by the project author.

The application currently expects the model at:

```text
models/colab_best_model/
```

The directory must contain the required configuration, model weights, and tokenizer files, including files such as:

- `config.json`
- `model.safetensors`
- `sentencepiece.bpe.model`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

The model must be compatible with the Transformers version used by the application.

**Note:** A public model download link and automated download instructions will be added after model hosting is configured.

### Step 5: Start the Application

From the project root, run:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal by Streamlit.

## 10. Testing

Run the automated tests from the project root:

```bash
python -m pytest tests -v
```

The available tests cover implemented functionality. Review the test results before relying on the application for sensitive workflows.

## 11. Model Evaluation Script

The project includes an evaluation script:

```bash
python src/evaluate_ner.py
```

Ensure the trained model and expected test dataset are available at the paths configured by the script.

The evaluation uses entity-level precision, recall, and F1-score to assess the quality of PII detection.

## 12. Limitations

- The training and evaluation data are synthetic and may not represent the full diversity of real-world Amharic text.
- TIN and National ID detection currently require improvement.
- Names, addresses, unusual number formats, and unseen patterns may be missed or incorrectly identified.
- Results on synthetic data may not generalize to real documents.
- Redaction quality depends on the accuracy of the detection pipeline.
- The application is an educational prototype, not a production-certified privacy solution.

A successful redaction operation does not guarantee that all PII has been removed.

## 13. Future Improvements

- Increase the diversity and quality of annotated training data.
- Improve detection of TIN, National ID, and address information.
- Expand evaluation using carefully annotated examples.
- Improve error handling and user feedback.
- Strengthen key management and security testing for reversible redaction.
- Host the trained model and simplify model setup.
- Deploy the application for public demonstration.

## 14. Privacy and Responsible Use

- Use synthetic information when testing the application.
- Avoid entering real personal, medical, financial, or confidential information into an unsecured deployment.
- Never commit encryption keys or decrypted vault contents to GitHub.
- Protect reversible-redaction keys and encrypted vaults.
- Review redacted text before sharing it because missed detections may leave sensitive information exposed.

## 15. Repository

**Project:** Amharic PII Detector and Redactor

**GitHub:** [MELAT-odi/PII-Redactor](https://github.com/MELAT-odi/PII-Redactor)

**Live demo:** [Open the application](https://pii-redactor-j3mvgg9j5yj6hqhjdz2euz.streamlit.app/)
