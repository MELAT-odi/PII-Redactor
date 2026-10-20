# Amharic PII Detector and Redactor

An NLP-based application for detecting and redacting Personally Identifiable Information (PII) in Amharic text.

## 1. Project Overview

The Amharic PII Detector and Redactor aims to help protect sensitive personal information in Amharic text. It combines a trained Named Entity Recognition (NER) model with rule-based detection to identify sensitive information and replace detected values with category labels.

For example:

Input:

ስሜ ሄርሜላ ነው። ስልኬ 0716284053 ነው።
Redacted output:

ስሜ [PERSON_NAME] ነው። ስልኬ [PHONE] ነው።
## 2. Objectives

* Detect sensitive information in Amharic text.
* Identify multiple PII categories using NLP and pattern-based rules.
* Replace detected values with category-specific placeholders.
* Evaluate model performance using held-out test data and an additional evaluation dataset.
* Provide a simple interface for users to submit text and view redacted results.

## 3. PII Categories

The project targets the following categories:

* PERSON_NAME — personal names
* PHONE — phone numbers
* EMAIL — email addresses
* ADDRESS_PLACE — place and address information
* DATE_OF_BIRTH — dates of birth
* NATIONAL_ID — 12-digit FIN identifiers
* TIN — Taxpayer Identification Numbers
* BANK_ACCOUNT — bank account numbers

Detection performance varies by category. Some categories require further training and evaluation.

## 4. Technologies Used

* Python — application logic and data processing
* Streamlit — interactive web interface
* PyTorch — model training and inference
* Hugging Face Transformers — token classification with XLM-RoBERTa
* Regular expressions — structured PII detection
* pytest — automated testing
* Git and GitHub — version control and project history

## 5. System Architecture

The application follows this general workflow:

1. The user enters Amharic text.
2. The detection pipeline processes the text using the trained NER model and supported rules.
3. Detected spans are assigned PII categories.
4. The redaction function replaces detected values with category placeholders.
5. The interface displays the redacted text and detection results.

## 6. Project Structure

pii_redactor/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── ner_model/
├── data/
│   ├── synthetic/
│   └── processed/
├── src/
│   ├── generate_synthetic_data.py
│   ├── spans_to_bio.py
│   ├── split_dataset.py
│   ├── prepare_tokenized_data.py
│   ├── train_ner.py
│   ├── evaluate_ner.py
│   └── rule_detector.py
└── tests/
Additional scripts and evaluation files are also included in the repository.

## 7. Installation

### Prerequisites

* Python 3.12 recommended for this project environment
* Git
* A computer capable of running the trained model

### Setup

Clone the repository:

git clone https://github.com/MELAT-odi/PII-Redactor.git
cd PII-Redactor
Create and activate a virtual environment.

Windows PowerShell:

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
Install the dependencies:

python -m pip install -r requirements.txt
The trained model files must also be present in models/ner_model for the application to run as configured.

## 8. Run the Application

From the project root, execute:

streamlit run app.py
Open the local URL printed in the terminal by Streamlit.

Enter Amharic text in the application to inspect the detections and redacted output.

## 9. Dataset and Model Evaluation

The project uses synthetically generated and annotated data to train and evaluate the NER model. The dataset was divided into training, validation, and test splits.

The held-out test evaluation produced the following entity-level results:

| Metric    |  Score |
| --------- | -----: |
| Precision | 0.8907 |
| Recall    | 0.8873 |
| F1-score  | 0.8890 |

An additional evaluation on 20 records containing 33 annotated entities produced:
| Metric    |  Score |
| --------- | -----: |
| Precision | 0.8667 |
| Recall    | 0.7879 |
| F1-score  | 0.8254 |

The additional evaluation is small and should not be interpreted as proof of performance on all real-world Amharic documents. Results can vary substantially between PII categories and text styles.

## 10. Testing

Run the rule-detector tests from the project root:

python -m pytest src/test_rule_detector.py -v
These tests cover FIN identifier length, grouped identifier formats, boundary conditions, and character offsets.

## 11. Current Limitations

* The training dataset is synthetic and may not represent all real-world Amharic writing styles.
* Some PII categories have lower precision or recall than others.
* Contextual names and addresses can be difficult to identify consistently.
* Unusual number formats, spelling variations, and unseen patterns may be missed.
* The current application implements irreversible redaction. Encrypted reversible redaction and restoration are not yet implemented.
* The application is a research and educational prototype, not a production-certified privacy solution.

Because detection is imperfect, users should not assume that all sensitive information has been removed.

## 12. Future Improvements

* Improve training data diversity and annotation quality.
* Expand evaluation with carefully annotated real-world-style examples.
* Improve detection of national IDs, TINs, and addresses.
* Add a securely designed reversible redaction mode using authenticated encryption.
* Improve privacy safeguards, error handling, and deployment security.
* Evaluate performance across different Amharic text sources.

## 13. Privacy and Responsible Use

Avoid submitting real sensitive personal information unless the environment has been appropriately secured. The application should be tested with synthetic data wherever possible.

Irreversible redaction replaces detected text, but missed detections can leave sensitive information exposed. The current prototype should not be relied on as the sole privacy protection mechanism for medical, financial, or other confidential records.

## 14. Author and Repository

Project: Amharic PII Detector and Redactor
Repository: https://github.com/MELAT-odi/PII-Redactor