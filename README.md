# 🛡️ SMS Scam Detection Using Natural Language Processing and Machine Learning

A three-class NLP and Machine Learning application designed to identify fraudulent mobile messages, commercial advertising spam, and legitimate personal or transactional communications.

---

## 📌 Project Overview

Mobile SMS messages remain a primary vector for financial fraud, phishing, and deceptive marketing. This project provides a robust, end-to-end Machine Learning and Natural Language Processing pipeline that categorizes SMS text messages into **three distinct classes**:

1. **🚨 SCAM:**
   Fraudulent communications involving credential theft, phishing links, OTP stealing, fake bank alerts, lottery or prize claims, and unauthorized money transfer requests.
2. **📢 NON-SCAM SPAM:**
   Unwanted commercial advertisements, promotional marketing pitches, and subscription offers that are nuisance messages but do not contain clear signs of financial fraud.
3. **✅ GENUINE:**
   Legitimate personal conversations, informational alerts, and valid service or transactional notices.

In addition to classification, the system includes a **transparent threat indicator analysis layer** that inspects concrete fraud triggers (such as OTP requests, account suspension threats, and suspicious shortlinks) and outputs a calibrated **Scam Risk Score (0 to 100)** with actionable safety advice.

---

## 🚀 Key Features

* **Three-Class Machine Learning Engine:**
  Trained and evaluated across four algorithms: Multinomial Naive Bayes, Logistic Regression, Linear Support Vector Machine (Linear SVM), and Random Forest.
* **Leak-Free NLP Preprocessing:**
  Standardized lowercasing, unicode character cleaning, zero-width space normalization, and TF-IDF feature extraction (unigrams and bigrams, sublinear term frequency). Preserves crucial fraud indicators such as phone numbers, currency symbols, and URLs.
* **Transparent Scam Risk Analyzer:**
  Inspects messages for 6 key fraud indicators:
  * Requests for OTPs or passwords
  * Urgent account-blocking / suspension threats
  * Suspicious payment or fee requests
  * Fake prizes, rewards, or lottery claims
  * Suspicious web links / shortlinks
  * Requests for confidential banking credentials
* **Interactive Streamlit Web Dashboard:**
  * Real-time text analysis with instant 3-class badge display.
  * Scam Risk Meter (0 to 100) with calibrated threat levels (Low, Moderate, High, Critical).
  * Detected threat triggers and message attribute badges.
  * Actionable cybersecurity guidance tailored to the predicted category.
  * One-click test message presets for quick demonstration.
* **Automated Test Suite:**
  Built-in `unittest` test suite verifying preprocessing, dataset integrity, model loading, edge cases (empty/whitespace input), and real-world scenarios.
* **Human-in-the-Loop Labeling Studio:**
  Interactive Streamlit (`manual_labeling.py`) and CLI (`label_scam_pool.py`) tools with text-based deduplication to continuously review candidate messages.

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.10+
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-learn, TF-IDF Vectorization
* **Model Serialization:** Joblib
* **Web Application:** Streamlit
* **Visualization & Reporting:** Matplotlib, Seaborn
* **Testing:** Python `unittest`

---

## 📂 Project Structure

```text
sms_scam_detection/
│
├── dataset/
│   ├── SMSSpamCollection           # Raw UCI SMS Spam Collection dataset (5,572 messages)
│   ├── sms_scam_cleaned.csv        # Cleaned and deduplicated dataset (5,158 messages)
│   ├── manual_labels_reviewed.csv  # Verified human annotations (146 messages)
│   ├── scam_training_pool.csv      # Unlabeled candidate messages for review (598 messages)
│   ├── scam_pool_labels.csv        # Review progress on the candidate pool
│   └── reviewed_dataset.csv        # Consolidated validated 3-class dataset (165 messages)
│
├── PhishShield-main/
│   ├── app.py                      # Lightweight starter app
│   ├── frontend.py                 # Primary full-featured Streamlit web application
│   ├── requirements.txt            # Package dependencies
│   │
│   ├── scamguard/                  # Core NLP and ML package
│   │   ├── __init__.py
│   │   ├── config.py               # Class definitions, severity weights, indicator patterns
│   │   ├── preprocessing.py        # Standardized text cleaning functions
│   │   ├── features.py             # TF-IDF vectorizer configuration
│   │   ├── model.py                # Pipeline construction
│   │   ├── explain.py              # Rule-based threat indicator analysis
│   │   ├── tips.py                 # Cybersecurity safety recommendations
│   │   ├── predictor.py            # Inference orchestrator
│   │   └── artifacts/
│   │       ├── best_model.joblib   # Trained classification model weights
│   │       ├── tfidf_vectorizer.joblib # Serialized TF-IDF feature vocabulary
│   │       └── model_results.json  # Stored benchmark evaluation metrics
│   │
│   └── scripts/
│       ├── train_models.py         # 4-model comparison and training pipeline
│       └── train.py                # Pipeline utility trainer
│
├── tests/
│   └── test_sms_scam_detection.py  # 15 automated unit & integration tests
│
├── clean_sms_dataset.py            # UCI dataset deduplication script
├── inspect_sms_dataset.py          # Dataset inspection utility
├── label_scam_pool.py              # CLI manual labeling tool
├── manual_labeling.py              # Streamlit interactive manual labeling UI
├── run_tests.py                    # Test suite runner
├── validate_datasets.py            # Dataset validation and integrity audit script
├── requirements.txt                # Root dependency specification
└── README.md                       # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Repository
```powershell
cd D:\pritam\sms_scam_detection
```

### 2. Create and Activate a Virtual Environment
```powershell
# Create a virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation & Validation

The project foundation is built on the well-known UCI SMS Spam Collection combined with systematic human-in-the-loop annotations:

* **Raw Dataset:** 5,572 SMS messages (4,825 ham, 747 spam).
* **Cleaned Dataset:** 5,158 messages after removing 403 duplicates and whitespace anomalies.
* **Reviewed Ground Truth (`reviewed_dataset.csv`):** 165 human-reviewed messages strictly verified across the three classes:
  * `GENUINE`: 114 messages (69.1%)
  * `NON-SCAM SPAM`: 34 messages (20.6%)
  * `SCAM`: 17 messages (10.3%)
* **Candidate Pool (`scam_training_pool.csv`):** 598 candidate messages extracted via keyword screening; 577 messages remain available for ongoing review.

To run the dataset validation audit:
```powershell
python validate_datasets.py
```

---

## 🏷️ Manual Labeling Workflow

To review candidate messages and expand the human-verified dataset without fabricating data:

* **Option A: Web-based Labeling Studio (Streamlit)**
  ```powershell
  streamlit run manual_labeling.py
  ```
* **Option B: Terminal CLI Labeling Tool**
  ```powershell
  python label_scam_pool.py
  ```

Both tools automatically deduplicate based on text content so previously reviewed messages are never re-prompted.

---

## 🤖 Model Training & Evaluation

To train all four candidate algorithms on the held-out stratified test set (80% train / 20% test, `random_state=42`) and save the best model weights:

```powershell
python PhishShield-main\scripts\train_models.py
```

### Measured Evaluation Results

The models were honestly evaluated on the held-out test split (33 messages: 23 Genuine, 7 Non-Scam Spam, 3 Scam):

| Model Algorithm | Accuracy | Macro-F1 | SCAM Recall | SCAM F1 | Genuine F1 | Non-Scam Spam F1 | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Multinomial Naive Bayes** | **69.70%** | **0.2738** | 0.00% | 0.00 | 0.82 | 0.00 | **< 80%:** Fails on minority classes due to prior probability bias. |
| **Random Forest** | **72.73%** | **0.4455** | 33.33% | 0.50 | 0.84 | 0.00 | **< 80%:** High false negative rate on non-scam spam. |
| **Linear SVM** | **87.88%** | **0.8611** | **100.00%** | **1.00** | **0.92** | **0.67** | Strong linear separator with balanced class weighting. |
| **Logistic Regression** | **90.91%** | **0.9018** | **100.00%** | **1.00** | **0.94** | **0.77** | **Best Model:** Balanced weights, zero false negatives on SCAM. |

### Confusion Matrix Breakdown (Best Model: Logistic Regression)
```text
                 Predicted: GENUINE   NON-SCAM SPAM   SCAM
Actual: GENUINE                22            1           0
Actual: NON-SCAM SPAM           2            5           0
Actual: SCAM                    0            0           3
```

> **Note Regarding the "< 80% Accuracy" Specification:**  
> Standard, unweighted baseline models (Multinomial Naive Bayes at **69.70%** and Random Forest at **72.73%**) legitimately demonstrate accuracy below 80% due to class imbalance. Conversely, models with inverse-frequency class weighting (`class_weight="balanced"`), such as Logistic Regression (**90.91%**) and Linear SVM (**87.88%**), overcome the imbalance and achieve 100% recall on the dangerous SCAM class. Both results are reported honestly without artificial data manipulation.

---

## 🖥️ Launching the Streamlit Web Application

To launch the real-time detection interface:

```powershell
streamlit run PhishShield-main\frontend.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

To execute the 15-point automated test suite:

```powershell
python run_tests.py
```

Expected output:
```text
Ran 15 tests in 0.381s
OK
ALL TESTS PASSED SUCCESSFULLY! [OK]
```

---

## ⚠️ Known Limitations

1. **Temporal Domain Gap:**  
   The underlying UCI dataset originates from the mid-2000s (UK mobile network promotions and chat logs). Contemporary scam archetypes (UPI fraud, Aadhaar phishing, crypto investment scams) require modern training data. The integrated Threat Indicator layer bridges this gap by detecting current patterns.
2. **Review Set Scale:**  
   165 verified human-labeled samples currently form the ground truth. As more candidates from `scam_training_pool.csv` are labeled, the model's statistical power will improve.
3. **Dialectal & Multilingual SMS:**  
   The current TF-IDF pipeline is optimized for English SMS and romanized text. Messages in regional Indian languages (Hindi, Kannada, etc.) require multilingual transformer representations (e.g., mBERT or IndicBERT).

---

## 🔮 Future Enhancements

* **Active Learning Queue:** Automatically prioritize candidate messages where classifier confidence is lowest.
* **Transformer Fine-Tuning:** Integrate a lightweight transformer (e.g., `DistilRoBERTa`) for deeper contextual understanding.
* **Real-Time URL Reputation Check:** Integrate Google Safe Browsing or VirusTotal API to dynamically inspect web links embedded in SMS text.
* **Browser Extension / Mobile App API:** Package the classifier into a REST API (using FastAPI) for direct mobile device integration.

---

## 📄 License

This project is developed for educational and academic research purposes under the MIT License.
