<div align="center">

<h1>🛡️ PhishShield — ScamGuard AI</h1>

<p>A multi-class NLP system that detects SMS and message-based scams in real time.<br/>
Built for student cybersecurity awareness. Classifies threats. Explains why. Tells you what to do.</p>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Try%20it%20now-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://phishshield-udpwbybgqf9yrzumyjlcps.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-97.33%25-success)
![CV Score](https://img.shields.io/badge/CV%20Score-96.33%25-success)
![Dataset](https://img.shields.io/badge/Dataset-300%20messages-blueviolet)
![Categories](https://img.shields.io/badge/Scam%20Categories-5-purple)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

</div>

---

## 🔗 Live Demo

> **Try PhishShield right now — no installation needed:**
> ### 👉 [https://phishshield-udpwbybgqf9yrzumyjlcps.streamlit.app/](https://phishshield-udpwbybgqf9yrzumyjlcps.streamlit.app/)

Paste any suspicious SMS, WhatsApp message, or email and get an instant scam analysis.

---

## What is PhishShield?

PhishShield (powered by **ScamGuard AI**) is a real-time scam message detection tool built with NLP. Paste any suspicious SMS, email, or chat message and it instantly tells you:

- 🔍 **What type of scam it is** — Phishing, OTP Scam, Lottery Scam, Job Scam, or Safe
- ⚠️ **How dangerous it is** — a risk score from 0 to 100 with a color-coded threat level
- 💡 **Why it flagged it** — suspicious keywords, scam-style phrases, URLs, phone numbers, money mentions
- 🛡️ **What to do next** — dynamic safety tips tailored to the specific scam type

> Built to protect students and everyday users from India's most common digital scams.

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **97.33%** |
| Cross-Validation Accuracy | **96.33% ± 2.87%** |
| Macro F1 Score | **0.97** |
| Training Dataset | **300 labeled messages** |
| Classes | **5 (Safe, Phishing, OTP Scam, Lottery Scam, Job Scam)** |

Cross-validation across 5 folds with low variance confirms the model is genuinely consistent — not just lucky on one split.

---

## Demo

| Message | Label | Risk Score | Reason |
|---|---|---|---|
| `Your bank account is suspended. Verify now: http://secure-login.xyz` | 🔴 Phishing | 91/100 | Contains a link; Suspicious keywords: verify, account, bank |
| `Your OTP is 928144. Reply with the code to verify.` | 🔴 OTP Scam | 85/100 | Scam-style phrase: your code is; Suspicious keywords: otp, code |
| `Congratulations! You have won a prize. Claim today.` | 🟠 Lottery Scam | 78/100 | Scam-style phrase: you have won; Suspicious keywords: prize, claim |
| `We are hiring! Work from home. Contact HR on Telegram.` | 🟠 Job Scam | 72/100 | Suspicious keywords: hiring, telegram, work from home |
| `Hey, are we still meeting for the study group at 5pm?` | 🟢 Safe | 4/100 | No common scam patterns detected |

---

## Features

- 🤖 **Multi-class scam detection** — 5 categories: Safe, Phishing, OTP Scam, Lottery Scam, Job Scam
- 📊 **Risk scoring** — 0–100 score combining model confidence and pattern signals
- 🔍 **Explainability panel** — shows exactly which keywords and phrases triggered the alert
- 🔗 **URL / phone / money detection** — regex-based signal extraction on top of the ML model
- 💬 **Dynamic safety tips** — advice changes based on the scam type detected
- 📈 **Session dashboard** — tracks how many scams detected in the current session
- 🇮🇳 **India-focused** — trained on Indian SMS patterns, brands, and scam styles

---

## Architecture

```
User pastes a message
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                    app.py  (Streamlit UI)                │
│       Input box · Analyze button · Result panel         │
│              Mini dashboard · Safety tips               │
└───────────────────┬─────────────────────────────────────┘
                    │  calls
                    ▼
┌─────────────────────────────────────────────────────────┐
│               scamguard/  (Python package)              │
│                                                         │
│  preprocessing.py  →  clean_text()                      │
│       │               Lowercase, strip URLs/noise       │
│       ▼                                                 │
│  model.py          →  ScamModel                         │
│       │               TF-IDF vectorizer                 │
│       │               + Logistic Regression             │
│       ▼                                                 │
│  predictor.py      →  ScamGuard.analyze()               │
│       │               Predicts label + probability      │
│       ▼                                                 │
│  explain.py        →  detect_patterns()                 │
│       │               Keywords, phrases, URL/phone/     │
│       │               money regex signals               │
│       ▼                                                 │
│  config.py         →  PATTERNS_BY_CLASS                 │
│       │               All scam keywords and phrases     │
│       ▼                                                 │
│  tips.py           →  get_tips()                        │
│                       Safety advice per scam type       │
└───────────────────┬─────────────────────────────────────┘
                    │  returns AnalysisResult
                    ▼
         Label · Confidence · Risk Score
         Suspicious Keywords · Phrases
         Has URL / Phone / Money
         Reason · Safety Tips
```

### Module responsibilities

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend — UI layout, session state, result rendering |
| `scamguard/__init__.py` | Exports the `ScamGuard` class as the public API |
| `scamguard/preprocessing.py` | Text cleaning — lowercasing, URL stripping, whitespace normalisation |
| `scamguard/features.py` | Feature extraction pipeline |
| `scamguard/model.py` | TF-IDF vectorizer + Logistic Regression training and loading |
| `scamguard/predictor.py` | Core `ScamGuard.analyze()` — orchestrates prediction and scoring |
| `scamguard/explain.py` | Pattern detection — keywords, phrases, URL/phone/money regex |
| `scamguard/config.py` | All scam patterns, severity weights, and class definitions |
| `scamguard/tips.py` | Safety tip generator, keyed by predicted scam type |
| `scripts/train.py` | Standalone training script — accuracy, F1, confusion matrix, cross-validation |
| `data/sample_messages.csv` | 300 labelled training messages across 5 scam categories |

---

## Project Structure

```
PhishShield/
├── app.py                        # Streamlit app entry point
├── requirements.txt              # Pinned dependencies
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guide
│
├── data/
│   └── sample_messages.csv       # 300 labeled messages (text, label)
│
├── scamguard/                    # Core detection package
│   ├── __init__.py
│   ├── config.py                 # Scam patterns, keywords, severity
│   ├── explain.py                # Explainability engine
│   ├── features.py               # Feature extraction
│   ├── model.py                  # ML model (TF-IDF + Logistic Regression)
│   ├── predictor.py              # Main analysis orchestrator
│   ├── preprocessing.py          # Text cleaning
│   ├── tips.py                   # Safety tip generator
│   └── artifacts/
│       └── model.joblib          # Saved trained model
│
└── scripts/
    └── train.py                  # Training script with full evaluation report
```

---

## Scam Categories

| Category | Description | Example signal |
|---|---|---|
| 🔴 **Phishing** | Fake login pages, credential theft, account suspension threats | "verify your account", suspicious URLs |
| 🔴 **OTP Scam** | Social engineering to steal one-time passwords | "share the code", "your OTP is" |
| 🟠 **Lottery Scam** | Fake prize or reward claims | "you have won", "claim your prize" |
| 🟠 **Job Scam** | Fake job offers, upfront payment requests via Telegram/WhatsApp | "work from home", "contact HR on Telegram" |
| 🟢 **Safe** | Legitimate messages with no scam signals | Normal conversation, study plans |

---

## Quickstart

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nihaaarika/PhishShield.git
cd PhishShield

# 2. Create a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the model (optional)

A pre-trained model is included. To retrain from scratch:

```bash
python -m scripts.train
```

Output includes accuracy, F1 score, confusion matrix, and 5-fold cross-validation.

### Run the app

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How It Works

### 1. Text preprocessing
Raw message → `preprocessing.py` → lowercased, URLs stripped, whitespace normalised.

### 2. ML classification
Cleaned text → TF-IDF vectorizer → Logistic Regression → predicted label + class probabilities.

### 3. Risk scoring
Risk score (0–100) = weighted combination of:
- Model confidence (probability of predicted class)
- Base severity by class (Phishing = 90, OTP Scam = 85, Lottery = 75, Job = 70, Safe = 0)
- Pattern bonus from `explain.py` (+6 per keyword, +10 per phrase, +10 for URL, +10 for money mention)

### 4. Explainability
`explain.py` runs independently of the ML model using regex and keyword matching from `config.py`:
- Detects class-specific suspicious keywords
- Detects scam-style phrases
- Detects URLs, phone numbers, and money/payment mentions (`₹`, `$`, `INR`)

### 5. Safety tips
`tips.py` returns tailored action items based on the predicted scam type.

---

## Dataset Format

Training data lives in `data/sample_messages.csv`:

```csv
text,label
"Your bank account is suspended. Verify now: http://secure-login.xyz","Phishing"
"Hey, are we meeting at 5pm?","Safe"
```

Valid labels: `Safe`, `Phishing`, `OTP Scam`, `Lottery Scam`, `Job Scam`

To contribute data, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Contributing

Contributions are welcome — code, data, documentation, or bug reports. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## Roadmap

- [x] 300 labeled real-world training messages
- [x] Model evaluation — accuracy, F1, confusion matrix, cross-validation
- [x] Live deployment on Streamlit Cloud
- [ ] Expand dataset to 1000+ messages
- [ ] Add Hindi and Hinglish scam pattern support
- [ ] Add pytest test suite for all core modules
- [ ] Add FastAPI REST endpoint for external integrations
- [ ] Integrate LIME/SHAP for model-level explainability

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built by [Nihaarika](https://github.com/nihaaarika) — student cybersecurity awareness project**

*If this tool helped you identify a scam, it worked. Stay safe online.* 🛡️

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20PhishShield%20Live-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://phishshield-udpwbybgqf9yrzumyjlcps.streamlit.app/)

</div>
