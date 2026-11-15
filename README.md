---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
---

# Credit Card Fraud Detection using Machine Learning on Imbalanced Data

An end-to-end, beginner-friendly Machine Learning project for detecting fraudulent credit card transactions in heavily imbalanced financial datasets.

---

## 📌 1. Problem Statement & Business Context

Credit card fraud represents billions of dollars in annual losses for financial institutions and cardholders worldwide. Detecting fraudulent transactions in real time is challenging because:
- **Extreme Class Imbalance:** Fraudulent transactions occur very rarely compared to legitimate day-to-day transactions (typically **< 1%** of all transactions).
- **The Accuracy Paradox:** If a model simply predicts "Legitimate" (0) for every transaction, it can achieve **99.4% accuracy**, yet fail to detect a single fraud case.
- **Cost Asymmetry:** A **False Negative** (missing a fraud event) results in direct financial loss and customer churn, whereas a **False Positive** (flagging a legitimate purchase) merely triggers a verification SMS or manual review.

**Project Objective:** Build, evaluate, and fine-tune machine learning models that maximize **Fraud Recall** and **Precision-Recall Area Under the Curve (PR-AUC)** while maintaining practical precision.

---

## 📊 2. Dataset Source & Information

* **Dataset:** Simulated Credit Card Transaction Dataset
* **Source:** Generated using the [Sparkov Data Generation Tool](https://github.com/namebrandon/Sparkov_Data_Generation) by Brandon Harris, available on [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection).
* **Time Span:** January 1, 2019 to December 31, 2020.
* **Files:**
  - `fraudTrain.csv`: 1,296,675 transactions (training split).
  - `fraudTest.csv`: 555,719 transactions (unseen evaluation split).
* **Class Imbalance:**
  - Legitimate (0): **99.42%** (~1.29 million transactions)
  - Fraudulent (1): **0.58%** (~7,506 transactions)
  - Imbalance Ratio: ~**170 : 1**

### Feature Overview
| Column | Description |
|---|---|
| `trans_date_trans_time` | Timestamp of the transaction |
| `cc_num` | Credit card number (anonymized/identifier) |
| `merchant` | Merchant name |
| `category` | Merchant spending category (e.g. `shopping_net`, `grocery_pos`) |
| `amt` | Transaction amount in USD |
| `first`, `last`, `gender` | Cardholder demographics |
| `street`, `city`, `state`, `zip` | Cardholder residence address |
| `lat`, `long` | Cardholder GPS coordinates |
| `city_pop` | Population of cardholder city |
| `job`, `dob` | Cardholder occupation and date of birth |
| `trans_num` | Unique transaction ID |
| `merch_lat`, `merch_long` | Merchant GPS coordinates |
| `is_fraud` | **Target variable** (0 = Normal, 1 = Fraud) |

---

## 🔍 3. Exploratory Data Analysis (Key Findings)

1. **Transaction Amount Disparity:**
   - **Legitimate transactions:** Median = **$47.35**, Mean = **$67.54**.
   - **Fraudulent transactions:** Median = **$346.44**, Mean = **$509.35**.
   - *Insight:* Fraudsters typically test small amounts first, then execute large-ticket purchases.
2. **High-Risk Categories:**
   - Online shopping (`shopping_net`), physical groceries (`grocery_pos`), and miscellaneous internet purchases (`misc_net`) experience the highest fraud frequencies.
3. **Late-Night Fraud Spike:**
   - Transactions occurring between **10:00 PM and 3:00 AM** have up to **10x higher fraud probability** compared to morning or afternoon transactions, as perpetrators exploit hours when cardholders are asleep.

---

## 🛠️ 4. Feature Engineering & Preprocessing

To provide maximum predictive signal to models without data leakage:
1. **Haversine Distance (`distance_km`):** Calculated great-circle distance between cardholder residence `(lat, long)` and merchant `(merch_lat, merch_long)` in kilometers.
2. **Temporal Features:**
   - `trans_hour`: Hour of transaction (0–23).
   - `trans_day_of_week`: Day of week (0–6).
   - `is_night`: Binary flag indicating transactions between 10 PM and 4 AM.
3. **Cardholder Age (`age`):** Calculated from Date of Birth (`dob`).
4. **Log Amount (`log_amt`):** $\log(1 + \text{amt})$ transformation to mitigate extreme right-skewness.
5. **Leakage Prevention:** Removed non-generalizable identifiers (`Unnamed: 0`, `trans_num`, `first`, `last`, `street`, `cc_num`, `unix_time`).
6. **Categorical Encoding:** `LabelEncoder` for `category` and `gender`.

---

## 🧠 5. Handling Imbalance & Modeling Strategy

1. **Stratified Splitting:** Splitting data with `stratify=y` ensures both train and validation sets retain the identical 0.58% fraud ratio.
2. **Cost-Sensitive Learning (`class_weight='balanced'`):** Penalizes false negatives proportionally to inverse class frequencies:
   $$W_1 = \frac{N_{\text{samples}}}{2 \times N_{\text{fraud}}}$$
3. **Primary Evaluation Metrics:**
   - **Recall (Fraud):** Fraction of actual fraud detected ($\frac{TP}{TP + FN}$).
   - **Precision (Fraud):** Fraction of flagged transactions that are genuinely fraudulent ($\frac{TP}{TP + FP}$).
   - **PR-AUC (Precision-Recall Area Under Curve):** Benchmark metric for evaluating imbalanced classification independent of fixed thresholds.
   - **ROC-AUC:** Area under receiver operating characteristic curve.

---

## 📈 6. Benchmark Results

Comparison across candidate models evaluated on validation data:

| Model | Accuracy | Recall (Fraud) | Precision (Fraud) | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| **Naive Baseline (Predict all 0s)** | 99.42% | 0.00% | 0.00% | 0.0058 | 0.5000 |
| **Logistic Regression (Standard)** | 99.41% | 22.34% | 61.20% | 0.2845 | 0.9120 |
| **Logistic Regression (Balanced)** | 87.46% | 81.11% | 5.57% | 0.3147 | 0.9447 |
| **Decision Tree (Balanced)** | 96.59% | 88.89% | 19.46% | 0.4295 | 0.9381 |
| **HistGradientBoosting (Balanced)** | **99.13%** | **91.11%** | **50.93%** | **0.8527** | **0.9966** |

### Evaluation on Unseen Test Set (`fraudTest.csv`)
* **Fraud Detection Rate (Recall):** **89.16%** (detected 74 out of 83 fraud cases in sample)
* **ROC-AUC Score:** **0.9759**
* **PR-AUC Score:** **0.7430**
* **Overall Accuracy:** **99.10%**

---

## 📁 7. Project Structure

```text
Credit_card_fraud_detection/
├── model.ipynb             # Interactive, step-by-step Jupyter Notebook for beginners
├── train.py                # CLI script to train models and export artifacts
├── predict.py              # CLI script to run inference on test data or single transactions
├── requirements.txt        # Project dependencies
├── .gitignore              # Ignores large datasets (>100MB) & cache
├── README.md               # Complete project documentation
├── models/                 # Saved model weights and encoders
│   └── fraud_detection_model.joblib
└── src/                    # Modular Python source code
    ├── __init__.py
    ├── preprocessing.py    # Feature extraction (Haversine distance, hour, age, encoders)
    ├── model.py            # Model definitions (Logistic Regression, Decision Tree, Boosted Trees)
    └── evaluate.py         # Evaluation metrics, confusion matrix, and reporting
```

---

## 🚀 8. Setup & Installation Instructions

### Prerequisites
- Python 3.9+ installed.
- Git installed.

### Step 1: Clone or Navigate to the Repository
```bash
cd Credit_card_fraud_detection
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 9. How to Run the Project

### Option A: Interactive Jupyter Notebook (`model.ipynb`)
Open `model.ipynb` in VS Code, JupyterLab, or Antigravity IDE:
```bash
jupyter notebook model.ipynb
```
Follow through the 11 step-by-step cells covering data loading, exploratory visualizations, feature engineering, model training, confusion matrices, PR curves, and threshold tuning.

---

### Option B: Run Training via Terminal
Train candidate models, view comparison tables, and save the best model:
```bash
# Fast benchmark run (first 50,000 rows):
python train.py --sample 50000

# Full dataset training:
python train.py --sample 250000
```

---

### Option C: Run Predictions & Inference
Evaluate the trained model on unseen test transactions or test a live transaction:

```bash
# 1. Evaluate on unseen test dataset (fraudTest.csv)
python predict.py --sample 25000

# 2. Evaluate with a custom decision threshold (e.g. 0.30 to catch more fraud)
python predict.py --sample 25000 --threshold 0.30

# 3. Run interactive single-transaction scoring demo
python predict.py --demo
```

**Single Transaction Demo Output:**
```text
--- Example 1: Suspicious Late-Night High-Value Purchase ---
 Transaction Amount:     $850.00
 Merchant Category:      shopping_net
 Transaction Hour:       23:00 (Night)
 Cardholder Age:         45
 Distance to Merchant:   120.0 km
--------------------------------------------------
 Fraud Probability:      98.74%
 Final Prediction:       🔴 FRAUD DETECTED!

--- Example 2: Normal Daytime Coffee/Grocery Purchase ---
 Transaction Amount:     $24.50
 Merchant Category:      food_dining
 Transaction Hour:       14:00 (Day)
 Cardholder Age:         32
 Distance to Merchant:   4.2 km
--------------------------------------------------
 Fraud Probability:      0.52%
 Final Prediction:       🟢 LEGITIMATE TRANSACTION
```

---

## 🎯 10. Key Takeaways for Freshers & Interview Talking Points

1. **Why accuracy fails on imbalanced data:** With 99.4% majority class, predicting constant 0 yields high accuracy but zero utility. Focus on PR-AUC, Recall, and Cost Matrix.
2. **Feature Engineering matters more than algorithm tuning:** Extracting distance using Haversine formula and isolating transaction hour (night flag) gave the tree models high predictive power.
3. **Threshold Calibration is a business decision:** In real banking systems, decision thresholds are dynamically tuned depending on the cost of customer friction vs. the dollar value at risk.
