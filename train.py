"""
Main Training Script for Credit Card Fraud Detection
Runs end-to-end: Loads data, extracts features, trains models, evaluates metrics, and saves the best model.
"""

import os
import argparse
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from src.preprocessing import prepare_data
from src.model import (
    get_logistic_regression,
    get_decision_tree,
    get_hist_gradient_boosting
)
from src.evaluate import evaluate_model, print_evaluation_summary


def train_pipeline(data_path: str = "fraudTrain.csv", sample_size: int = 250000):
    print("=" * 65)
    print(" CREDIT CARD FRAUD DETECTION TRAINING PIPELINE (IMBALANCED DATA)")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    # 1. Load Data
    t0 = time.time()
    print(f"\n[1/5] Loading training dataset from '{data_path}'...")
    if sample_size and sample_size > 0:
        print(f"      (Using first {sample_size:,} rows for fast, balanced benchmarking)")
        df = pd.read_csv(data_path, nrows=sample_size)
    else:
        print("      (Loading full dataset)")
        df = pd.read_csv(data_path)

    fraud_count = df["is_fraud"].sum()
    total_count = len(df)
    fraud_pct = (fraud_count / total_count) * 100
    print(f"      Loaded {total_count:,} transactions in {time.time() - t0:.2f}s")
    print(f"      Fraud Count: {fraud_count:,} ({fraud_pct:.2f}% of total) -> Imbalanced ratio: ~1:{int((total_count-fraud_count)/fraud_count)}")

    # 2. Preprocess & Feature Engineering
    t0 = time.time()
    print("\n[2/5] Engineering features (Haversine distance, hour, night flag, age, log amount)...")
    X, y, label_encoders = prepare_data(df, target_col="is_fraud")
    print(f"      Feature matrix shape: {X.shape} (processed in {time.time() - t0:.2f}s)")
    print(f"      Engineered Features: {list(X.columns)}")

    # 3. Stratified Train / Validation Split
    print("\n[3/5] Splitting data into 80% Train and 20% Validation (Stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train shape: {X_train.shape} (Fraud: {y_train.sum():,})")
    print(f"      Val shape:   {X_val.shape} (Fraud: {y_val.sum():,})")

    # 4. Model Training & Comparison
    print("\n[4/5] Training candidate models with balanced class weighting...")
    models = {
        "Logistic Regression (Balanced)": get_logistic_regression(balanced=True),
        "Decision Tree (Balanced)": get_decision_tree(balanced=True, max_depth=8),
        "HistGradientBoosting (Balanced)": get_hist_gradient_boosting(max_iter=100)
    }

    eval_summaries = []
    best_pr_auc = -1.0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        print(f"\n      Training {name}...")
        t_start = time.time()
        model.fit(X_train, y_train)
        train_duration = time.time() - t_start

        metrics = evaluate_model(model, X_val, y_val, model_name=name)
        print_evaluation_summary(metrics)

        eval_summaries.append({
            "Model": name,
            "Recall (Fraud %)": f"{metrics['Recall (Fraud)'] * 100:.2f}%",
            "Precision (%)": f"{metrics['Precision (Fraud)'] * 100:.2f}%",
            "PR-AUC": metrics["PR-AUC"],
            "ROC-AUC": metrics["ROC-AUC"],
            "Train Time (s)": round(train_duration, 2)
        })

        if metrics["PR-AUC"] > best_pr_auc:
            best_pr_auc = metrics["PR-AUC"]
            best_model_name = name
            best_model = model

    # Comparison Table
    print("\n" + "=" * 65)
    print(" MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 65)
    summary_df = pd.DataFrame(eval_summaries)
    print(summary_df.to_string(index=False))

    print(f"\n>>> Best Performing Model by PR-AUC: {best_model_name} (PR-AUC: {best_pr_auc:.4f})")

    # 5. Save Artifacts
    print("\n[5/5] Saving model and artifacts for deployment/inference...")
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "model": best_model,
        "model_name": best_model_name,
        "label_encoders": label_encoders,
        "feature_names": list(X.columns)
    }
    save_path = os.path.join("models", "fraud_detection_model.joblib")
    joblib.dump(artifacts, save_path)
    print(f"      Saved model artifacts to '{save_path}'")
    print("=" * 65)
    print(" TRAINING COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Credit Card Fraud Detection Models")
    parser.add_argument("--data", type=str, default="fraudTrain.csv", help="Path to training CSV file")
    parser.add_argument("--sample", type=int, default=250000, help="Number of rows to train on (default 250,000)")
    args = parser.parse_args()

    train_pipeline(data_path=args.data, sample_size=args.sample)
