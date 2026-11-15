"""
Prediction and Inference Script for Credit Card Fraud Detection
Loads saved model artifacts and evaluates on test transactions or test dataset.
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib

from src.preprocessing import prepare_data
from src.evaluate import evaluate_model, print_evaluation_summary


def load_artifacts(model_path: str = "models/fraud_detection_model.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Please run 'python train.py' first."
        )
    return joblib.load(model_path)


def predict_test_dataset(test_path: str = "fraudTest.csv", sample_size: int = 50000, threshold: float = 0.5):
    print("=" * 65)
    print(" EVALUATION ON UNSEEN TEST DATASET")
    print("=" * 65)

    artifacts = load_artifacts()
    model = artifacts["model"]
    model_name = artifacts.get("model_name", "HistGradientBoosting")
    label_encoders = artifacts["label_encoders"]

    print(f"Loaded trained model: {model_name}")
    print(f"Reading test data from '{test_path}' (Sample: {sample_size:,} rows)...")
    test_df = pd.read_csv(test_path, nrows=sample_size)

    X_test, y_test, _ = prepare_data(test_df, target_col="is_fraud", label_encoders=label_encoders)

    # Evaluate using standard 0.5 threshold
    metrics = evaluate_model(model, X_test, y_test, model_name=f"{model_name} on Test Data")
    print_evaluation_summary(metrics)

    # If custom threshold is specified
    if threshold != 0.5:
        print(f"\n--- Custom Decision Threshold Evaluation (Threshold = {threshold:.2f}) ---")
        y_prob = metrics["y_prob"]
        y_custom_pred = (y_prob >= threshold).astype(int)
        
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(y_test, y_custom_pred, target_names=["Legitimate (0)", "Fraud (1)"], digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_custom_pred))

    print("=" * 65)


def predict_single_transaction(amt: float, category: str, hour: int, age: int, distance_km: float, is_night: int):
    """
    Demonstration of scoring a single real-time transaction.
    """
    artifacts = load_artifacts()
    model = artifacts["model"]
    label_encoders = artifacts["label_encoders"]

    # Transform category
    cat_le = label_encoders.get("category")
    cat_encoded = cat_le.transform([category])[0] if category in cat_le.classes_ else 0

    log_amt = np.log1p(amt)
    
    # Feature order: ['category', 'amt', 'gender', 'city_pop', 'trans_hour', 'trans_day_of_week', 'is_night', 'age', 'distance_km', 'log_amt']
    sample_feature = pd.DataFrame([{
        "category": cat_encoded,
        "amt": amt,
        "gender": 0,
        "city_pop": 50000,
        "trans_hour": hour,
        "trans_day_of_week": 2,
        "is_night": is_night,
        "age": age,
        "distance_km": distance_km,
        "log_amt": log_amt
    }])

    prob = model.predict_proba(sample_feature)[0, 1]
    is_fraud = int(prob >= 0.5)

    print("\n" + "=" * 50)
    print(" SINGLE TRANSACTION SCORING RESULT")
    print("=" * 50)
    print(f" Transaction Amount:     ${amt:.2f}")
    print(f" Merchant Category:      {category}")
    print(f" Transaction Hour:       {hour}:00 ({'Night' if is_night else 'Day'})")
    print(f" Cardholder Age:         {age}")
    print(f" Distance to Merchant:   {distance_km:.1f} km")
    print("-" * 50)
    print(f" Fraud Probability:      {prob * 100:.2f}%")
    print(f" Final Prediction:       {'🔴 FRAUD DETECTED!' if is_fraud else '🟢 LEGITIMATE TRANSACTION'}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference or test evaluation")
    parser.add_argument("--test_data", type=str, default="fraudTest.csv", help="Path to test CSV")
    parser.add_argument("--sample", type=int, default=50000, help="Number of test rows to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (0.0 to 1.0)")
    parser.add_argument("--demo", action="store_true", help="Run single transaction demo")

    args = parser.parse_args()

    if args.demo:
        # Score a suspicious late-night high-amount transaction
        print("\n--- Example 1: Suspicious Late-Night High-Value Purchase ---")
        predict_single_transaction(amt=850.0, category="shopping_net", hour=23, age=45, distance_km=120.0, is_night=1)

        # Score a normal daytime small purchase
        print("\n--- Example 2: Normal Daytime Coffee/Grocery Purchase ---")
        predict_single_transaction(amt=24.50, category="food_dining", hour=14, age=32, distance_km=4.2, is_night=0)
    else:
        predict_test_dataset(test_path=args.test_data, sample_size=args.sample, threshold=args.threshold)
