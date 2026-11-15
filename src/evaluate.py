"""
Model Evaluation and Metrics Module
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def evaluate_model(model, X, y, model_name: str = "Model") -> dict:
    """
    Evaluates a model and returns a dictionary of performance metrics
    specifically tailored for imbalanced fraud detection.
    """
    y_pred = model.predict(X)

    # Get predicted probabilities for positive class (fraud)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        y_prob = y_pred

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y, y_prob)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y, y_pred)

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision (Fraud)": prec,
        "Recall (Fraud)": rec,
        "F1-Score (Fraud)": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion Matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    return metrics


def print_evaluation_summary(metrics: dict):
    """
    Pretty print model evaluation results.
    """
    print(f"\n{'='*55}")
    print(f" Evaluation Results: {metrics['Model']}")
    print(f"{'='*55}")
    print(f" Accuracy:            {metrics['Accuracy']:.4f}")
    print(f" Precision (Fraud):   {metrics['Precision (Fraud)']:.4f}")
    print(f" Recall (Fraud):      {metrics['Recall (Fraud)']:.4f}  <-- Caught fraud %")
    print(f" F1-Score (Fraud):    {metrics['F1-Score (Fraud)']:.4f}")
    print(f" ROC-AUC Score:       {metrics['ROC-AUC']:.4f}")
    print(f" PR-AUC Score:        {metrics['PR-AUC']:.4f}")
    print(f"\n Confusion Matrix:")
    cm = metrics["Confusion Matrix"]
    print(f"   [True Legitimate: {cm[0][0]:<7} | False Fraud (Alarm): {cm[0][1]:<6}]")
    print(f"   [Missed Fraud:    {cm[1][0]:<7} | Detected Fraud:     {cm[1][1]:<6}]")
    print(f"{'='*55}\n")
