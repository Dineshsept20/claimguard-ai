"""Custom evaluation metrics for anomaly detection.

Provides metrics tailored to the pharmacy fraud detection use case:
per-anomaly-type detection rates, PR-AUC, and formatted reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute overall binary classification metrics.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Predicted probabilities for positive class (optional).

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = 0.0

    return metrics


def compute_per_type_detection_rate(
    df: pd.DataFrame,
    y_pred_col: str = "predicted_anomaly",
    anomaly_type_col: str = "anomaly_type",
) -> pd.DataFrame:
    """Compute detection rate for each anomaly type.

    Args:
        df: DataFrame with true anomaly type and predicted label.
        y_pred_col: Column with binary predicted label.
        anomaly_type_col: Column with the anomaly type string.

    Returns:
        DataFrame with columns: anomaly_type, total, detected, detection_rate.
    """
    anomalous = df[df[anomaly_type_col] != "none"].copy()
    if len(anomalous) == 0:
        return pd.DataFrame(columns=["anomaly_type", "total", "detected", "detection_rate"])

    results = []
    for atype in anomalous[anomaly_type_col].unique():
        mask = anomalous[anomaly_type_col] == atype
        total = mask.sum()
        detected = (anomalous.loc[mask, y_pred_col] == 1).sum()
        rate = detected / total if total > 0 else 0.0
        results.append({
            "anomaly_type": atype,
            "total": int(total),
            "detected": int(detected),
            "detection_rate": round(rate, 4),
        })

    return pd.DataFrame(results).sort_values("detection_rate", ascending=False).reset_index(drop=True)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute confusion matrix components.

    Returns:
        Dictionary with tn, fp, fn, tp and derived rates.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        "false_negative_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
    }


def print_model_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Print a formatted model evaluation report.

    Returns:
        Dictionary of all computed metrics.
    """
    metrics = compute_overall_metrics(y_true, y_pred, y_prob)
    cm = compute_confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    if "pr_auc" in metrics:
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={cm['true_positives']:,}  FP={cm['false_positives']:,}")
    print(f"    FN={cm['false_negatives']:,}  TN={cm['true_negatives']:,}")
    print(f"    FPR={cm['false_positive_rate']:.4f}  FNR={cm['false_negative_rate']:.4f}")

    return {**metrics, **cm}
