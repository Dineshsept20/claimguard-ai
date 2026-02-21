"""Hybrid ensemble model combining Isolation Forest, XGBoost, and business rules.

The ensemble produces a final risk score by weighting:
  1. Isolation Forest anomaly score (unsupervised signal)
  2. XGBoost fraud probability (supervised signal)
  3. Business rule flag count (domain knowledge signal)

This hybrid approach catches anomalies that any single model might miss.
"""

import os
import pickle

import numpy as np
import pandas as pd

from src.models.isolation_forest import predict_isolation_forest
from src.models.xgboost_model import predict_xgboost
from src.utils.metrics import print_model_report


# Default ensemble weights
DEFAULT_WEIGHTS = {
    "isolation_forest": 0.25,
    "xgboost": 0.55,
    "business_rules": 0.20,
}


def build_ensemble_scores(
    df: pd.DataFrame,
    if_model,
    if_scaler,
    if_features: list[str],
    xgb_model,
    xgb_scaler,
    xgb_features: list[str],
    weights: dict | None = None,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Build ensemble risk scores combining all three models.

    Args:
        df: Feature-engineered claims DataFrame with rule_flags_count.
        if_model: Trained Isolation Forest model.
        if_scaler: Isolation Forest scaler.
        if_features: Isolation Forest feature columns.
        xgb_model: Trained XGBoost model.
        xgb_scaler: XGBoost scaler.
        xgb_features: XGBoost feature columns.
        weights: Dictionary of model weights (must sum to 1.0).
        threshold: Classification threshold for final prediction.

    Returns:
        DataFrame with ensemble columns added.
    """
    w = weights or DEFAULT_WEIGHTS

    # Get Isolation Forest scores
    df = predict_isolation_forest(df, if_model, if_scaler, if_features)

    # Get XGBoost probabilities
    df = predict_xgboost(df, xgb_model, xgb_scaler, xgb_features)

    # Normalize rule_flags_count to 0–1
    max_rules = df["rule_flags_count"].max()
    df["rule_score"] = df["rule_flags_count"] / max(max_rules, 1)

    # Weighted ensemble score
    df["ensemble_score"] = (
        w["isolation_forest"] * df["if_anomaly_score"]
        + w["xgboost"] * df["xgb_probability"]
        + w["business_rules"] * df["rule_score"]
    )

    # Final prediction
    df["ensemble_prediction"] = (df["ensemble_score"] >= threshold).astype(int)

    # Risk tier
    df["risk_tier"] = pd.cut(
        df["ensemble_score"],
        bins=[-0.01, 0.3, 0.6, 0.8, 1.01],
        labels=["Low", "Medium", "High", "Critical"],
    )

    return df


def evaluate_ensemble(df: pd.DataFrame) -> dict:
    """Evaluate the ensemble model on labeled data.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_true = df["is_anomaly"].astype(int).values
    y_pred = df["ensemble_prediction"].values
    y_prob = df["ensemble_score"].values

    metrics = print_model_report("Ensemble (IF + XGBoost + Rules)", y_true, y_pred, y_prob)
    return metrics


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Compare individual model performance vs ensemble.

    Args:
        df: DataFrame with all prediction columns.

    Returns:
        Comparison DataFrame.
    """
    y_true = df["is_anomaly"].astype(int).values
    results = []

    # Isolation Forest
    if "if_prediction" in df.columns:
        from src.utils.metrics import compute_overall_metrics
        m = compute_overall_metrics(y_true, df["if_prediction"].values, df["if_anomaly_score"].values)
        results.append({"model": "Isolation Forest", **m})

    # XGBoost
    if "xgb_prediction" in df.columns:
        m = compute_overall_metrics(y_true, df["xgb_prediction"].values, df["xgb_probability"].values)
        results.append({"model": "XGBoost", **m})

    # Business Rules Only
    if "rule_flags_count" in df.columns:
        rule_pred = (df["rule_flags_count"] >= 2).astype(int).values
        m = compute_overall_metrics(y_true, rule_pred, df["rule_score"].values if "rule_score" in df.columns else None)
        results.append({"model": "Business Rules (≥2 flags)", **m})

    # Ensemble
    if "ensemble_prediction" in df.columns:
        m = compute_overall_metrics(y_true, df["ensemble_prediction"].values, df["ensemble_score"].values)
        results.append({"model": "Ensemble", **m})

    comparison = pd.DataFrame(results)
    if len(comparison) > 0:
        print(f"\n{'='*70}")
        print("  Model Comparison")
        print(f"{'='*70}")
        for _, row in comparison.iterrows():
            roc_str = ""
            if "roc_auc" in row.index and pd.notna(row.get("roc_auc")):
                roc_str = f"ROC={row['roc_auc']:.4f}"
            print(f"  {row['model']:<30} F1={row['f1_score']:.4f}  "
                  f"Prec={row['precision']:.4f}  Rec={row['recall']:.4f}  {roc_str}")

    return comparison


def save_ensemble_config(
    if_path: str = "models/isolation_forest.pkl",
    xgb_path: str = "models/xgboost_model.pkl",
    weights: dict | None = None,
    threshold: float = 0.50,
    save_path: str = "models/ensemble_config.pkl",
) -> None:
    """Save ensemble configuration for dashboard loading."""
    config = {
        "if_path": if_path,
        "xgb_path": xgb_path,
        "weights": weights or DEFAULT_WEIGHTS,
        "threshold": threshold,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(config, f)
    print(f"  Ensemble config saved to {save_path}")


def load_ensemble(config_path: str = "models/ensemble_config.pkl"):
    """Load all ensemble components."""
    from src.models.isolation_forest import load_isolation_forest
    from src.models.xgboost_model import load_xgboost

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    if_model, if_scaler, if_features = load_isolation_forest(config["if_path"])
    xgb_model, xgb_scaler, xgb_features = load_xgboost(config["xgb_path"])

    return {
        "if_model": if_model,
        "if_scaler": if_scaler,
        "if_features": if_features,
        "xgb_model": xgb_model,
        "xgb_scaler": xgb_scaler,
        "xgb_features": xgb_features,
        "weights": config["weights"],
        "threshold": config["threshold"],
    }
