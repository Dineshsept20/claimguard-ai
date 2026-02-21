"""Isolation Forest unsupervised anomaly detection model.

Uses scikit-learn's Isolation Forest as a baseline anomaly detector
that doesn't require labels. Anomalous pharmacy claims are isolated
more quickly in the random partitioning trees, yielding lower
anomaly scores.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import compute_overall_metrics, print_model_report


# Feature columns used by the Isolation Forest
FEATURE_COLUMNS = [
    # Claim-level features
    "cost_vs_awp_ratio",
    "quantity_vs_typical",
    "days_supply_qty_mismatch",
    "is_controlled",
    "is_specialty",
    "is_weekend",
    "is_after_hours",
    "cost_percentile",
    "dispensing_fee_ratio",
    "opioid_mme_daily",
    "high_refill",
    "plan_paid_ratio",
    # Network features
    "pair_presc_exclusivity",
    "pair_pharm_exclusivity",
    "pair_log_volume",
    "member_class_unique_prescribers",
    "member_unique_pharmacies",
    "member_unique_prescribers",
    "member_controlled_ratio",
    "doctor_shopping_signal",
    "pharmacy_shopping_signal",
    # Prescriber features (joined)
    "presc_controlled_ratio",
    "presc_top_pharmacy_pct",
    "presc_avg_mme_per_patient",
    "presc_after_hours_ratio",
    "presc_cost_peer_zscore",
    "presc_claims_per_member",
    # Pharmacy features (joined)
    "pharm_controlled_ratio",
    "pharm_reversal_rate",
    "pharm_brand_ratio",
    "pharm_after_hours_ratio",
    "pharm_top_prescriber_pct",
    "pharm_cost_peer_zscore",
    "pharm_volume_peer_zscore",
    "pharm_claims_per_member",
    # Business rules
    "rule_flags_count",
]


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of FEATURE_COLUMNS that exist in the DataFrame."""
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.04,
    n_estimators: int = 200,
    max_samples: str | int = "auto",
    random_state: int = 42,
    save_path: str | None = "models/isolation_forest.pkl",
) -> tuple[IsolationForest, StandardScaler, list[str]]:
    """Train an Isolation Forest model.

    Args:
        df: Feature-engineered claims DataFrame.
        contamination: Expected proportion of anomalies.
        n_estimators: Number of trees.
        max_samples: Samples per tree.
        random_state: Random seed.
        save_path: Path to save the trained model (None to skip).

    Returns:
        Tuple of (trained model, fitted scaler, feature column list).
    """
    features = get_available_features(df)
    print(f"  Training Isolation Forest with {len(features)} features...")

    X = df[features].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        artifact = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "contamination": contamination,
        }
        with open(save_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"  Model saved to {save_path}")

    return model, scaler, features


def predict_isolation_forest(
    df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
    features: list[str],
) -> pd.DataFrame:
    """Generate predictions from a trained Isolation Forest.

    Args:
        df: Feature-engineered claims DataFrame.
        model: Trained IsolationForest.
        scaler: Fitted StandardScaler.
        features: Feature column names.

    Returns:
        DataFrame with `if_prediction` (0/1) and `if_anomaly_score` columns.
    """
    X = df[features].copy().fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)

    # predict: 1 = normal, -1 = anomaly
    raw_preds = model.predict(X_scaled)
    df = df.copy()
    df["if_prediction"] = (raw_preds == -1).astype(int)

    # Anomaly score: lower = more anomalous; we negate so higher = more anomalous
    raw_scores = model.decision_function(X_scaled)
    df["if_anomaly_score"] = -raw_scores  # flip sign: higher = more anomalous

    # Normalize to 0-1 range
    min_score = df["if_anomaly_score"].min()
    max_score = df["if_anomaly_score"].max()
    if max_score > min_score:
        df["if_anomaly_score"] = (df["if_anomaly_score"] - min_score) / (max_score - min_score)

    return df


def evaluate_isolation_forest(
    df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
    features: list[str],
) -> dict:
    """Evaluate the Isolation Forest on labeled data.

    Returns:
        Dictionary of evaluation metrics.
    """
    df = predict_isolation_forest(df, model, scaler, features)
    y_true = df["is_anomaly"].astype(int).values
    y_pred = df["if_prediction"].values
    y_prob = df["if_anomaly_score"].values

    metrics = print_model_report("Isolation Forest", y_true, y_pred, y_prob)
    return metrics


def load_isolation_forest(path: str = "models/isolation_forest.pkl"):
    """Load a saved Isolation Forest model artifact."""
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["scaler"], artifact["features"]
