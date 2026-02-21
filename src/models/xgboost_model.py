"""XGBoost supervised anomaly classification model.

Uses XGBoost gradient boosting with class imbalance handling
(scale_pos_weight + optional SMOTE) for binary classification
of fraudulent pharmacy claims.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.metrics import print_model_report

# Feature columns — same set as Isolation Forest for consistency
FEATURE_COLUMNS = [
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
    "pair_presc_exclusivity",
    "pair_pharm_exclusivity",
    "pair_log_volume",
    "member_class_unique_prescribers",
    "member_unique_pharmacies",
    "member_unique_prescribers",
    "member_controlled_ratio",
    "doctor_shopping_signal",
    "pharmacy_shopping_signal",
    "presc_controlled_ratio",
    "presc_top_pharmacy_pct",
    "presc_avg_mme_per_patient",
    "presc_after_hours_ratio",
    "presc_cost_peer_zscore",
    "presc_claims_per_member",
    "pharm_controlled_ratio",
    "pharm_reversal_rate",
    "pharm_brand_ratio",
    "pharm_after_hours_ratio",
    "pharm_top_prescriber_pct",
    "pharm_cost_peer_zscore",
    "pharm_volume_peer_zscore",
    "pharm_claims_per_member",
    "rule_flags_count",
]


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of FEATURE_COLUMNS that exist in the DataFrame."""
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def train_xgboost(
    df: pd.DataFrame,
    use_smote: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str | None = "models/xgboost_model.pkl",
) -> tuple[XGBClassifier, StandardScaler, list[str], dict]:
    """Train an XGBoost classifier for fraud detection.

    Handles class imbalance via scale_pos_weight and optional SMOTE.
    Uses stratified train/test split for evaluation.

    Args:
        df: Feature-engineered claims DataFrame with is_anomaly label.
        use_smote: Whether to apply SMOTE oversampling on training data.
        test_size: Fraction reserved for testing.
        random_state: Random seed.
        save_path: Path to save the trained model.

    Returns:
        Tuple of (model, scaler, feature_columns, eval_metrics).
    """
    features = get_available_features(df)
    print(f"  Training XGBoost with {len(features)} features...")

    X = df[features].copy().fillna(0).replace([np.inf, -np.inf], 0)
    y = df["is_anomaly"].astype(int).values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Handle imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / max(pos_count, 1)

    # Optional SMOTE
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state, sampling_strategy=0.3)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"  SMOTE: {len(y_train):,} → {len(y_train_resampled):,} training samples")
        except ImportError:
            print("  Warning: imbalanced-learn not installed, skipping SMOTE")
            X_train_resampled, y_train_resampled = X_train, y_train
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = print_model_report("XGBoost", y_test, y_pred, y_prob)

    # Cross-validation F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"] = cv_scores.std()

    # Feature importance
    importance = dict(zip(features, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top 10 Features:")
    for feat, imp in top_features:
        print(f"    {feat}: {imp:.4f}")

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        artifact = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "metrics": metrics,
            "feature_importance": importance,
        }
        with open(save_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"\n  Model saved to {save_path}")

    return model, scaler, features, metrics


def predict_xgboost(
    df: pd.DataFrame,
    model: XGBClassifier,
    scaler: StandardScaler,
    features: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate predictions from a trained XGBoost model.

    Args:
        df: Feature-engineered claims DataFrame.
        model: Trained XGBClassifier.
        scaler: Fitted StandardScaler.
        features: Feature column names.
        threshold: Classification threshold.

    Returns:
        DataFrame with `xgb_prediction` and `xgb_probability` columns.
    """
    X = df[features].copy().fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)

    df = df.copy()
    df["xgb_probability"] = model.predict_proba(X_scaled)[:, 1]
    df["xgb_prediction"] = (df["xgb_probability"] >= threshold).astype(int)

    return df


def load_xgboost(path: str = "models/xgboost_model.pkl"):
    """Load a saved XGBoost model artifact."""
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["scaler"], artifact["features"]
