"""End-to-end training pipeline for ClaimGuard AI.

Loads raw data, runs the full feature engineering pipeline,
trains all models, evaluates performance, and saves artifacts.

Usage:
    cd /Users/329228/Dinesh/claimguard-ai
    conda run -n claimguard python -m src.models.train_pipeline
"""

import os
import sys
import time

import pandas as pd

# Feature engineering
from src.features.claim_features import build_claim_features
from src.features.prescriber_features import build_prescriber_features
from src.features.pharmacy_features import build_pharmacy_features
from src.features.network_features import build_network_features
from src.explainability.rule_engine import apply_business_rules

# Models
from src.models.isolation_forest import train_isolation_forest
from src.models.xgboost_model import train_xgboost
from src.models.ensemble import (
    build_ensemble_scores,
    evaluate_ensemble,
    compare_models,
    save_ensemble_config,
)

# Drug reference for AWP / typical quantity lookups
from src.data_generator.reference_data import DRUG_REFERENCE

# Metrics
from src.utils.metrics import compute_per_type_detection_rate


def load_raw_data(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Load all CSV files from the raw data directory."""
    print("\n[1/6] Loading raw data...")
    t0 = time.time()

    claims = pd.read_csv(os.path.join(data_dir, "claims.csv"), parse_dates=["service_date"])
    pharmacies = pd.read_csv(os.path.join(data_dir, "pharmacies.csv"))
    prescribers = pd.read_csv(os.path.join(data_dir, "prescribers.csv"))
    members = pd.read_csv(os.path.join(data_dir, "members.csv"))

    print(f"  Claims:      {len(claims):>10,} rows")
    print(f"  Pharmacies:  {len(pharmacies):>10,} rows")
    print(f"  Prescribers: {len(prescribers):>10,} rows")
    print(f"  Members:     {len(members):>10,} rows")
    print(f"  Anomaly rate: {claims['is_anomaly'].mean():.2%}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    return {
        "claims": claims,
        "pharmacies": pharmacies,
        "prescribers": prescribers,
        "members": members,
    }


def run_feature_engineering(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    print("\n[2/6] Running feature engineering pipeline...")
    t0 = time.time()

    claims = data["claims"]
    pharmacies = data["pharmacies"]
    prescribers = data["prescribers"]

    # Step 1: Claim-level features
    print("  Step 1: Claim-level features...")
    claims = build_claim_features(claims, DRUG_REFERENCE)

    # Step 2: Prescriber-level features
    print("  Step 2: Prescriber-level features...")
    presc_features = build_prescriber_features(claims, prescribers)
    claims = claims.merge(presc_features, on="prescriber_id", how="left", suffixes=("", "_presc_dup"))
    # Drop any duplicate columns
    dup_cols = [c for c in claims.columns if c.endswith("_presc_dup")]
    claims = claims.drop(columns=dup_cols)

    # Step 3: Pharmacy-level features
    print("  Step 3: Pharmacy-level features...")
    pharm_features = build_pharmacy_features(claims, pharmacies)
    claims = claims.merge(pharm_features, on="pharmacy_id", how="left", suffixes=("", "_pharm_dup"))
    dup_cols = [c for c in claims.columns if c.endswith("_pharm_dup")]
    claims = claims.drop(columns=dup_cols)

    # Step 4: Network features
    print("  Step 4: Network features...")
    claims = build_network_features(claims)

    # Step 5: Business rules
    print("  Step 5: Business rules...")
    claims = apply_business_rules(claims)

    print(f"  Feature engineering completed in {time.time() - t0:.1f}s")
    print(f"  Final shape: {claims.shape}")

    return claims


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/claims_features.csv") -> None:
    """Save the feature-engineered dataset."""
    print(f"\n[3/6] Saving processed data to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {len(df):,} rows, {df.shape[1]} columns ({size_mb:.1f} MB)")


def train_all_models(df: pd.DataFrame) -> dict:
    """Train Isolation Forest, XGBoost, and ensemble."""
    print("\n[4/6] Training models...")
    t0 = time.time()

    # Isolation Forest (unsupervised)
    print("\n  --- Isolation Forest ---")
    if_model, if_scaler, if_features = train_isolation_forest(
        df, contamination=0.04, n_estimators=200
    )

    # XGBoost (supervised)
    print("\n  --- XGBoost ---")
    xgb_model, xgb_scaler, xgb_features, xgb_metrics = train_xgboost(
        df, use_smote=True
    )

    print(f"\n  All models trained in {time.time() - t0:.1f}s")

    return {
        "if_model": if_model,
        "if_scaler": if_scaler,
        "if_features": if_features,
        "xgb_model": xgb_model,
        "xgb_scaler": xgb_scaler,
        "xgb_features": xgb_features,
    }


def run_ensemble(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Run ensemble scoring and evaluation."""
    print("\n[5/6] Running ensemble scoring...")

    df = build_ensemble_scores(
        df,
        if_model=models["if_model"],
        if_scaler=models["if_scaler"],
        if_features=models["if_features"],
        xgb_model=models["xgb_model"],
        xgb_scaler=models["xgb_scaler"],
        xgb_features=models["xgb_features"],
    )

    # Evaluate ensemble
    print("\n  --- Ensemble Evaluation ---")
    evaluate_ensemble(df)

    # Compare all models
    comparison = compare_models(df)

    # Per-anomaly-type detection
    print("\n  --- Per-Anomaly-Type Detection (Ensemble) ---")
    per_type = compute_per_type_detection_rate(
        df, y_pred_col="ensemble_prediction", anomaly_type_col="anomaly_type"
    )
    for _, row in per_type.iterrows():
        print(f"  {row['anomaly_type']:<35} Detection: {row['detection_rate']:.2%}  "
              f"({int(row['detected'])}/{int(row['total'])})")

    # Save ensemble config
    save_ensemble_config()

    # Risk tier distribution
    print(f"\n  Risk Tier Distribution:")
    tier_counts = df["risk_tier"].value_counts()
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        print(f"    {tier:<10} {count:>8,} ({pct:.1f}%)")

    return df


def save_scored_data(df: pd.DataFrame, path: str = "data/processed/claims_scored.csv") -> None:
    """Save the scored dataset for dashboard consumption."""
    print(f"\n[6/6] Saving scored data to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Select key columns for the dashboard
    score_cols = [
        "claim_id", "service_date", "member_id", "prescriber_id",
        "pharmacy_id", "ndc", "drug_name", "therapeutic_class",
        "quantity", "days_supply", "total_cost", "is_anomaly", "anomaly_type",
        "if_anomaly_score", "if_prediction",
        "xgb_probability", "xgb_prediction",
        "rule_flags_count", "rule_score",
        "ensemble_score", "ensemble_prediction", "risk_tier",
    ]
    # Only include columns that exist
    out_cols = [c for c in score_cols if c in df.columns]
    df[out_cols].to_csv(path, index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {len(df):,} rows, {len(out_cols)} columns ({size_mb:.1f} MB)")


def main():
    """Run the full ClaimGuard AI training pipeline."""
    print("=" * 70)
    print("  ClaimGuard AI — Training Pipeline")
    print("=" * 70)
    pipeline_start = time.time()

    # 1. Load
    data = load_raw_data()

    # 2. Feature engineering
    df = run_feature_engineering(data)

    # 3. Save processed features
    save_processed_data(df)

    # 4. Train models
    models = train_all_models(df)

    # 5. Ensemble scoring
    df = run_ensemble(df, models)

    # 6. Save scored data
    save_scored_data(df)

    elapsed = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"  Artifacts saved to models/ and data/processed/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
