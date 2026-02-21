"""Per-claim feature engineering.

Computes features for each individual claim by comparing it against
drug reference data and statistical baselines. These features capture
signals like unusual quantities, cost deviations, and timing anomalies.
"""

import numpy as np
import pandas as pd


def build_claim_features(
    claims_df: pd.DataFrame,
    drug_ref: list[dict] | None = None,
) -> pd.DataFrame:
    """Engineer per-claim features from raw claims data.

    Args:
        claims_df: Raw claims DataFrame.
        drug_ref: Optional drug reference list. If None, imports from reference_data.

    Returns:
        DataFrame with original columns plus new feature columns.
    """
    if drug_ref is None:
        from src.data_generator.reference_data import DRUG_REFERENCE
        drug_ref = DRUG_REFERENCE

    df = claims_df.copy()

    # Parse dates if needed
    if df["service_date"].dtype == object:
        df["service_date"] = pd.to_datetime(df["service_date"])

    # --- Build drug reference lookup ---
    drug_lookup = {}
    for drug in drug_ref:
        drug_lookup[drug["ndc"]] = {
            "awp": drug["awp"],
            "typical_quantity": drug["typical_quantity"],
            "typical_days_supply": drug["typical_days_supply"],
        }

    # Map reference values to claims
    df["ref_awp"] = df["ndc"].map(lambda x: drug_lookup.get(x, {}).get("awp", np.nan))
    df["ref_typical_qty"] = df["ndc"].map(lambda x: drug_lookup.get(x, {}).get("typical_quantity", np.nan))
    df["ref_typical_ds"] = df["ndc"].map(lambda x: drug_lookup.get(x, {}).get("typical_days_supply", np.nan))

    # --- Feature 1: cost_vs_awp_ratio ---
    # ingredient_cost / (AWP * quantity) — values > 1.5 are suspicious
    expected_cost = df["ref_awp"] * df["quantity"]
    df["cost_vs_awp_ratio"] = np.where(
        expected_cost > 0,
        df["ingredient_cost"] / expected_cost,
        1.0,
    )

    # --- Feature 2: quantity_vs_typical ---
    # quantity / typical_quantity for that drug — values > 2.0 are suspicious
    df["quantity_vs_typical"] = np.where(
        df["ref_typical_qty"] > 0,
        df["quantity"] / df["ref_typical_qty"],
        1.0,
    )

    # --- Feature 3: days_supply_quantity_mismatch ---
    # How much does the days_supply deviate from what quantity implies?
    # Expected: quantity/typical_qty ≈ days_supply/typical_ds
    qty_ratio = df["quantity"] / df["ref_typical_qty"].replace(0, np.nan)
    ds_ratio = df["days_supply"] / df["ref_typical_ds"].replace(0, np.nan)
    df["days_supply_qty_mismatch"] = np.abs(qty_ratio.fillna(1) - ds_ratio.fillna(1))

    # --- Feature 4: is_controlled_substance ---
    df["is_controlled"] = (df["dea_schedule"] != "NONE").astype(int)

    # --- Feature 5: is_specialty_drug ---
    specialty_classes = {"BIOLOGIC_IMMUNOLOGY", "ONCOLOGY", "HEPATITIS_C"}
    df["is_specialty"] = df["therapeutic_class"].isin(specialty_classes).astype(int)

    # --- Feature 6: is_weekend ---
    df["is_weekend"] = (df["service_date"].dt.weekday >= 5).astype(int)

    # --- Feature 7: is_after_hours ---
    df["is_after_hours"] = ((df["submit_hour"] < 8) | (df["submit_hour"] > 18)).astype(int)

    # --- Feature 8: cost_percentile ---
    # Where this claim falls in cost distribution for its therapeutic class
    df["cost_percentile"] = df.groupby("therapeutic_class")["total_cost"].rank(pct=True)

    # --- Feature 9: dispensing_fee_ratio ---
    # dispensing_fee / total_cost — inflated fees signal upcoding
    df["dispensing_fee_ratio"] = np.where(
        df["total_cost"] > 0,
        df["dispensing_fee"] / df["total_cost"],
        0,
    )

    # --- Feature 10: opioid_mme_daily ---
    # morphine milligram equivalent per day (for opioids)
    df["opioid_mme_daily"] = np.where(
        df["days_supply"] > 0,
        (df["mme_factor"] * df["quantity"]) / df["days_supply"],
        0,
    )

    # --- Feature 11: month / day_of_week (temporal) ---
    df["service_month"] = df["service_date"].dt.month
    df["service_dow"] = df["service_date"].dt.weekday

    # --- Feature 12: high_refill_flag ---
    df["high_refill"] = (df["refill_number"] >= 6).astype(int)

    # --- Feature 13: plan_paid_ratio ---
    # What fraction of total cost the plan bears
    df["plan_paid_ratio"] = np.where(
        df["total_cost"] > 0,
        df["plan_paid"] / df["total_cost"],
        0,
    )

    # Drop temporary reference columns
    df.drop(columns=["ref_awp", "ref_typical_qty", "ref_typical_ds"], inplace=True)

    return df
