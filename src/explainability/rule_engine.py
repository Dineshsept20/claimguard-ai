"""Domain-specific business rules engine.

Applies pharmacy claims adjudication rules based on real-world PBM
knowledge. Each rule flags claims that violate known thresholds,
producing binary flags that feed into the ensemble model.
"""

import numpy as np
import pandas as pd


# Business rule thresholds (configurable)
RULE_THRESHOLDS = {
    "opioid_mme_daily_max": 90.0,
    "prescriber_top_pharmacy_pct_max": 0.60,
    "cost_vs_awp_ratio_max": 3.0,
    "quantity_vs_typical_max": 3.0,
    "doctor_shopping_prescribers_min": 3,
    "pharmacy_shopping_pharmacies_min": 3,
    "days_supply_mismatch_max": 0.5,
    "refill_number_max": 6,
    "dispensing_fee_max": 15.0,
    "pharmacy_controlled_ratio_max": 0.40,
}


def apply_business_rules(
    claims_df: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Apply domain business rules to flag suspicious claims.

    Each rule adds a binary column (0/1) indicating whether the
    rule fired. A composite `rule_flags_count` sums all flags.

    Args:
        claims_df: Claims DataFrame with engineered features.
        thresholds: Optional custom thresholds (overrides defaults).

    Returns:
        DataFrame with rule flag columns appended.
    """
    t = {**RULE_THRESHOLDS, **(thresholds or {})}
    df = claims_df.copy()

    # Rule 1: High opioid MME per day
    if "opioid_mme_daily" in df.columns:
        df["rule_high_mme"] = (df["opioid_mme_daily"] > t["opioid_mme_daily_max"]).astype(int)
    else:
        df["rule_high_mme"] = 0

    # Rule 2: Prescriber concentration (too many scripts to one pharmacy)
    if "pair_presc_exclusivity" in df.columns:
        df["rule_presc_concentration"] = (
            df["pair_presc_exclusivity"] > t["prescriber_top_pharmacy_pct_max"]
        ).astype(int)
    else:
        df["rule_presc_concentration"] = 0

    # Rule 3: Cost above AWP threshold
    if "cost_vs_awp_ratio" in df.columns:
        df["rule_high_cost"] = (
            df["cost_vs_awp_ratio"] > t["cost_vs_awp_ratio_max"]
        ).astype(int)
    else:
        df["rule_high_cost"] = 0

    # Rule 4: Quantity much higher than typical
    if "quantity_vs_typical" in df.columns:
        df["rule_high_quantity"] = (
            df["quantity_vs_typical"] > t["quantity_vs_typical_max"]
        ).astype(int)
    else:
        df["rule_high_quantity"] = 0

    # Rule 5: Doctor shopping signal
    if "member_class_unique_prescribers" in df.columns:
        df["rule_doctor_shopping"] = (
            df["member_class_unique_prescribers"] >= t["doctor_shopping_prescribers_min"]
        ).astype(int)
    elif "doctor_shopping_signal" in df.columns:
        df["rule_doctor_shopping"] = df["doctor_shopping_signal"]
    else:
        df["rule_doctor_shopping"] = 0

    # Rule 6: Pharmacy shopping signal
    if "member_unique_pharmacies" in df.columns:
        df["rule_pharmacy_shopping"] = (
            df["member_unique_pharmacies"] >= t["pharmacy_shopping_pharmacies_min"]
        ).astype(int)
    elif "pharmacy_shopping_signal" in df.columns:
        df["rule_pharmacy_shopping"] = df["pharmacy_shopping_signal"]
    else:
        df["rule_pharmacy_shopping"] = 0

    # Rule 7: Days supply / quantity mismatch
    if "days_supply_qty_mismatch" in df.columns:
        df["rule_ds_mismatch"] = (
            df["days_supply_qty_mismatch"] > t["days_supply_mismatch_max"]
        ).astype(int)
    else:
        df["rule_ds_mismatch"] = 0

    # Rule 8: High refill number
    df["rule_high_refill"] = (df["refill_number"] >= t["refill_number_max"]).astype(int)

    # Rule 9: Inflated dispensing fee
    df["rule_high_dispensing_fee"] = (
        df["dispensing_fee"] > t["dispensing_fee_max"]
    ).astype(int)

    # Rule 10: After-hours submission
    if "is_after_hours" in df.columns:
        df["rule_after_hours"] = df["is_after_hours"]
    else:
        df["rule_after_hours"] = (
            (df["submit_hour"] < 8) | (df["submit_hour"] > 18)
        ).astype(int)

    # Composite: count of rules fired
    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    df["rule_flags_count"] = df[rule_cols].sum(axis=1)

    return df


def get_rule_columns() -> list[str]:
    """Return the list of business rule flag column names."""
    return [
        "rule_high_mme",
        "rule_presc_concentration",
        "rule_high_cost",
        "rule_high_quantity",
        "rule_doctor_shopping",
        "rule_pharmacy_shopping",
        "rule_ds_mismatch",
        "rule_high_refill",
        "rule_high_dispensing_fee",
        "rule_after_hours",
        "rule_flags_count",
    ]
