"""Network / relationship feature engineering.

Computes features that capture relationships between entities:
prescriber–pharmacy links, member–prescriber patterns (doctor shopping),
and peer deviation scores. These are the highest-signal features
for detecting collusion and shopping fraud.
"""

import numpy as np
import pandas as pd


def build_network_features(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer network/relationship features and merge back to claims.

    Builds three sets of relationship features:
    1. Prescriber-pharmacy pair exclusivity metrics
    2. Member-level doctor/pharmacy shopping signals
    3. Prescriber/pharmacy peer deviation scores

    Args:
        claims_df: Claims DataFrame.

    Returns:
        Claims DataFrame with network feature columns appended.
    """
    df = claims_df.copy()
    if df["service_date"].dtype == object:
        df["service_date"] = pd.to_datetime(df["service_date"])

    # ============================================================
    # 1. PRESCRIBER-PHARMACY PAIR FEATURES
    # ============================================================
    # How exclusive is this prescriber-pharmacy relationship?
    pair_counts = (
        df.groupby(["prescriber_id", "pharmacy_id"])["claim_id"]
        .count()
        .reset_index()
        .rename(columns={"claim_id": "pair_claim_count"})
    )

    presc_totals = (
        df.groupby("prescriber_id")["claim_id"]
        .count()
        .reset_index()
        .rename(columns={"claim_id": "presc_total_for_pair"})
    )

    pharm_totals = (
        df.groupby("pharmacy_id")["claim_id"]
        .count()
        .reset_index()
        .rename(columns={"claim_id": "pharm_total_for_pair"})
    )

    pair_counts = pair_counts.merge(presc_totals, on="prescriber_id", how="left")
    pair_counts = pair_counts.merge(pharm_totals, on="pharmacy_id", how="left")

    # Prescriber-pharmacy exclusivity (from prescriber's perspective)
    pair_counts["pair_presc_exclusivity"] = (
        pair_counts["pair_claim_count"] / pair_counts["presc_total_for_pair"]
    )

    # Pharmacy-prescriber exclusivity (from pharmacy's perspective)
    pair_counts["pair_pharm_exclusivity"] = (
        pair_counts["pair_claim_count"] / pair_counts["pharm_total_for_pair"]
    )

    # Log volume of the pair
    pair_counts["pair_log_volume"] = np.log1p(pair_counts["pair_claim_count"])

    # Merge back to claims
    pair_features = pair_counts[[
        "prescriber_id", "pharmacy_id",
        "pair_claim_count", "pair_presc_exclusivity",
        "pair_pharm_exclusivity", "pair_log_volume",
    ]]
    df = df.merge(pair_features, on=["prescriber_id", "pharmacy_id"], how="left")

    # ============================================================
    # 2. MEMBER-LEVEL SHOPPING SIGNALS
    # ============================================================
    # Unique prescribers per member (for same drug class)
    member_class_prescribers = (
        df.groupby(["member_id", "therapeutic_class"])["prescriber_id"]
        .nunique()
        .reset_index()
        .rename(columns={"prescriber_id": "member_class_unique_prescribers"})
    )
    df = df.merge(
        member_class_prescribers,
        on=["member_id", "therapeutic_class"],
        how="left",
    )

    # Unique pharmacies per member
    member_pharmacies = (
        df.groupby("member_id")["pharmacy_id"]
        .nunique()
        .reset_index()
        .rename(columns={"pharmacy_id": "member_unique_pharmacies"})
    )
    df = df.merge(member_pharmacies, on="member_id", how="left")

    # Unique prescribers per member (overall)
    member_prescribers = (
        df.groupby("member_id")["prescriber_id"]
        .nunique()
        .reset_index()
        .rename(columns={"prescriber_id": "member_unique_prescribers"})
    )
    df = df.merge(member_prescribers, on="member_id", how="left")

    # Member controlled substance ratio
    member_totals = df.groupby("member_id")["claim_id"].count().reset_index()
    member_totals.columns = ["member_id", "member_total_claims"]
    member_ctrl = (
        df[df["dea_schedule"] != "NONE"]
        .groupby("member_id")["claim_id"]
        .count()
        .reset_index()
    )
    member_ctrl.columns = ["member_id", "member_ctrl_count"]
    member_totals = member_totals.merge(member_ctrl, on="member_id", how="left")
    member_totals["member_ctrl_count"] = member_totals["member_ctrl_count"].fillna(0)
    member_totals["member_controlled_ratio"] = (
        member_totals["member_ctrl_count"] / member_totals["member_total_claims"]
    )
    df = df.merge(
        member_totals[["member_id", "member_total_claims", "member_controlled_ratio"]],
        on="member_id",
        how="left",
    )

    # Doctor shopping flag: ≥3 prescribers for same drug class
    df["doctor_shopping_signal"] = (df["member_class_unique_prescribers"] >= 3).astype(int)

    # Pharmacy shopping flag: ≥3 pharmacies overall
    df["pharmacy_shopping_signal"] = (df["member_unique_pharmacies"] >= 3).astype(int)

    return df
