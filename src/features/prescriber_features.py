"""Prescriber-level behavioral feature engineering.

Computes aggregated prescriber profiles over the claims data,
capturing prescribing patterns that may indicate fraud, waste,
or abuse when they deviate significantly from peers.
"""

import numpy as np
import pandas as pd


def build_prescriber_features(
    claims_df: pd.DataFrame,
    prescribers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Engineer prescriber-level behavioral features.

    Aggregates claims data at the prescriber level to build profiles
    capturing volume, controlled substance patterns, pharmacy
    concentration, and cost behaviors.

    Args:
        claims_df: Claims DataFrame (ideally with claim-level features).
        prescribers_df: Prescribers DataFrame.

    Returns:
        DataFrame indexed by prescriber_id with behavioral features.
    """
    df = claims_df.copy()
    if df["service_date"].dtype == object:
        df["service_date"] = pd.to_datetime(df["service_date"])

    controlled_mask = df["dea_schedule"] != "NONE"
    opioid_mask = df["therapeutic_class"] == "OPIOID_ANALGESIC"

    # --- Core aggregation ---
    agg = df.groupby("prescriber_id").agg(
        presc_total_claims=("claim_id", "count"),
        presc_unique_members=("member_id", "nunique"),
        presc_unique_pharmacies=("pharmacy_id", "nunique"),
        presc_avg_cost=("total_cost", "mean"),
        presc_total_cost=("total_cost", "sum"),
        presc_avg_quantity=("quantity", "mean"),
        presc_median_cost=("total_cost", "median"),
    ).reset_index()

    # --- Controlled substance ratio ---
    ctrl_counts = df[controlled_mask].groupby("prescriber_id")["claim_id"].count().reset_index()
    ctrl_counts.columns = ["prescriber_id", "presc_controlled_count"]
    agg = agg.merge(ctrl_counts, on="prescriber_id", how="left")
    agg["presc_controlled_count"] = agg["presc_controlled_count"].fillna(0)
    agg["presc_controlled_ratio"] = agg["presc_controlled_count"] / agg["presc_total_claims"]

    # --- Top pharmacy concentration ---
    # What % of this prescriber's claims go to their most-used pharmacy?
    top_pharm = (
        df.groupby(["prescriber_id", "pharmacy_id"])["claim_id"]
        .count()
        .reset_index()
        .sort_values(["prescriber_id", "claim_id"], ascending=[True, False])
    )
    top_pharm_max = top_pharm.groupby("prescriber_id")["claim_id"].first().reset_index()
    top_pharm_max.columns = ["prescriber_id", "presc_top_pharmacy_count"]
    agg = agg.merge(top_pharm_max, on="prescriber_id", how="left")
    agg["presc_top_pharmacy_pct"] = agg["presc_top_pharmacy_count"] / agg["presc_total_claims"]
    agg.drop(columns=["presc_top_pharmacy_count"], inplace=True)

    # --- Opioid MME per patient ---
    opioid_claims = df[opioid_mask].copy()
    if len(opioid_claims) > 0:
        opioid_claims["mme_total"] = opioid_claims["mme_factor"] * opioid_claims["quantity"]
        mme_by_presc_member = (
            opioid_claims.groupby(["prescriber_id", "member_id"])["mme_total"]
            .sum()
            .reset_index()
        )
        mme_per_patient = (
            mme_by_presc_member.groupby("prescriber_id")["mme_total"]
            .mean()
            .reset_index()
        )
        mme_per_patient.columns = ["prescriber_id", "presc_avg_mme_per_patient"]
        agg = agg.merge(mme_per_patient, on="prescriber_id", how="left")
    agg["presc_avg_mme_per_patient"] = agg.get("presc_avg_mme_per_patient", 0).fillna(0)

    # --- Weekend prescribing ratio ---
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["service_date"].dt.weekday >= 5).astype(int)
    weekend_counts = df[df["is_weekend"] == 1].groupby("prescriber_id")["claim_id"].count().reset_index()
    weekend_counts.columns = ["prescriber_id", "presc_weekend_count"]
    agg = agg.merge(weekend_counts, on="prescriber_id", how="left")
    agg["presc_weekend_count"] = agg["presc_weekend_count"].fillna(0)
    agg["presc_weekend_ratio"] = agg["presc_weekend_count"] / agg["presc_total_claims"]
    agg.drop(columns=["presc_weekend_count"], inplace=True)

    # --- After-hours ratio ---
    after_hours = df[(df["submit_hour"] < 8) | (df["submit_hour"] > 18)]
    ah_counts = after_hours.groupby("prescriber_id")["claim_id"].count().reset_index()
    ah_counts.columns = ["prescriber_id", "presc_after_hours_count"]
    agg = agg.merge(ah_counts, on="prescriber_id", how="left")
    agg["presc_after_hours_count"] = agg["presc_after_hours_count"].fillna(0)
    agg["presc_after_hours_ratio"] = agg["presc_after_hours_count"] / agg["presc_total_claims"]
    agg.drop(columns=["presc_after_hours_count"], inplace=True)

    # --- Peer deviation: cost vs same-specialty peers ---
    # Merge specialty from prescribers table
    agg = agg.merge(
        prescribers_df[["prescriber_id", "specialty"]],
        on="prescriber_id",
        how="left",
    )
    specialty_avg = agg.groupby("specialty")["presc_avg_cost"].transform("mean")
    specialty_std = agg.groupby("specialty")["presc_avg_cost"].transform("std").replace(0, 1)
    agg["presc_cost_peer_zscore"] = (agg["presc_avg_cost"] - specialty_avg) / specialty_std

    # --- Claims per member (intensity) ---
    agg["presc_claims_per_member"] = agg["presc_total_claims"] / agg["presc_unique_members"].replace(0, 1)

    return agg
