"""Pharmacy-level behavioral feature engineering.

Computes aggregated pharmacy profiles capturing dispensing patterns,
cost behaviors, and relationship metrics that signal potential fraud,
waste, or abuse.
"""

import numpy as np
import pandas as pd


def build_pharmacy_features(
    claims_df: pd.DataFrame,
    pharmacies_df: pd.DataFrame,
) -> pd.DataFrame:
    """Engineer pharmacy-level behavioral features.

    Args:
        claims_df: Claims DataFrame.
        pharmacies_df: Pharmacies DataFrame.

    Returns:
        DataFrame indexed by pharmacy_id with behavioral features.
    """
    df = claims_df.copy()
    if df["service_date"].dtype == object:
        df["service_date"] = pd.to_datetime(df["service_date"])

    controlled_mask = df["dea_schedule"] != "NONE"

    # --- Core aggregation ---
    agg = df.groupby("pharmacy_id").agg(
        pharm_total_claims=("claim_id", "count"),
        pharm_unique_prescribers=("prescriber_id", "nunique"),
        pharm_unique_members=("member_id", "nunique"),
        pharm_avg_cost=("total_cost", "mean"),
        pharm_total_cost=("total_cost", "sum"),
        pharm_median_cost=("total_cost", "median"),
        pharm_avg_quantity=("quantity", "mean"),
    ).reset_index()

    # --- Controlled substance ratio ---
    ctrl_counts = df[controlled_mask].groupby("pharmacy_id")["claim_id"].count().reset_index()
    ctrl_counts.columns = ["pharmacy_id", "pharm_controlled_count"]
    agg = agg.merge(ctrl_counts, on="pharmacy_id", how="left")
    agg["pharm_controlled_count"] = agg["pharm_controlled_count"].fillna(0)
    agg["pharm_controlled_ratio"] = agg["pharm_controlled_count"] / agg["pharm_total_claims"]

    # --- Reversal rate ---
    reversed_counts = (
        df[df["claim_status"] == "reversed"]
        .groupby("pharmacy_id")["claim_id"]
        .count()
        .reset_index()
    )
    reversed_counts.columns = ["pharmacy_id", "pharm_reversed_count"]
    agg = agg.merge(reversed_counts, on="pharmacy_id", how="left")
    agg["pharm_reversed_count"] = agg["pharm_reversed_count"].fillna(0)
    agg["pharm_reversal_rate"] = agg["pharm_reversed_count"] / agg["pharm_total_claims"]
    agg.drop(columns=["pharm_reversed_count"], inplace=True)

    # --- Rejected rate ---
    rejected_counts = (
        df[df["claim_status"] == "rejected"]
        .groupby("pharmacy_id")["claim_id"]
        .count()
        .reset_index()
    )
    rejected_counts.columns = ["pharmacy_id", "pharm_rejected_count"]
    agg = agg.merge(rejected_counts, on="pharmacy_id", how="left")
    agg["pharm_rejected_count"] = agg["pharm_rejected_count"].fillna(0)
    agg["pharm_rejection_rate"] = agg["pharm_rejected_count"] / agg["pharm_total_claims"]
    agg.drop(columns=["pharm_rejected_count"], inplace=True)

    # --- Brand when generic available ratio ---
    brand_counts = (
        df[df["is_brand"] == True]
        .groupby("pharmacy_id")["claim_id"]
        .count()
        .reset_index()
    )
    brand_counts.columns = ["pharmacy_id", "pharm_brand_count"]
    agg = agg.merge(brand_counts, on="pharmacy_id", how="left")
    agg["pharm_brand_count"] = agg["pharm_brand_count"].fillna(0)
    agg["pharm_brand_ratio"] = agg["pharm_brand_count"] / agg["pharm_total_claims"]
    agg.drop(columns=["pharm_brand_count"], inplace=True)

    # --- After-hours ratio ---
    after_hours = df[(df["submit_hour"] < 8) | (df["submit_hour"] > 18)]
    ah_counts = after_hours.groupby("pharmacy_id")["claim_id"].count().reset_index()
    ah_counts.columns = ["pharmacy_id", "pharm_after_hours_count"]
    agg = agg.merge(ah_counts, on="pharmacy_id", how="left")
    agg["pharm_after_hours_count"] = agg["pharm_after_hours_count"].fillna(0)
    agg["pharm_after_hours_ratio"] = agg["pharm_after_hours_count"] / agg["pharm_total_claims"]
    agg.drop(columns=["pharm_after_hours_count"], inplace=True)

    # --- Top prescriber concentration ---
    top_presc = (
        df.groupby(["pharmacy_id", "prescriber_id"])["claim_id"]
        .count()
        .reset_index()
        .sort_values(["pharmacy_id", "claim_id"], ascending=[True, False])
    )
    top_presc_max = top_presc.groupby("pharmacy_id")["claim_id"].first().reset_index()
    top_presc_max.columns = ["pharmacy_id", "pharm_top_prescriber_count"]
    agg = agg.merge(top_presc_max, on="pharmacy_id", how="left")
    agg["pharm_top_prescriber_pct"] = (
        agg["pharm_top_prescriber_count"] / agg["pharm_total_claims"]
    )
    agg.drop(columns=["pharm_top_prescriber_count"], inplace=True)

    # --- Merge pharmacy type from entity table ---
    agg = agg.merge(
        pharmacies_df[["pharmacy_id", "pharmacy_type", "state"]],
        on="pharmacy_id",
        how="left",
    )

    # --- Peer deviation: cost vs same-type pharmacies in same state ---
    peer_avg = agg.groupby(["pharmacy_type", "state"])["pharm_avg_cost"].transform("mean")
    peer_std = agg.groupby(["pharmacy_type", "state"])["pharm_avg_cost"].transform("std").replace(0, 1)
    agg["pharm_cost_peer_zscore"] = (agg["pharm_avg_cost"] - peer_avg) / peer_std

    # --- Volume peer deviation ---
    vol_avg = agg.groupby("pharmacy_type")["pharm_total_claims"].transform("mean")
    vol_std = agg.groupby("pharmacy_type")["pharm_total_claims"].transform("std").replace(0, 1)
    agg["pharm_volume_peer_zscore"] = (agg["pharm_total_claims"] - vol_avg) / vol_std

    # --- Claims per member (density) ---
    agg["pharm_claims_per_member"] = (
        agg["pharm_total_claims"] / agg["pharm_unique_members"].replace(0, 1)
    )

    return agg
