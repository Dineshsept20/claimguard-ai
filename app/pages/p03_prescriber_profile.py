"""Page 3: Prescriber Risk Profile.

Search by prescriber ID or name, view risk breakdown,
prescribing patterns, and peer comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.data_loader import load_scored_claims, load_feature_claims, load_prescribers
from app.components.charts import prescriber_radar, top_risky_entities, COLORS


def render():
    st.markdown("# 👨‍⚕️ Prescriber Risk Profile")
    st.markdown("Investigate prescriber prescribing patterns and compare to peers")
    st.markdown("---")

    scored = load_scored_claims()
    prescribers = load_prescribers()

    try:
        features = load_feature_claims()
    except FileNotFoundError:
        features = None

    # --- Prescriber Search ---
    col_search, col_info = st.columns([2, 3])

    with col_search:
        search = st.text_input("🔎 Search Prescriber (ID or Name)", "")

    # Build prescriber list with risk scores
    presc_risk = (
        scored.groupby("prescriber_id")
        .agg(
            total_claims=("claim_id", "count"),
            flagged=("ensemble_prediction", "sum"),
            avg_score=("ensemble_score", "mean"),
            total_cost=("total_cost", "sum"),
        )
        .reset_index()
        .merge(prescribers[["prescriber_id", "full_name", "specialty", "npi", "state"]], on="prescriber_id", how="left")
        .sort_values("avg_score", ascending=False)
    )

    if search:
        mask = (
            presc_risk["prescriber_id"].astype(str).str.contains(search, case=False)
            | presc_risk["full_name"].astype(str).str.contains(search, case=False)
            | presc_risk["npi"].astype(str).str.contains(search, case=False)
        )
        presc_risk = presc_risk[mask]

    with col_info:
        st.markdown(f"**{len(presc_risk):,}** prescribers found")

    # --- Top Prescribers Table ---
    st.markdown("### Top Prescribers by Risk Score")
    display = presc_risk.head(20).copy()
    display["flag_rate"] = (display["flagged"] / display["total_claims"] * 100).round(1)
    display["avg_score"] = display["avg_score"].round(4)

    st.dataframe(
        display[["prescriber_id", "full_name", "specialty", "state",
                 "total_claims", "flagged", "flag_rate", "avg_score"]],
        width="stretch",
        hide_index=True,
        column_config={
            "avg_score": st.column_config.ProgressColumn(
                "Avg Risk Score", format="%.4f", min_value=0, max_value=1,
            ),
            "flag_rate": st.column_config.NumberColumn("Flag Rate %", format="%.1f%%"),
        },
    )

    # --- Prescriber Detail ---
    st.markdown("---")
    st.markdown("### 📋 Prescriber Detail")

    presc_ids = presc_risk["prescriber_id"].head(50).tolist()
    if not presc_ids:
        st.warning("No prescribers match your search.")
        return

    selected = st.selectbox("Select prescriber", presc_ids,
                            format_func=lambda x: f"{x} — {presc_risk[presc_risk['prescriber_id'] == x]['full_name'].values[0]}")

    presc_info = prescribers[prescribers["prescriber_id"] == selected].iloc[0]
    presc_stats = presc_risk[presc_risk["prescriber_id"] == selected].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", f"{int(presc_stats['total_claims']):,}")
    with col2:
        st.metric("Flagged Claims", f"{int(presc_stats['flagged']):,}")
    with col3:
        st.metric("Avg Risk Score", f"{presc_stats['avg_score']:.4f}")
    with col4:
        flag_rate = presc_stats['flagged'] / max(presc_stats['total_claims'], 1) * 100
        st.metric("Flag Rate", f"{flag_rate:.1f}%")

    # Info card
    with st.expander("Prescriber Information", expanded=True):
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"**Name:** {presc_info['full_name']}")
            st.markdown(f"**NPI:** {presc_info['npi']}")
            st.markdown(f"**DEA:** {presc_info.get('dea_number', 'N/A')}")
        with info_col2:
            st.markdown(f"**Specialty:** {presc_info['specialty']}")
            st.markdown(f"**State:** {presc_info['state']}")
            st.markdown(f"**Practice:** {presc_info.get('practice_type', 'N/A')}")

    # --- Prescriber Claims Breakdown ---
    presc_claims = scored[scored["prescriber_id"] == selected]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Anomaly Type Breakdown**")
        type_counts = presc_claims[presc_claims["ensemble_prediction"] == 1]["anomaly_type"].value_counts()
        if len(type_counts) > 0:
            st.bar_chart(type_counts)
        else:
            st.info("No flagged claims for this prescriber.")

    with col_right:
        st.markdown("**Top Drugs Prescribed**")
        drug_counts = presc_claims["drug_name"].value_counts().head(10)
        st.bar_chart(drug_counts)

    # --- Peer Comparison Radar ---
    if features is not None:
        st.markdown("---")
        st.markdown("### 🎯 Peer Comparison")

        presc_features_cols = [
            "presc_controlled_ratio", "presc_top_pharmacy_pct",
            "presc_after_hours_ratio", "presc_cost_peer_zscore",
            "presc_claims_per_member",
        ]

        available_cols = [c for c in presc_features_cols if c in features.columns]

        if available_cols:
            presc_data = features[features["prescriber_id"] == selected][available_cols]
            if len(presc_data) > 0:
                presc_row = presc_data.iloc[0]

                # Peer = same specialty
                specialty = presc_info["specialty"]
                peers = features[
                    features["prescriber_id"].isin(
                        prescribers[prescribers["specialty"] == specialty]["prescriber_id"]
                    )
                ][available_cols]
                peer_avg = peers.mean()

                fig = prescriber_radar(presc_row, peer_avg)
                st.plotly_chart(fig, width="stretch")

    # --- Pharmacy Network ---
    st.markdown("---")
    st.markdown("### 🏪 Pharmacy Relationships")

    pharm_network = (
        presc_claims.groupby("pharmacy_id")
        .agg(claims=("claim_id", "count"), flagged=("ensemble_prediction", "sum"))
        .sort_values("claims", ascending=False)
        .head(10)
        .reset_index()
    )
    pharm_network["pct"] = (pharm_network["claims"] / pharm_network["claims"].sum() * 100).round(1)

    st.dataframe(pharm_network, width="stretch", hide_index=True,
                 column_config={
                     "pct": st.column_config.NumberColumn("% of Total", format="%.1f%%"),
                 })
