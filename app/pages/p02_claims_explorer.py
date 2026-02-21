"""Page 2: Claims Explorer.

Filterable table of flagged claims with drill-down,
filters, and CSV export.
"""

import streamlit as st
import pandas as pd

from app.data_loader import load_scored_claims


def render():
    st.markdown("# 🔍 Claims Explorer")
    st.markdown("Filter and investigate flagged pharmacy claims")
    st.markdown("---")

    df = load_scored_claims()

    # --- Filters ---
    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            min_score = st.slider("Min Risk Score", 0.0, 1.0, 0.3, 0.05)
        with col2:
            anomaly_types = ["All"] + sorted(df["anomaly_type"].dropna().unique().tolist())
            selected_type = st.selectbox("Anomaly Type", anomaly_types)
        with col3:
            risk_tiers = ["All"] + ["Critical", "High", "Medium", "Low"]
            selected_tier = st.selectbox("Risk Tier", risk_tiers)

        col4, col5 = st.columns(2)
        with col4:
            pharmacy_filter = st.text_input("Pharmacy ID (contains)", "")
        with col5:
            prescriber_filter = st.text_input("Prescriber ID (contains)", "")

    # Apply filters
    filtered = df[df["ensemble_score"] >= min_score].copy()

    if selected_type != "All":
        filtered = filtered[filtered["anomaly_type"] == selected_type]

    if selected_tier != "All":
        filtered = filtered[filtered["risk_tier"] == selected_tier]

    if pharmacy_filter:
        filtered = filtered[filtered["pharmacy_id"].astype(str).str.contains(pharmacy_filter, case=False)]

    if prescriber_filter:
        filtered = filtered[filtered["prescriber_id"].astype(str).str.contains(prescriber_filter, case=False)]

    # Sort by risk
    filtered = filtered.sort_values("ensemble_score", ascending=False)

    # --- Results Summary ---
    st.markdown(f"### Showing **{len(filtered):,}** claims (of {len(df):,} total)")

    # Quick stats
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Filtered Claims", f"{len(filtered):,}")
    with col_b:
        st.metric("True Anomalies", f"{int(filtered['is_anomaly'].sum()):,}")
    with col_c:
        st.metric("Avg Risk Score", f"{filtered['ensemble_score'].mean():.3f}" if len(filtered) > 0 else "N/A")
    with col_d:
        cost = filtered["total_cost"].sum()
        st.metric("Total Cost", f"${cost:,.0f}")

    st.markdown("---")

    # --- Claims Table ---
    display_cols = [
        "claim_id", "service_date", "member_id", "prescriber_id",
        "pharmacy_id", "drug_name", "quantity", "total_cost",
        "ensemble_score", "risk_tier", "anomaly_type",
    ]
    avail_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[avail_cols].head(500),
        width="stretch",
        hide_index=True,
        column_config={
            "ensemble_score": st.column_config.ProgressColumn(
                "Risk Score", format="%.3f", min_value=0, max_value=1,
            ),
            "total_cost": st.column_config.NumberColumn("Total Cost", format="$%.2f"),
            "service_date": st.column_config.DateColumn("Service Date"),
        },
    )

    # --- Claim Detail Drill-Down ---
    st.markdown("---")
    st.markdown("### 🔎 Claim Detail")

    claim_ids = filtered["claim_id"].head(100).tolist()
    if claim_ids:
        selected_claim = st.selectbox("Select a claim to inspect", claim_ids)
        claim_row = df[df["claim_id"] == selected_claim].iloc[0]

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Claim Details**")
            details = {
                "Claim ID": claim_row["claim_id"],
                "Service Date": str(claim_row["service_date"])[:10],
                "Drug": claim_row.get("drug_name", "N/A"),
                "Therapeutic Class": claim_row.get("therapeutic_class", "N/A"),
                "Quantity": claim_row.get("quantity", "N/A"),
                "Days Supply": claim_row.get("days_supply", "N/A"),
                "Total Cost": f"${claim_row.get('total_cost', 0):,.2f}",
            }
            for k, v in details.items():
                st.markdown(f"**{k}:** {v}")

        with col_r:
            st.markdown("**Risk Scores**")
            scores = {
                "Ensemble Score": f"{claim_row.get('ensemble_score', 0):.4f}",
                "XGBoost Probability": f"{claim_row.get('xgb_probability', 0):.4f}",
                "IF Anomaly Score": f"{claim_row.get('if_anomaly_score', 0):.4f}",
                "Rule Flags": f"{claim_row.get('rule_flags_count', 0):.0f}",
                "Risk Tier": claim_row.get("risk_tier", "N/A"),
                "Is Anomaly (ground truth)": "✅ Yes" if claim_row.get("is_anomaly") else "❌ No",
                "Anomaly Type": claim_row.get("anomaly_type", "N/A"),
            }
            for k, v in scores.items():
                st.markdown(f"**{k}:** {v}")

    # --- CSV Export ---
    st.markdown("---")
    csv = filtered[avail_cols].to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Claims as CSV",
        csv,
        "claimguard_flagged_claims.csv",
        "text/csv",
    )
