"""Page 1: Executive Overview.

Displays KPI metrics, anomaly breakdown, monthly trends,
and top risky entities.
"""

import streamlit as st
import pandas as pd

from app.data_loader import load_scored_claims, load_prescribers, load_pharmacies, format_currency, format_number
from app.components.charts import (
    anomaly_breakdown_pie,
    risk_tier_bar,
    monthly_trend,
    top_risky_entities,
)


def render():
    st.markdown("# 📊 Executive Overview")
    st.markdown("Real-time pharmacy claims anomaly detection summary")
    st.markdown("---")

    df = load_scored_claims()
    prescribers = load_prescribers()
    pharmacies = load_pharmacies()

    # --- KPI Metrics ---
    total_claims = len(df)
    flagged = int(df["ensemble_prediction"].sum())
    flag_rate = flagged / total_claims * 100
    total_cost = df["total_cost"].sum()
    flagged_cost = df.loc[df["ensemble_prediction"] == 1, "total_cost"].sum()
    critical = int((df["risk_tier"] == "Critical").sum())

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Claims", format_number(total_claims))
    with col2:
        st.metric("Flagged Claims", format_number(flagged), f"{flag_rate:.1f}%")
    with col3:
        st.metric("Estimated Savings", format_currency(flagged_cost * 0.3),
                   help="30% of flagged claim costs (industry benchmark)")
    with col4:
        st.metric("Critical Alerts", format_number(critical), delta=None)
    with col5:
        avg_score = df.loc[df["ensemble_prediction"] == 1, "ensemble_score"].mean()
        st.metric("Avg Risk Score", f"{avg_score:.3f}")

    st.markdown("---")

    # --- Charts Row 1: Anomaly Breakdown + Risk Tiers ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(anomaly_breakdown_pie(df), width="stretch")

    with col_right:
        st.plotly_chart(risk_tier_bar(df), width="stretch")

    # --- Chart Row 2: Monthly Trend ---
    st.plotly_chart(monthly_trend(df), width="stretch")

    # --- Chart Row 3: Top Risky Entities ---
    col_pharm, col_presc = st.columns(2)

    with col_pharm:
        st.plotly_chart(
            top_risky_entities(df, "pharmacy_id", top_n=10),
            width="stretch",
        )

    with col_presc:
        st.plotly_chart(
            top_risky_entities(df, "prescriber_id", top_n=10),
            width="stretch",
        )

    # --- Summary Table ---
    st.markdown("### 📋 Anomaly Type Summary")
    anomaly_summary = (
        df[df["is_anomaly"] == 1]
        .groupby("anomaly_type")
        .agg(
            count=("claim_id", "count"),
            detected=("ensemble_prediction", "sum"),
            avg_score=("ensemble_score", "mean"),
            total_cost=("total_cost", "sum"),
        )
        .reset_index()
    )
    anomaly_summary["detection_rate"] = (anomaly_summary["detected"] / anomaly_summary["count"] * 100).round(1)
    anomaly_summary["avg_score"] = anomaly_summary["avg_score"].round(3)
    anomaly_summary["total_cost"] = anomaly_summary["total_cost"].apply(format_currency)
    anomaly_summary.columns = ["Anomaly Type", "Total", "Detected", "Avg Score", "Total Cost", "Detection %"]

    st.dataframe(anomaly_summary, width="stretch", hide_index=True)
