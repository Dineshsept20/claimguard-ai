"""Page 4: Model Performance.

Displays ROC curves, PR curves, confusion matrices,
detection rate by anomaly type, and model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.data_loader import load_scored_claims, load_xgb_model
from app.components.charts import (
    roc_curve_chart,
    pr_curve_chart,
    confusion_matrix_chart,
    detection_rate_chart,
)
from src.utils.metrics import compute_overall_metrics, compute_per_type_detection_rate


def render():
    st.markdown("# 📈 Model Performance")
    st.markdown("Evaluate and compare anomaly detection models")
    st.markdown("---")

    df = load_scored_claims()
    y_true = df["is_anomaly"].astype(int).values

    # --- Model Comparison Table ---
    st.markdown("### 🏆 Model Comparison")

    models_data = []

    # Isolation Forest
    if "if_prediction" in df.columns:
        m = compute_overall_metrics(y_true, df["if_prediction"].values, df["if_anomaly_score"].values)
        m["model"] = "Isolation Forest"
        models_data.append(m)

    # XGBoost
    if "xgb_prediction" in df.columns:
        m = compute_overall_metrics(y_true, df["xgb_prediction"].values, df["xgb_probability"].values)
        m["model"] = "XGBoost"
        models_data.append(m)

    # Business Rules (≥2 flags)
    if "rule_flags_count" in df.columns:
        rule_pred = (df["rule_flags_count"] >= 2).astype(int).values
        rule_score = df["rule_flags_count"] / df["rule_flags_count"].max()
        m = compute_overall_metrics(y_true, rule_pred, rule_score.values)
        m["model"] = "Business Rules (≥2)"
        models_data.append(m)

    # Ensemble
    if "ensemble_prediction" in df.columns:
        m = compute_overall_metrics(y_true, df["ensemble_prediction"].values, df["ensemble_score"].values)
        m["model"] = "Ensemble"
        models_data.append(m)

    if models_data:
        comparison = pd.DataFrame(models_data)
        display_cols = ["model", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "accuracy"]
        avail = [c for c in display_cols if c in comparison.columns]
        comp_display = comparison[avail].copy()

        # Format numeric columns
        for col in avail:
            if col != "model":
                comp_display[col] = comp_display[col].round(4)

        st.dataframe(comp_display, width="stretch", hide_index=True)

        # Highlight best
        best_f1_model = comparison.loc[comparison["f1_score"].idxmax(), "model"]
        best_f1 = comparison["f1_score"].max()
        st.success(f"🏆 Best F1 Score: **{best_f1_model}** ({best_f1:.4f})")

    st.markdown("---")

    # --- ROC & PR Curves ---
    st.markdown("### 📉 ROC & Precision-Recall Curves")

    y_prob_dict = {}
    if "if_anomaly_score" in df.columns:
        y_prob_dict["Isolation Forest"] = df["if_anomaly_score"].values
    if "xgb_probability" in df.columns:
        y_prob_dict["XGBoost"] = df["xgb_probability"].values
    if "ensemble_score" in df.columns:
        y_prob_dict["Ensemble"] = df["ensemble_score"].values

    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.plotly_chart(roc_curve_chart(y_true, y_prob_dict), width="stretch")
    with col_pr:
        st.plotly_chart(pr_curve_chart(y_true, y_prob_dict), width="stretch")

    # --- Confusion Matrices ---
    st.markdown("---")
    st.markdown("### 🔢 Confusion Matrices")

    cm_cols = st.columns(3)

    with cm_cols[0]:
        if "if_prediction" in df.columns:
            st.plotly_chart(
                confusion_matrix_chart(y_true, df["if_prediction"].values, "Isolation Forest"),
                width="stretch",
            )

    with cm_cols[1]:
        if "xgb_prediction" in df.columns:
            st.plotly_chart(
                confusion_matrix_chart(y_true, df["xgb_prediction"].values, "XGBoost"),
                width="stretch",
            )

    with cm_cols[2]:
        if "ensemble_prediction" in df.columns:
            st.plotly_chart(
                confusion_matrix_chart(y_true, df["ensemble_prediction"].values, "Ensemble"),
                width="stretch",
            )

    # --- Detection Rate by Anomaly Type ---
    st.markdown("---")
    st.markdown("### 🎯 Detection Rate by Anomaly Type")

    model_choice = st.selectbox("Select model", ["Ensemble", "XGBoost", "Isolation Forest"])
    pred_col = {
        "Ensemble": "ensemble_prediction",
        "XGBoost": "xgb_prediction",
        "Isolation Forest": "if_prediction",
    }[model_choice]

    if pred_col in df.columns:
        per_type = compute_per_type_detection_rate(df, pred_col, "anomaly_type")
        st.plotly_chart(detection_rate_chart(per_type), width="stretch")

        # Detail table
        per_type_display = per_type.copy()
        per_type_display["detection_rate"] = (per_type_display["detection_rate"] * 100).round(1)
        per_type_display.columns = ["Anomaly Type", "Total", "Detected", "Detection %"]
        st.dataframe(per_type_display, width="stretch", hide_index=True)

    # --- XGBoost Feature Importance ---
    st.markdown("---")
    st.markdown("### 🌲 XGBoost Feature Importance")

    try:
        artifact = load_xgb_model()
        importance = artifact.get("feature_importance", {})
        if importance:
            imp_df = (
                pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
                .sort_values("Importance", ascending=False)
                .head(15)
            )
            st.bar_chart(imp_df.set_index("Feature")["Importance"])
    except FileNotFoundError:
        st.info("XGBoost model not found. Train the model first.")
