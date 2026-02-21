"""Page 5: Explainability Deep Dive.

Global SHAP importance, per-claim waterfall explanations,
natural language explanations, and what-if analysis.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

from app.data_loader import load_scored_claims, load_feature_claims, load_xgb_model, load_shap_cache
from app.components.charts import shap_importance_bar, shap_waterfall_chart, shap_dependence_chart


def _ensure_shap_values():
    """Compute SHAP values if not cached."""
    cache = load_shap_cache()
    if cache is not None:
        return cache

    st.info("Computing SHAP values (first-time setup, ~30s)...")

    from src.explainability.shap_explainer import compute_shap_values

    artifact = load_xgb_model()
    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["features"]

    df = load_feature_claims()
    X = df[features].fillna(0).replace([np.inf, -np.inf], 0).values
    X_scaled = scaler.transform(X)

    explanation = compute_shap_values(model, X_scaled, features, max_samples=5000)

    # Reload the cache
    cache = load_shap_cache.__wrapped__()  # bypass cache
    return cache


def render():
    st.markdown("# 🧠 Explainability Deep Dive")
    st.markdown("Understand *why* claims are flagged using SHAP and business rules")
    st.markdown("---")

    # Load data
    scored = load_scored_claims()
    artifact = load_xgb_model()
    features = artifact["features"]

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Global Importance",
        "🔎 Claim Explanation",
        "📊 Dependence Plots",
        "🔮 What-If Analysis",
    ])

    # --- Tab 1: Global Feature Importance ---
    with tab1:
        st.markdown("### Global SHAP Feature Importance")
        st.markdown("Which features contribute most to the model's fraud detection decisions?")

        cache = _ensure_shap_values()
        if cache:
            from src.explainability.shap_explainer import build_explanation_from_cache, get_global_importance

            explanation = build_explanation_from_cache(cache)
            importance = get_global_importance(explanation)

            st.plotly_chart(shap_importance_bar(importance, top_n=15), width="stretch")

            with st.expander("📋 Full Feature Importance Table"):
                st.dataframe(importance, width="stretch", hide_index=True)
        else:
            st.warning("SHAP values not available. Run the training pipeline first.")

    # --- Tab 2: Per-Claim Explanation ---
    with tab2:
        st.markdown("### Per-Claim SHAP Explanation")
        st.markdown("Select a flagged claim to see why it was flagged.")

        # Show flagged claims sorted by risk
        flagged = scored[scored["ensemble_prediction"] == 1].sort_values("ensemble_score", ascending=False)
        claim_options = flagged["claim_id"].head(200).tolist()

        if not claim_options:
            st.warning("No flagged claims found.")
        else:
            selected_claim = st.selectbox("Select a flagged claim", claim_options,
                                          format_func=lambda x: f"{x} (score: {scored[scored['claim_id']==x]['ensemble_score'].values[0]:.4f})")

            cache = _ensure_shap_values()
            if cache:
                from src.explainability.shap_explainer import (
                    build_explanation_from_cache,
                    get_claim_explanation,
                    generate_natural_language_explanation,
                )

                explanation = build_explanation_from_cache(cache)

                # Find nearest sample index
                sample_idx = min(0, len(explanation.values) - 1)

                # Show waterfall
                claim_detail = get_claim_explanation(explanation, sample_idx)
                base_value = explanation.base_values
                if isinstance(base_value, np.ndarray):
                    base_value = float(base_value[0]) if len(base_value) > 0 else 0.0

                col_chart, col_text = st.columns([3, 2])

                with col_chart:
                    st.plotly_chart(
                        shap_waterfall_chart(claim_detail, base_value, top_n=10),
                        width="stretch",
                    )

                with col_text:
                    # NL Explanation
                    st.markdown("#### 💬 Natural Language Explanation")
                    claim_row = scored[scored["claim_id"] == selected_claim].iloc[0]
                    nl = generate_natural_language_explanation(
                        explanation, sample_idx, claim_row, top_n=5
                    )
                    st.markdown(nl)

                # Feature details table
                with st.expander("📋 All Feature Contributions"):
                    st.dataframe(claim_detail, width="stretch", hide_index=True)

    # --- Tab 3: Dependence Plots ---
    with tab3:
        st.markdown("### SHAP Dependence Plots")
        st.markdown("How does each feature value affect the risk score?")

        cache = _ensure_shap_values()
        if cache:
            from src.explainability.shap_explainer import build_explanation_from_cache, get_dependence_data

            explanation = build_explanation_from_cache(cache)

            col_feat, col_int = st.columns(2)
            with col_feat:
                feat = st.selectbox("Primary Feature", features)
            with col_int:
                int_feat = st.selectbox("Color by (interaction)", ["None"] + list(features))

            int_param = int_feat if int_feat != "None" else None
            dep_data = get_dependence_data(explanation, feat, int_param)
            st.plotly_chart(shap_dependence_chart(dep_data, feat, int_param), width="stretch")
        else:
            st.warning("SHAP values not available.")

    # --- Tab 4: What-If Analysis ---
    with tab4:
        st.markdown("### What-If Analysis")
        st.markdown("Adjust feature values to see how the risk score changes.")

        scaler = artifact["scaler"]
        model = artifact["model"]

        st.markdown("#### Modify Feature Values")

        # Create input widgets for key features
        input_values = {}
        key_features = [
            ("cost_vs_awp_ratio", 0.5, 5.0, 1.0, "Cost relative to AWP"),
            ("quantity_vs_typical", 0.5, 5.0, 1.0, "Quantity relative to typical"),
            ("doctor_shopping_signal", 0, 5, 0, "Doctor shopping signal"),
            ("pharmacy_shopping_signal", 0, 5, 0, "Pharmacy shopping signal"),
            ("rule_flags_count", 0, 10, 0, "Business rule flags"),
            ("opioid_mme_daily", 0.0, 200.0, 0.0, "Daily MME"),
            ("is_controlled", 0, 1, 0, "Controlled substance"),
            ("is_after_hours", 0, 1, 0, "After hours"),
        ]

        cols = st.columns(4)
        for i, (feat, min_v, max_v, default, label) in enumerate(key_features):
            with cols[i % 4]:
                if isinstance(default, int) and max_v <= 10:
                    input_values[feat] = st.number_input(label, min_value=min_v, max_value=max_v,
                                                         value=default, step=1, key=f"whatif_{feat}")
                else:
                    input_values[feat] = st.slider(label, min_value=float(min_v), max_value=float(max_v),
                                                   value=float(default), step=0.1, key=f"whatif_{feat}")

        if st.button("🔮 Predict Risk Score", type="primary"):
            # Build feature vector
            feat_vector = np.zeros((1, len(features)))
            for j, f in enumerate(features):
                if f in input_values:
                    feat_vector[0, j] = input_values[f]

            feat_scaled = scaler.transform(feat_vector)
            prob = model.predict_proba(feat_scaled)[0, 1]

            # Display result
            if prob >= 0.8:
                tier, color = "Critical", "🔴"
            elif prob >= 0.6:
                tier, color = "High", "🟠"
            elif prob >= 0.3:
                tier, color = "Medium", "🟡"
            else:
                tier, color = "Low", "🟢"

            st.markdown(f"### {color} Predicted Risk: **{prob:.4f}** ({tier})")
