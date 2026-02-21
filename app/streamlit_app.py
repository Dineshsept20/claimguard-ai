"""ClaimGuard AI — Streamlit Dashboard.

Main entry point for the multi-page Streamlit application.
Launch with:  streamlit run app/streamlit_app.py
"""

import os
import sys

import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    st.set_page_config(
        page_title="ClaimGuard AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    [data-testid="stSidebar"] {background-color: #1E3A5F;}
    [data-testid="stSidebar"] .css-1d391kg {color: white;}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {color: white !important;}
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F, #4A90D9);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .metric-card h3 {margin: 0; font-size: 0.9rem; opacity: 0.85;}
    .metric-card h1 {margin: 0; font-size: 2rem;}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("# 🛡️ ClaimGuard AI")
        st.markdown("*AI-Powered Pharmacy Claims Anomaly Detection*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "📊 Executive Overview",
                "🔍 Claims Explorer",
                "👨‍⚕️ Prescriber Profile",
                "📈 Model Performance",
                "🧠 Explainability",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("##### About")
        st.markdown(
            "ClaimGuard AI uses a hybrid ensemble of "
            "Isolation Forest, XGBoost, and domain rules "
            "to detect pharmacy claims fraud, waste, and abuse."
        )

    # Route to page
    if "Overview" in page:
        from app.pages.p01_overview import render
        render()
    elif "Claims Explorer" in page:
        from app.pages.p02_claims_explorer import render
        render()
    elif "Prescriber" in page:
        from app.pages.p03_prescriber_profile import render
        render()
    elif "Model Performance" in page:
        from app.pages.p04_model_performance import render
        render()
    elif "Explainability" in page:
        from app.pages.p05_explainability import render
        render()


if __name__ == "__main__":
    main()
