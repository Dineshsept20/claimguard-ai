"""Shared data loading utilities for the Streamlit dashboard.

Uses st.cache_data to load data once and share across pages.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_data(ttl=3600)
def load_scored_claims() -> pd.DataFrame:
    """Load the scored claims dataset."""
    path = os.path.join(DATA_DIR, "processed", "claims_scored.csv")
    df = pd.read_csv(path, parse_dates=["service_date"])
    return df


@st.cache_data(ttl=3600)
def load_feature_claims() -> pd.DataFrame:
    """Load the full feature-engineered claims dataset."""
    path = os.path.join(DATA_DIR, "processed", "claims_features.csv")
    df = pd.read_csv(path, parse_dates=["service_date"])
    return df


@st.cache_data(ttl=3600)
def load_prescribers() -> pd.DataFrame:
    """Load prescribers reference data."""
    path = os.path.join(DATA_DIR, "raw", "prescribers.csv")
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_pharmacies() -> pd.DataFrame:
    """Load pharmacies reference data."""
    path = os.path.join(DATA_DIR, "raw", "pharmacies.csv")
    return pd.read_csv(path)


@st.cache_resource
def load_xgb_model():
    """Load the XGBoost model artifact."""
    path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact


@st.cache_resource
def load_shap_cache():
    """Load cached SHAP values."""
    path = os.path.join(MODELS_DIR, "shap_values.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def format_currency(value: float) -> str:
    """Format number as currency."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def format_number(value: float) -> str:
    """Format number with comma separators."""
    return f"{value:,.0f}"
