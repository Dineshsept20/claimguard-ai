"""Tests for the feature engineering pipeline.

Validates feature value ranges, no NaN in critical features,
and deterministic computation.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="session")
def raw_claims():
    return pd.read_csv(
        os.path.join(DATA_DIR, "raw", "claims.csv"),
        parse_dates=["service_date"],
    )


@pytest.fixture(scope="session")
def feature_claims():
    return pd.read_csv(
        os.path.join(DATA_DIR, "processed", "claims_features.csv"),
        parse_dates=["service_date"],
    )


# ---------------------------------------------------------------------------
# Feature Existence
# ---------------------------------------------------------------------------

class TestFeatureColumnsExist:
    """All expected feature columns are present."""

    CLAIM_FEATURES = [
        "cost_vs_awp_ratio", "quantity_vs_typical", "days_supply_qty_mismatch",
        "is_controlled", "is_specialty", "is_weekend", "is_after_hours",
        "cost_percentile", "dispensing_fee_ratio", "opioid_mme_daily",
        "high_refill", "plan_paid_ratio",
    ]

    PRESCRIBER_FEATURES = [
        "presc_total_claims", "presc_unique_members", "presc_unique_pharmacies",
        "presc_avg_cost", "presc_controlled_ratio", "presc_top_pharmacy_pct",
    ]

    PHARMACY_FEATURES = [
        "pharm_total_claims", "pharm_unique_prescribers", "pharm_unique_members",
        "pharm_avg_cost", "pharm_controlled_ratio",
    ]

    NETWORK_FEATURES = [
        "pair_claim_count", "pair_presc_exclusivity", "pair_pharm_exclusivity",
        "member_unique_pharmacies", "member_unique_prescribers",
        "doctor_shopping_signal", "pharmacy_shopping_signal",
    ]

    RULE_FEATURES = ["rule_flags_count"]

    def test_claim_features(self, feature_claims):
        for col in self.CLAIM_FEATURES:
            assert col in feature_claims.columns, f"Missing claim feature: {col}"

    def test_prescriber_features(self, feature_claims):
        for col in self.PRESCRIBER_FEATURES:
            assert col in feature_claims.columns, f"Missing prescriber feature: {col}"

    def test_pharmacy_features(self, feature_claims):
        for col in self.PHARMACY_FEATURES:
            assert col in feature_claims.columns, f"Missing pharmacy feature: {col}"

    def test_network_features(self, feature_claims):
        for col in self.NETWORK_FEATURES:
            assert col in feature_claims.columns, f"Missing network feature: {col}"

    def test_rule_features(self, feature_claims):
        for col in self.RULE_FEATURES:
            assert col in feature_claims.columns, f"Missing rule feature: {col}"


# ---------------------------------------------------------------------------
# Feature Value Ranges
# ---------------------------------------------------------------------------

class TestFeatureRanges:
    """Feature values are in expected ranges."""

    def test_binary_features(self, feature_claims):
        binary_cols = [
            "is_controlled", "is_specialty", "is_weekend",
            "is_after_hours", "high_refill",
        ]
        for col in binary_cols:
            vals = set(feature_claims[col].dropna().unique())
            assert vals.issubset({0, 1, 0.0, 1.0}), f"{col} should be binary, got {vals}"

    def test_ratio_features_non_negative(self, feature_claims):
        ratio_cols = [
            "cost_vs_awp_ratio", "quantity_vs_typical",
            "dispensing_fee_ratio", "plan_paid_ratio",
            "presc_controlled_ratio", "pharm_controlled_ratio",
        ]
        for col in ratio_cols:
            if col in feature_claims.columns:
                min_val = feature_claims[col].min()
                assert min_val >= 0, f"{col} has negative values (min={min_val})"

    def test_cost_percentile_range(self, feature_claims):
        assert feature_claims["cost_percentile"].between(0, 1).all(), \
            "cost_percentile should be in [0, 1]"

    def test_rule_flags_count_non_negative(self, feature_claims):
        assert (feature_claims["rule_flags_count"] >= 0).all(), \
            "rule_flags_count must be ≥ 0"

    def test_opioid_mme_non_negative(self, feature_claims):
        assert (feature_claims["opioid_mme_daily"] >= 0).all(), \
            "opioid_mme_daily must be ≥ 0"

    def test_shopping_signals_non_negative(self, feature_claims):
        for col in ["doctor_shopping_signal", "pharmacy_shopping_signal"]:
            assert (feature_claims[col] >= 0).all(), f"{col} must be ≥ 0"

    def test_exclusivity_range(self, feature_claims):
        for col in ["pair_presc_exclusivity", "pair_pharm_exclusivity"]:
            if col in feature_claims.columns:
                assert feature_claims[col].between(0, 1).all(), \
                    f"{col} should be in [0, 1]"


# ---------------------------------------------------------------------------
# No NaN in Critical Features
# ---------------------------------------------------------------------------

class TestNoNaNInCriticalFeatures:
    """Critical features used by models should not have NaN values."""

    CRITICAL = [
        "cost_vs_awp_ratio", "quantity_vs_typical",
        "is_controlled", "is_weekend", "is_after_hours",
        "cost_percentile", "dispensing_fee_ratio",
        "rule_flags_count", "is_anomaly",
    ]

    def test_no_nan(self, feature_claims):
        for col in self.CRITICAL:
            nan_count = feature_claims[col].isna().sum()
            assert nan_count == 0, f"{col} has {nan_count:,} NaN values"


# ---------------------------------------------------------------------------
# Row Count Consistency
# ---------------------------------------------------------------------------

class TestRowConsistency:
    """Feature engineering preserves row count."""

    def test_same_row_count(self, raw_claims, feature_claims):
        assert len(raw_claims) == len(feature_claims), \
            f"Raw ({len(raw_claims):,}) != features ({len(feature_claims):,})"

    def test_label_preserved(self, raw_claims, feature_claims):
        """is_anomaly column is unchanged after feature engineering."""
        assert (raw_claims["is_anomaly"].values == feature_claims["is_anomaly"].values).all()


# ---------------------------------------------------------------------------
# Deterministic Computation
# ---------------------------------------------------------------------------

class TestDeterministic:
    """Feature engineering produces the same results on the same data."""

    def test_claim_features_deterministic(self):
        from src.features.claim_features import build_claim_features
        from src.data_generator.reference_data import DRUG_REFERENCE

        sample = pd.read_csv(
            os.path.join(DATA_DIR, "raw", "claims.csv"),
            nrows=1000,
            parse_dates=["service_date"],
        )
        result1 = build_claim_features(sample.copy(), DRUG_REFERENCE)
        result2 = build_claim_features(sample.copy(), DRUG_REFERENCE)

        for col in ["cost_vs_awp_ratio", "quantity_vs_typical", "is_controlled"]:
            pd.testing.assert_series_equal(result1[col], result2[col], check_names=False)

    def test_network_features_deterministic(self):
        from src.features.network_features import build_network_features

        sample = pd.read_csv(
            os.path.join(DATA_DIR, "raw", "claims.csv"),
            nrows=1000,
            parse_dates=["service_date"],
        )
        result1 = build_network_features(sample.copy())
        result2 = build_network_features(sample.copy())

        for col in ["pair_claim_count", "doctor_shopping_signal"]:
            pd.testing.assert_series_equal(result1[col], result2[col], check_names=False)
