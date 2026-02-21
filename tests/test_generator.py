"""Tests for the synthetic data generator.

Validates data schema, anomaly rates, FK integrity,
and no impossible data combinations.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ---------------------------------------------------------------------------
# Fixtures — load data once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def claims():
    return pd.read_csv(os.path.join(DATA_DIR, "claims.csv"), parse_dates=["service_date"])


@pytest.fixture(scope="session")
def pharmacies():
    return pd.read_csv(os.path.join(DATA_DIR, "pharmacies.csv"))


@pytest.fixture(scope="session")
def prescribers():
    return pd.read_csv(os.path.join(DATA_DIR, "prescribers.csv"))


@pytest.fixture(scope="session")
def members():
    return pd.read_csv(os.path.join(DATA_DIR, "members.csv"))


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestClaimsSchema:
    """Validate the claims CSV has the correct columns and types."""

    EXPECTED_COLUMNS = [
        "claim_id", "service_date", "submit_hour", "member_id",
        "prescriber_id", "pharmacy_id", "ndc", "drug_name",
        "generic_name", "therapeutic_class", "dea_schedule",
        "quantity", "days_supply", "refill_number", "ingredient_cost",
        "dispensing_fee", "copay", "plan_paid", "total_cost",
        "diagnosis_code", "diagnosis_desc", "prior_auth_flag",
        "claim_status", "is_brand", "mme_factor", "is_anomaly",
        "anomaly_type",
    ]

    def test_all_columns_present(self, claims):
        for col in self.EXPECTED_COLUMNS:
            assert col in claims.columns, f"Missing column: {col}"

    def test_no_extra_columns(self, claims):
        extra = set(claims.columns) - set(self.EXPECTED_COLUMNS)
        assert len(extra) == 0, f"Unexpected columns: {extra}"

    def test_claim_id_unique(self, claims):
        assert claims["claim_id"].is_unique, "claim_id must be unique"

    def test_row_count(self, claims):
        assert len(claims) >= 100_000, f"Expected ≥100K claims, got {len(claims):,}"

    def test_no_null_ids(self, claims):
        for col in ["claim_id", "member_id", "prescriber_id", "pharmacy_id", "ndc"]:
            assert claims[col].notna().all(), f"{col} has null values"

    def test_numeric_types(self, claims):
        numeric_cols = ["quantity", "days_supply", "ingredient_cost",
                        "dispensing_fee", "copay", "plan_paid", "total_cost"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(claims[col]), f"{col} should be numeric"


class TestPharmaciesSchema:
    EXPECTED_COLUMNS = [
        "pharmacy_id", "pharmacy_name", "npi", "pharmacy_type",
        "address", "city", "state", "zip_code", "chain_flag",
        "chain_name", "is_suspicious",
    ]

    def test_all_columns_present(self, pharmacies):
        for col in self.EXPECTED_COLUMNS:
            assert col in pharmacies.columns, f"Missing column: {col}"

    def test_pharmacy_id_unique(self, pharmacies):
        assert pharmacies["pharmacy_id"].is_unique


class TestPrescribersSchema:
    EXPECTED_COLUMNS = [
        "prescriber_id", "first_name", "last_name", "full_name",
        "npi", "dea_number", "specialty", "state",
        "practice_type", "is_suspicious",
    ]

    def test_all_columns_present(self, prescribers):
        for col in self.EXPECTED_COLUMNS:
            assert col in prescribers.columns, f"Missing column: {col}"

    def test_prescriber_id_unique(self, prescribers):
        assert prescribers["prescriber_id"].is_unique


class TestMembersSchema:
    EXPECTED_COLUMNS = [
        "member_id", "first_name", "last_name", "age", "gender",
        "state", "plan_type", "chronic_conditions", "num_conditions",
        "is_doctor_shopper",
    ]

    def test_all_columns_present(self, members):
        for col in self.EXPECTED_COLUMNS:
            assert col in members.columns, f"Missing column: {col}"

    def test_member_id_unique(self, members):
        assert members["member_id"].is_unique


# ---------------------------------------------------------------------------
# Data Quality Tests
# ---------------------------------------------------------------------------

class TestAnomalyRate:
    """Validate anomaly injection is within expected bounds."""

    def test_anomaly_rate_range(self, claims):
        rate = claims["is_anomaly"].mean()
        assert 0.02 <= rate <= 0.08, f"Anomaly rate {rate:.2%} outside [2%, 8%]"

    def test_anomaly_types_present(self, claims):
        expected_types = {
            "quantity_manipulation", "prescriber_pharmacy_collusion",
            "doctor_shopping", "therapeutic_duplication",
            "phantom_billing", "upcoding", "refill_too_soon",
        }
        actual_types = set(claims[claims["is_anomaly"] == 1]["anomaly_type"].dropna().unique())
        missing = expected_types - actual_types
        assert len(missing) == 0, f"Missing anomaly types: {missing}"

    def test_normal_claims_no_anomaly_type(self, claims):
        normal = claims[claims["is_anomaly"] == 0]
        # Normal claims should have anomaly_type as 'none' or NaN
        has_real_type = normal["anomaly_type"].apply(
            lambda x: pd.notna(x) and str(x).lower() != "none"
        ).sum()
        assert has_real_type == 0, f"{has_real_type} normal claims have a real anomaly_type set"


class TestDataRanges:
    """Validate data values are in realistic ranges."""

    def test_quantity_positive(self, claims):
        assert (claims["quantity"] > 0).all(), "All quantities must be positive"

    def test_days_supply_positive(self, claims):
        assert (claims["days_supply"] > 0).all(), "All days_supply must be positive"

    def test_costs_non_negative(self, claims):
        for col in ["ingredient_cost", "dispensing_fee", "copay", "plan_paid", "total_cost"]:
            assert (claims[col] >= 0).all(), f"{col} has negative values"

    def test_submit_hour_range(self, claims):
        assert claims["submit_hour"].between(0, 23).all(), "submit_hour must be 0-23"

    def test_refill_number_non_negative(self, claims):
        assert (claims["refill_number"] >= 0).all(), "refill_number must be ≥0"

    def test_member_age_range(self, members):
        assert members["age"].between(0, 120).all(), "age must be 0-120"

    def test_total_cost_equals_sum(self, claims):
        """total_cost ≈ ingredient_cost + dispensing_fee (within rounding)."""
        expected = claims["ingredient_cost"] + claims["dispensing_fee"]
        diff = (claims["total_cost"] - expected).abs()
        assert (diff < 0.02).all(), "total_cost should equal ingredient_cost + dispensing_fee"


# ---------------------------------------------------------------------------
# Foreign Key Integrity Tests
# ---------------------------------------------------------------------------

class TestForeignKeys:
    """Every FK in claims references a valid entity."""

    def test_member_ids_valid(self, claims, members):
        invalid = set(claims["member_id"]) - set(members["member_id"])
        assert len(invalid) == 0, f"{len(invalid)} invalid member_ids: {list(invalid)[:5]}"

    def test_prescriber_ids_valid(self, claims, prescribers):
        invalid = set(claims["prescriber_id"]) - set(prescribers["prescriber_id"])
        assert len(invalid) == 0, f"{len(invalid)} invalid prescriber_ids: {list(invalid)[:5]}"

    def test_pharmacy_ids_valid(self, claims, pharmacies):
        invalid = set(claims["pharmacy_id"]) - set(pharmacies["pharmacy_id"])
        assert len(invalid) == 0, f"{len(invalid)} invalid pharmacy_ids: {list(invalid)[:5]}"


# ---------------------------------------------------------------------------
# No Impossible Combinations
# ---------------------------------------------------------------------------

class TestDataConsistency:
    """No impossible data combinations."""

    def test_dea_schedule_values(self, claims):
        valid = {"II", "III", "IV", "V", "NONE"}
        actual = set(claims["dea_schedule"].unique())
        invalid = actual - valid
        assert len(invalid) == 0, f"Invalid DEA schedules: {invalid}"

    def test_claim_status_values(self, claims):
        valid = {"paid", "rejected", "reversed"}
        actual = set(claims["claim_status"].str.lower().unique())
        invalid = actual - valid
        assert len(invalid) == 0, f"Invalid claim statuses: {invalid}"

    def test_plan_type_values(self, members):
        valid = {"commercial", "medicare", "medicaid", "exchange", "fep"}
        actual = set(members["plan_type"].str.lower().unique())
        invalid = actual - valid
        assert len(invalid) == 0, f"Invalid plan types: {invalid}"

    def test_pharmacy_type_values(self, pharmacies):
        valid = {"retail", "mail_order", "specialty", "compounding"}
        actual = set(pharmacies["pharmacy_type"].str.lower().unique())
        invalid = actual - valid
        assert len(invalid) == 0, f"Invalid pharmacy types: {invalid}"

    def test_brand_flag_binary(self, claims):
        assert set(claims["is_brand"].unique()).issubset({0, 1, True, False})

    def test_anomaly_flag_binary(self, claims):
        assert set(claims["is_anomaly"].unique()).issubset({0, 1, True, False})
