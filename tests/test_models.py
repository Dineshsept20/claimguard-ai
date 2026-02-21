"""Tests for ML models.

Validates model loading, prediction ranges, minimum performance
thresholds, and ensemble superiority.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def scored_claims():
    return pd.read_csv(os.path.join(DATA_DIR, "processed", "claims_scored.csv"))


@pytest.fixture(scope="session")
def if_artifact():
    with open(os.path.join(MODELS_DIR, "isolation_forest.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def xgb_artifact():
    with open(os.path.join(MODELS_DIR, "xgboost_model.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def ensemble_config():
    with open(os.path.join(MODELS_DIR, "ensemble_config.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def shap_cache():
    path = os.path.join(MODELS_DIR, "shap_values.pkl")
    if not os.path.exists(path):
        pytest.skip("SHAP values not pre-computed")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Model Artifacts Exist
# ---------------------------------------------------------------------------

class TestModelArtifacts:
    """All model artifacts exist on disk."""

    def test_isolation_forest_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "isolation_forest.pkl"))

    def test_xgboost_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

    def test_ensemble_config_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "ensemble_config.pkl"))

    def test_shap_values_exist(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "shap_values.pkl"))


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    """Models load correctly with expected components."""

    def test_if_has_model_and_scaler(self, if_artifact):
        assert "model" in if_artifact
        assert "scaler" in if_artifact
        assert "features" in if_artifact
        assert len(if_artifact["features"]) > 10

    def test_xgb_has_model_and_scaler(self, xgb_artifact):
        assert "model" in xgb_artifact
        assert "scaler" in xgb_artifact
        assert "features" in xgb_artifact
        assert "feature_importance" in xgb_artifact
        assert len(xgb_artifact["features"]) > 10

    def test_ensemble_config_valid(self, ensemble_config):
        assert "weights" in ensemble_config
        assert "threshold" in ensemble_config
        weights = ensemble_config["weights"]
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"

    def test_shap_cache_valid(self, shap_cache):
        assert "shap_values" in shap_cache
        assert "base_value" in shap_cache
        assert "feature_names" in shap_cache
        assert shap_cache["shap_values"].shape[1] > 10


# ---------------------------------------------------------------------------
# Prediction Range Tests
# ---------------------------------------------------------------------------

class TestPredictionRanges:
    """Model predictions are in valid ranges."""

    def test_if_anomaly_score_range(self, scored_claims):
        scores = scored_claims["if_anomaly_score"]
        assert scores.min() >= 0 - 0.01, f"IF score min={scores.min()}"
        assert scores.max() <= 1 + 0.01, f"IF score max={scores.max()}"

    def test_if_prediction_binary(self, scored_claims):
        vals = set(scored_claims["if_prediction"].unique())
        assert vals.issubset({0, 1})

    def test_xgb_probability_range(self, scored_claims):
        probs = scored_claims["xgb_probability"]
        assert probs.min() >= 0, f"XGB prob min={probs.min()}"
        assert probs.max() <= 1, f"XGB prob max={probs.max()}"

    def test_xgb_prediction_binary(self, scored_claims):
        vals = set(scored_claims["xgb_prediction"].unique())
        assert vals.issubset({0, 1})

    def test_ensemble_score_range(self, scored_claims):
        scores = scored_claims["ensemble_score"]
        assert scores.min() >= 0 - 0.01, f"Ensemble score min={scores.min()}"
        assert scores.max() <= 1 + 0.01, f"Ensemble score max={scores.max()}"

    def test_ensemble_prediction_binary(self, scored_claims):
        vals = set(scored_claims["ensemble_prediction"].unique())
        assert vals.issubset({0, 1})

    def test_risk_tier_values(self, scored_claims):
        valid = {"Low", "Medium", "High", "Critical"}
        actual = set(scored_claims["risk_tier"].dropna().unique())
        assert actual.issubset(valid), f"Invalid risk tiers: {actual - valid}"


# ---------------------------------------------------------------------------
# Minimum Performance Thresholds
# ---------------------------------------------------------------------------

class TestMinimumPerformance:
    """Models meet minimum acceptable performance thresholds."""

    def _metrics(self, y_true, y_pred, y_prob=None):
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        result = {
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }
        if y_prob is not None:
            result["roc_auc"] = roc_auc_score(y_true, y_prob)
        return result

    def test_xgboost_f1_above_threshold(self, scored_claims):
        y_true = scored_claims["is_anomaly"].astype(int)
        y_pred = scored_claims["xgb_prediction"]
        m = self._metrics(y_true, y_pred)
        assert m["f1"] >= 0.60, f"XGBoost F1={m['f1']:.4f} < 0.60 minimum"

    def test_xgboost_recall_above_threshold(self, scored_claims):
        y_true = scored_claims["is_anomaly"].astype(int)
        y_pred = scored_claims["xgb_prediction"]
        m = self._metrics(y_true, y_pred)
        assert m["recall"] >= 0.70, f"XGBoost Recall={m['recall']:.4f} < 0.70 minimum"

    def test_ensemble_f1_above_threshold(self, scored_claims):
        y_true = scored_claims["is_anomaly"].astype(int)
        y_pred = scored_claims["ensemble_prediction"]
        m = self._metrics(y_true, y_pred)
        assert m["f1"] >= 0.65, f"Ensemble F1={m['f1']:.4f} < 0.65 minimum"

    def test_ensemble_roc_auc_above_threshold(self, scored_claims):
        y_true = scored_claims["is_anomaly"].astype(int)
        y_prob = scored_claims["ensemble_score"]
        m = self._metrics(y_true, scored_claims["ensemble_prediction"], y_prob)
        assert m["roc_auc"] >= 0.85, f"Ensemble ROC-AUC={m['roc_auc']:.4f} < 0.85 minimum"


# ---------------------------------------------------------------------------
# Ensemble Beats Individual Models
# ---------------------------------------------------------------------------

class TestEnsembleSuperior:
    """Ensemble model outperforms individual models on F1."""

    def test_ensemble_f1_beats_isolation_forest(self, scored_claims):
        from sklearn.metrics import f1_score
        y_true = scored_claims["is_anomaly"].astype(int)

        if_f1 = f1_score(y_true, scored_claims["if_prediction"])
        ens_f1 = f1_score(y_true, scored_claims["ensemble_prediction"])

        assert ens_f1 > if_f1, \
            f"Ensemble F1={ens_f1:.4f} should beat IF F1={if_f1:.4f}"

    def test_ensemble_f1_beats_rules_only(self, scored_claims):
        from sklearn.metrics import f1_score
        y_true = scored_claims["is_anomaly"].astype(int)

        if "rule_flags_count" in scored_claims.columns:
            rule_pred = (scored_claims["rule_flags_count"] >= 2).astype(int)
            rule_f1 = f1_score(y_true, rule_pred)
            ens_f1 = f1_score(y_true, scored_claims["ensemble_prediction"])
            assert ens_f1 > rule_f1, \
                f"Ensemble F1={ens_f1:.4f} should beat Rules F1={rule_f1:.4f}"


# ---------------------------------------------------------------------------
# SHAP Values Tests
# ---------------------------------------------------------------------------

class TestSHAPValues:
    """SHAP values are valid and consistent."""

    def test_shap_shape_matches_features(self, shap_cache):
        n_features = len(shap_cache["feature_names"])
        assert shap_cache["shap_values"].shape[1] == n_features

    def test_shap_no_nan(self, shap_cache):
        assert not np.isnan(shap_cache["shap_values"]).any(), "SHAP values contain NaN"

    def test_shap_no_inf(self, shap_cache):
        assert not np.isinf(shap_cache["shap_values"]).any(), "SHAP values contain Inf"

    def test_base_value_is_scalar(self, shap_cache):
        bv = shap_cache["base_value"]
        assert isinstance(bv, (int, float, np.floating)), f"base_value is {type(bv)}"
