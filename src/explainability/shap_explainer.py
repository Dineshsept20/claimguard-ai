"""SHAP explainability module for ClaimGuard AI.

Provides global feature importance, per-claim local explanations,
natural language explanation generation, and dependence plot data.
"""

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 5000,
    save_path: str | None = "models/shap_values.pkl",
) -> shap.Explanation:
    """Compute SHAP values for an XGBoost model.

    Uses TreeExplainer for fast, exact SHAP values on tree models.

    Args:
        model: Trained XGBoost model.
        X: Scaled feature matrix.
        feature_names: List of feature column names.
        max_samples: Max samples to explain (for speed).
        save_path: Path to cache SHAP values.

    Returns:
        shap.Explanation object.
    """
    # Subsample for speed
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
        idx = np.arange(len(X))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample,
        feature_names=feature_names,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        artifact = {
            "shap_values": shap_values,
            "base_value": explainer.expected_value,
            "X_sample": X_sample,
            "feature_names": feature_names,
            "sample_indices": idx,
        }
        with open(save_path, "wb") as f:
            pickle.dump(artifact, f)

    return explanation


def get_global_importance(explanation: shap.Explanation) -> pd.DataFrame:
    """Get global feature importance ranked by mean |SHAP|.

    Returns:
        DataFrame with columns: feature, mean_abs_shap, rank.
    """
    vals = np.abs(explanation.values)
    mean_abs = vals.mean(axis=0)

    importance = pd.DataFrame({
        "feature": explanation.feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance["rank"] = range(1, len(importance) + 1)
    return importance


def get_claim_explanation(
    explanation: shap.Explanation,
    claim_index: int,
) -> pd.DataFrame:
    """Get per-feature SHAP contributions for a single claim.

    Args:
        explanation: SHAP Explanation object.
        claim_index: Index within the explanation (0-based).

    Returns:
        DataFrame with columns: feature, shap_value, feature_value, direction.
    """
    sv = explanation.values[claim_index]
    fv = explanation.data[claim_index]
    names = explanation.feature_names

    result = pd.DataFrame({
        "feature": names,
        "shap_value": sv,
        "feature_value": fv,
        "abs_shap": np.abs(sv),
    }).sort_values("abs_shap", ascending=False)

    result["direction"] = np.where(result["shap_value"] > 0, "↑ increases risk", "↓ decreases risk")
    return result.drop(columns=["abs_shap"]).reset_index(drop=True)


# Feature name → human-readable label mapping
_FEATURE_LABELS = {
    "cost_vs_awp_ratio": "ingredient cost relative to AWP",
    "quantity_vs_typical": "quantity relative to typical",
    "days_supply_qty_mismatch": "days supply / quantity mismatch",
    "is_controlled": "controlled substance",
    "is_specialty": "specialty drug",
    "is_weekend": "weekend fill",
    "is_after_hours": "after-hours submission",
    "cost_percentile": "cost percentile within drug class",
    "dispensing_fee_ratio": "dispensing fee as fraction of total cost",
    "opioid_mme_daily": "daily MME (opioid strength)",
    "high_refill": "high refill number (≥6)",
    "plan_paid_ratio": "plan-paid ratio",
    "pair_presc_exclusivity": "prescriber–pharmacy pair exclusivity",
    "pair_pharm_exclusivity": "pharmacy–prescriber pair exclusivity",
    "pair_log_volume": "prescriber–pharmacy pair volume",
    "member_class_unique_prescribers": "unique prescribers for drug class",
    "member_unique_pharmacies": "unique pharmacies used by member",
    "member_unique_prescribers": "unique prescribers used by member",
    "member_controlled_ratio": "member controlled-substance ratio",
    "doctor_shopping_signal": "doctor shopping signal",
    "pharmacy_shopping_signal": "pharmacy shopping signal",
    "presc_controlled_ratio": "prescriber controlled-substance ratio",
    "presc_top_pharmacy_pct": "prescriber top-pharmacy concentration",
    "presc_avg_mme_per_patient": "prescriber average MME per patient",
    "presc_after_hours_ratio": "prescriber after-hours ratio",
    "presc_cost_peer_zscore": "prescriber cost vs peers (z-score)",
    "presc_claims_per_member": "prescriber claims per member",
    "pharm_controlled_ratio": "pharmacy controlled-substance ratio",
    "pharm_reversal_rate": "pharmacy reversal rate",
    "pharm_brand_ratio": "pharmacy brand-name ratio",
    "pharm_after_hours_ratio": "pharmacy after-hours ratio",
    "pharm_top_prescriber_pct": "pharmacy top-prescriber concentration",
    "pharm_cost_peer_zscore": "pharmacy cost vs peers (z-score)",
    "pharm_volume_peer_zscore": "pharmacy volume vs peers (z-score)",
    "pharm_claims_per_member": "pharmacy claims per member",
    "rule_flags_count": "number of business rule flags",
}


def generate_natural_language_explanation(
    explanation: shap.Explanation,
    claim_index: int,
    claim_row: pd.Series | None = None,
    top_n: int = 5,
) -> str:
    """Generate a human-readable explanation for why a claim was flagged.

    Args:
        explanation: SHAP Explanation object.
        claim_index: Index within the explanation.
        claim_row: Optional original claim row for context.
        top_n: Number of top contributing features to mention.

    Returns:
        Multi-line natural language explanation string.
    """
    detail = get_claim_explanation(explanation, claim_index)
    top = detail.head(top_n)

    lines = []

    # Header with claim context if available
    if claim_row is not None:
        cid = claim_row.get("claim_id", "Unknown")
        drug = claim_row.get("drug_name", "Unknown")
        cost = claim_row.get("total_cost", 0)
        lines.append(f"**Claim {cid}** — {drug} (${cost:,.2f})")
    else:
        lines.append("**Flagged Claim Analysis**")

    lines.append("")
    lines.append("This claim was flagged because:")

    for _, row in top.iterrows():
        feat = row["feature"]
        sv = row["shap_value"]
        fv = row["feature_value"]
        label = _FEATURE_LABELS.get(feat, feat.replace("_", " "))

        direction = "increases" if sv > 0 else "decreases"
        strength = abs(sv)

        if strength > 1.0:
            qualifier = "strongly"
        elif strength > 0.3:
            qualifier = "moderately"
        else:
            qualifier = "slightly"

        # Format value contextually
        if "ratio" in feat or "pct" in feat or "percentile" in feat:
            val_str = f"{fv:.1%}" if fv <= 1.5 else f"{fv:.2f}x"
        elif "zscore" in feat:
            val_str = f"{fv:+.1f}σ"
        elif "mme" in feat.lower():
            val_str = f"{fv:.0f} MME"
        elif feat in ("is_controlled", "is_specialty", "is_weekend", "is_after_hours", "high_refill"):
            val_str = "Yes" if fv else "No"
        elif "count" in feat or "signal" in feat:
            val_str = f"{fv:.0f}"
        else:
            val_str = f"{fv:.2f}"

        lines.append(f"  • The **{label}** ({val_str}) {qualifier} {direction} the risk score")

    # Base risk
    base_val = explanation.base_values
    if isinstance(base_val, np.ndarray):
        base_val = base_val[0]
    lines.append(f"\n_Base risk score: {base_val:.3f}_")

    return "\n".join(lines)


def get_dependence_data(
    explanation: shap.Explanation,
    feature: str,
    interaction_feature: str | None = None,
) -> pd.DataFrame:
    """Get data for a SHAP dependence plot.

    Args:
        explanation: SHAP Explanation object.
        feature: Primary feature name.
        interaction_feature: Optional interaction feature for coloring.

    Returns:
        DataFrame with feature values and SHAP values.
    """
    feat_idx = list(explanation.feature_names).index(feature)

    result = pd.DataFrame({
        "feature_value": explanation.data[:, feat_idx],
        "shap_value": explanation.values[:, feat_idx],
    })

    if interaction_feature and interaction_feature in explanation.feature_names:
        int_idx = list(explanation.feature_names).index(interaction_feature)
        result["interaction_value"] = explanation.data[:, int_idx]

    return result


def load_shap_values(path: str = "models/shap_values.pkl") -> dict[str, Any]:
    """Load cached SHAP values."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_explanation_from_cache(cache: dict) -> shap.Explanation:
    """Reconstruct a SHAP Explanation from cached data."""
    return shap.Explanation(
        values=cache["shap_values"],
        base_values=cache["base_value"],
        data=cache["X_sample"],
        feature_names=cache["feature_names"],
    )
