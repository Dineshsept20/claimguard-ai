"""Pharmacy claims generation engine.

Generates realistic prescription drug claims over a 12-month period,
matching members' chronic conditions to appropriate drug classes,
with realistic pricing, refill patterns, and temporal distributions.

Optimized for fast generation using pre-computed lookups.
"""

import numpy as np
import pandas as pd

from .reference_data import (
    CONDITION_DRUG_CLASSES,
    DRUG_REFERENCE,
    SPECIALTY_DRUG_CLASSES,
    get_diagnosis_for_drug,
)


def generate_claims(
    members_df: pd.DataFrame,
    pharmacies_df: pd.DataFrame,
    prescribers_df: pd.DataFrame,
    num_claims: int = 500_000,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    seed: int = 45,
) -> pd.DataFrame:
    """Generate normal (non-anomalous) pharmacy claims.

    Uses pre-computed lookup tables for fast generation:
    1. Pre-build member→drug options map (one-time)
    2. Pre-build drug_class→prescriber specialty map (one-time)
    3. Pre-compute pharmacy ID arrays by type (one-time)
    4. Generate dates and hours via vectorized numpy ops
    5. Loop claims with cheap dict/array lookups only

    Args:
        members_df: Members DataFrame from generate_members().
        pharmacies_df: Pharmacies DataFrame from generate_pharmacies().
        prescribers_df: Prescribers DataFrame from generate_prescribers().
        num_claims: Target number of claims to generate.
        start_date: Start of the claims period.
        end_date: End of the claims period.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with all claim fields. No anomalies injected yet.
    """
    rng = np.random.default_rng(seed)
    print(f"  Generating {num_claims:,} claims (optimized)...")

    # ===== ONE-TIME PRE-COMPUTATION =====

    # 1. Drug lookup by therapeutic class
    drug_by_class = {}
    for drug in DRUG_REFERENCE:
        drug_by_class.setdefault(drug["therapeutic_class"], []).append(drug)

    # 2. Member → list of eligible drugs (pre-computed once)
    member_ids = members_df["member_id"].values
    member_drug_options = {}
    for _, row in members_df.iterrows():
        conds = row["chronic_conditions"].split(",") if row["chronic_conditions"] else []
        drugs = []
        for cond in conds:
            for cls in CONDITION_DRUG_CLASSES.get(cond, []):
                drugs.extend(drug_by_class.get(cls, []))
        if not drugs:
            drugs = drug_by_class.get("STATIN", DRUG_REFERENCE[:1])
        member_drug_options[row["member_id"]] = drugs

    # 3. drug_class → list of matching prescriber IDs (pre-computed once)
    prescriber_ids_all = prescribers_df["prescriber_id"].values
    prescriber_by_spec = {}
    for _, row in prescribers_df.iterrows():
        prescriber_by_spec.setdefault(row["specialty"], []).append(row["prescriber_id"])

    # Pre-build: drug_class → array of matching prescriber IDs
    class_prescriber_map = {}
    for drug_class_name in drug_by_class:
        matching = []
        for spec, classes in SPECIALTY_DRUG_CLASSES.items():
            if classes is None or drug_class_name in classes:
                matching.extend(prescriber_by_spec.get(spec, []))
        class_prescriber_map[drug_class_name] = np.array(matching) if matching else None

    # 4. Pharmacy ID arrays by type (pre-computed once, no per-call masking)
    pharm_ids = pharmacies_df["pharmacy_id"].values
    pharm_types = pharmacies_df["pharmacy_type"].values
    retail_ids = pharm_ids[pharm_types == "retail"]
    specialty_pharm_ids = pharm_ids[pharm_types == "specialty"]
    compounding_ids = pharm_ids[pharm_types == "compounding"]
    mail_ids = pharm_ids[pharm_types == "mail_order"]

    # 5. Member weights (more conditions → more claims)
    condition_counts = members_df["num_conditions"].values.astype(float)
    member_weights = condition_counts / condition_counts.sum()

    # ===== VECTORIZED DATE & HOUR GENERATION =====
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    total_days = (end_ts - start_ts).days

    day_offsets = rng.integers(0, total_days, size=num_claims)
    service_dates = start_ts + pd.to_timedelta(day_offsets, unit="D")

    # Shift ~60% of weekend dates to nearest Monday
    weekend_mask = service_dates.weekday >= 5
    shift_mask = weekend_mask & (rng.random(num_claims) < 0.60)
    if shift_mask.any():
        dates_array = service_dates.values.copy()
        wd = service_dates[shift_mask].weekday
        shift_days = np.where(wd == 5, 2, 1)  # Sat→Mon(+2), Sun→Mon(+1)
        dates_array[shift_mask] = dates_array[shift_mask] + pd.to_timedelta(shift_days, unit="D").values
        service_dates = pd.DatetimeIndex(dates_array)

    # Submit hours: 80% business hours (8-18), 20% off-hours
    submit_hours = np.zeros(num_claims, dtype=int)
    biz_mask = rng.random(num_claims) < 0.80
    submit_hours[biz_mask] = np.clip(
        rng.normal(13, 2.5, size=biz_mask.sum()).astype(int), 8, 18
    )
    off_choices = np.array(list(range(0, 8)) + list(range(19, 24)))
    submit_hours[~biz_mask] = rng.choice(off_choices, size=(~biz_mask).sum())

    # ===== MAIN CLAIM LOOP (optimized: cheap lookups only) =====
    print("  Building claim records...")
    records = []

    # Pre-select all members at once
    member_indices = rng.choice(len(member_ids), size=num_claims, p=member_weights)
    claim_member_ids = member_ids[member_indices]

    for i in range(num_claims):
        mid = claim_member_ids[i]
        options = member_drug_options[mid]
        drug = options[rng.integers(0, len(options))]
        tc = drug["therapeutic_class"]

        # Quantity with small variation
        quantity = max(1, int(drug["typical_quantity"] * rng.uniform(0.9, 1.1)))
        days_supply = drug["typical_days_supply"]

        # Refill number
        refill_number = 0 if tc == "ANTIBIOTIC" else int(rng.integers(0, 12))

        # Pricing (inlined for speed — avoids function call overhead)
        awp = drug["awp"]
        discount = rng.uniform(0.80, 0.95)
        ingredient_cost = round(awp * quantity * discount, 2)
        dispensing_fee = round(
            rng.uniform(2.0, 6.0) if drug["is_brand"] else rng.uniform(1.0, 4.0), 2
        )
        total_cost = round(ingredient_cost + dispensing_fee, 2)

        if tc in ("BIOLOGIC_IMMUNOLOGY", "ONCOLOGY", "HEPATITIS_C"):
            copay = round(min(rng.uniform(50, 200), total_cost * 0.20), 2)
        elif drug["is_brand"]:
            copay = round(min(rng.uniform(25, 75), total_cost * 0.30), 2)
        else:
            copay = round(min(rng.uniform(5, 25), total_cost * 0.50), 2)
        plan_paid = round(max(total_cost - copay, 0), 2)

        # Diagnosis
        icd_code, icd_desc = get_diagnosis_for_drug(tc, rng)

        # Prescriber (use pre-computed class→prescriber map)
        class_candidates = class_prescriber_map.get(tc)
        if class_candidates is not None and len(class_candidates) > 0 and rng.random() < 0.70:
            prescriber_id = class_candidates[rng.integers(0, len(class_candidates))]
        else:
            prescriber_id = prescriber_ids_all[rng.integers(0, len(prescriber_ids_all))]

        # Pharmacy (use pre-computed arrays)
        if tc in ("BIOLOGIC_IMMUNOLOGY", "ONCOLOGY", "HEPATITIS_C") and len(specialty_pharm_ids) > 0 and rng.random() < 0.60:
            pharmacy_id = specialty_pharm_ids[rng.integers(0, len(specialty_pharm_ids))]
        elif tc == "COMPOUNDING" and len(compounding_ids) > 0 and rng.random() < 0.70:
            pharmacy_id = compounding_ids[rng.integers(0, len(compounding_ids))]
        elif len(retail_ids) > 0 and rng.random() < 0.80:
            pharmacy_id = retail_ids[rng.integers(0, len(retail_ids))]
        else:
            pharmacy_id = pharm_ids[rng.integers(0, len(pharm_ids))]

        # Prior auth
        prior_auth = tc in ("BIOLOGIC_IMMUNOLOGY", "ONCOLOGY", "HEPATITIS_C") or (
            drug["is_brand"] and rng.random() < 0.15
        )

        # Claim status
        status_roll = rng.random()
        claim_status = "paid" if status_roll < 0.92 else ("rejected" if status_roll < 0.97 else "reversed")

        records.append({
            "claim_id": f"CLM{i+1:08d}",
            "service_date": service_dates[i],
            "submit_hour": int(submit_hours[i]),
            "member_id": mid,
            "prescriber_id": prescriber_id,
            "pharmacy_id": pharmacy_id,
            "ndc": drug["ndc"],
            "drug_name": drug["drug_name"],
            "generic_name": drug["generic_name"],
            "therapeutic_class": tc,
            "dea_schedule": drug["dea_schedule"],
            "quantity": quantity,
            "days_supply": days_supply,
            "refill_number": refill_number,
            "ingredient_cost": ingredient_cost,
            "dispensing_fee": dispensing_fee,
            "copay": copay,
            "plan_paid": plan_paid,
            "total_cost": total_cost,
            "diagnosis_code": icd_code,
            "diagnosis_desc": icd_desc,
            "prior_auth_flag": prior_auth,
            "claim_status": claim_status,
            "is_brand": drug["is_brand"],
            "mme_factor": drug["mme_factor"],
            "is_anomaly": False,
            "anomaly_type": "none",
        })

        if (i + 1) % 50_000 == 0:
            print(f"    {i+1:,} / {num_claims:,} claims done...")

    print(f"  All {num_claims:,} claims generated.")
    return pd.DataFrame(records)
