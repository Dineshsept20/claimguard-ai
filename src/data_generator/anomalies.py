"""Anomaly injection engine for pharmacy claims.

Injects 7 realistic fraud/waste/abuse patterns into a baseline claims
DataFrame. Each anomaly type mirrors real-world patterns seen in PBM
claims adjudication systems (RxClaim, FEP, EDF).

Anomaly Distribution (of total anomalies):
  1. Quantity Manipulation        — 15%
  2. Prescriber-Pharmacy Collusion — 20%
  3. Doctor Shopping              — 15%
  4. Therapeutic Duplication      — 10%
  5. Phantom Billing              — 15%
  6. Upcoding / Price Manipulation — 15%
  7. Refill Too Soon              — 10%
"""

import numpy as np
import pandas as pd

from .reference_data import (
    DRUG_REFERENCE,
    get_brand_generic_pairs,
    get_diagnosis_for_drug,
)

# Anomaly type distribution (must sum to 1.0)
ANOMALY_DISTRIBUTION = {
    "quantity_manipulation": 0.15,
    "prescriber_pharmacy_collusion": 0.20,
    "doctor_shopping": 0.15,
    "therapeutic_duplication": 0.10,
    "phantom_billing": 0.15,
    "upcoding": 0.15,
    "refill_too_soon": 0.10,
}


def inject_anomalies(
    claims_df: pd.DataFrame,
    members_df: pd.DataFrame,
    pharmacies_df: pd.DataFrame,
    prescribers_df: pd.DataFrame,
    anomaly_rate: float = 0.04,
    seed: int = 99,
) -> pd.DataFrame:
    """Inject fraud/waste/abuse anomalies into a clean claims DataFrame.

    Args:
        claims_df: Clean claims from generate_claims() (is_anomaly=False).
        members_df: Members DataFrame.
        pharmacies_df: Pharmacies DataFrame.
        prescribers_df: Prescribers DataFrame.
        anomaly_rate: Target fraction of claims to be anomalous (0.03–0.05).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with anomalies injected. Modified claims have
        is_anomaly=True and anomaly_type set to the pattern name.
    """
    rng = np.random.default_rng(seed)
    df = claims_df.copy()
    n_total = len(df)
    n_anomalies = int(n_total * anomaly_rate)

    print(f"  Injecting {n_anomalies:,} anomalies ({anomaly_rate*100:.1f}% of {n_total:,} claims)...")

    # Allocate anomaly counts per type
    anomaly_counts = {}
    remaining = n_anomalies
    types_list = list(ANOMALY_DISTRIBUTION.keys())
    for atype in types_list[:-1]:
        count = int(n_anomalies * ANOMALY_DISTRIBUTION[atype])
        anomaly_counts[atype] = count
        remaining -= count
    anomaly_counts[types_list[-1]] = remaining  # last type gets the remainder

    # Pre-compute useful lookups
    suspicious_pharmacies = pharmacies_df[pharmacies_df["is_suspicious"]]["pharmacy_id"].values
    suspicious_prescribers = prescribers_df[prescribers_df["is_suspicious"]]["prescriber_id"].values
    doctor_shoppers = members_df[members_df["is_doctor_shopper"]]["member_id"].values

    # Drug lookup by class
    drug_by_class = {}
    for drug in DRUG_REFERENCE:
        drug_by_class.setdefault(drug["therapeutic_class"], []).append(drug)

    # Controlled substance drug classes
    controlled_classes = ["OPIOID_ANALGESIC", "BENZODIAZEPINE", "STIMULANT"]

    # Track which indices have been used for anomalies
    used_indices = set()

    # ---- 1. Quantity Manipulation (15%) ----
    n_qty = anomaly_counts["quantity_manipulation"]
    qty_candidates = df.index[~df.index.isin(used_indices)].to_numpy()
    qty_indices = rng.choice(qty_candidates, size=min(n_qty, len(qty_candidates)), replace=False)

    for idx in qty_indices:
        row = df.loc[idx]
        # Find typical quantity for this drug
        matching = [d for d in DRUG_REFERENCE if d["ndc"] == row["ndc"]]
        if matching:
            typical_qty = matching[0]["typical_quantity"]
            # Inflate quantity 3x-5x typical
            inflated_qty = int(typical_qty * rng.uniform(3.0, 5.0))
            df.at[idx, "quantity"] = inflated_qty
            # Days supply may not change (mismatch signal)
            if rng.random() < 0.6:
                pass  # Keep original days_supply → creates mismatch
            else:
                df.at[idx, "days_supply"] = int(row["days_supply"] * rng.uniform(2.0, 3.0))
            # Recalculate cost based on inflated quantity
            awp = matching[0]["awp"]
            df.at[idx, "ingredient_cost"] = round(awp * inflated_qty * rng.uniform(0.85, 0.98), 2)
            df.at[idx, "total_cost"] = round(df.at[idx, "ingredient_cost"] + df.at[idx, "dispensing_fee"], 2)
            df.at[idx, "plan_paid"] = round(max(df.at[idx, "total_cost"] - df.at[idx, "copay"], 0), 2)
        # Prefer suspicious pharmacies (60% chance)
        if len(suspicious_pharmacies) > 0 and rng.random() < 0.6:
            df.at[idx, "pharmacy_id"] = rng.choice(suspicious_pharmacies)
        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = "quantity_manipulation"

    used_indices.update(qty_indices)
    print(f"    1. Quantity Manipulation: {len(qty_indices):,} claims")

    # ---- 2. Prescriber-Pharmacy Collusion (20%) ----
    n_collusion = anomaly_counts["prescriber_pharmacy_collusion"]
    # Pick collusion pairs from suspicious entities
    n_pairs = max(3, min(10, len(suspicious_prescribers), len(suspicious_pharmacies)))
    if n_pairs > 0 and len(suspicious_prescribers) > 0 and len(suspicious_pharmacies) > 0:
        collusion_prescribers = rng.choice(suspicious_prescribers, size=n_pairs, replace=True)
        collusion_pharmacies = rng.choice(suspicious_pharmacies, size=n_pairs, replace=True)
    else:
        # Fallback: pick random entities
        collusion_prescribers = rng.choice(prescribers_df["prescriber_id"].values, size=n_pairs, replace=False)
        collusion_pharmacies = rng.choice(pharmacies_df["pharmacy_id"].values, size=n_pairs, replace=False)

    collusion_candidates = df.index[~df.index.isin(used_indices)].to_numpy()
    collusion_indices = rng.choice(collusion_candidates, size=min(n_collusion, len(collusion_candidates)), replace=False)

    for i, idx in enumerate(collusion_indices):
        pair_idx = i % n_pairs
        df.at[idx, "prescriber_id"] = collusion_prescribers[pair_idx]
        df.at[idx, "pharmacy_id"] = collusion_pharmacies[pair_idx]
        # Collusion claims tend to be controlled substances
        if rng.random() < 0.7:
            ctrl_class = rng.choice(controlled_classes)
            ctrl_drugs = drug_by_class.get(ctrl_class, [])
            if ctrl_drugs:
                drug = ctrl_drugs[rng.integers(0, len(ctrl_drugs))]
                df.at[idx, "ndc"] = drug["ndc"]
                df.at[idx, "drug_name"] = drug["drug_name"]
                df.at[idx, "generic_name"] = drug["generic_name"]
                df.at[idx, "therapeutic_class"] = drug["therapeutic_class"]
                df.at[idx, "dea_schedule"] = drug["dea_schedule"]
                df.at[idx, "mme_factor"] = drug["mme_factor"]
                df.at[idx, "is_brand"] = drug["is_brand"]
        # After-hours submissions more common in collusion
        if rng.random() < 0.5:
            df.at[idx, "submit_hour"] = int(rng.choice([0, 1, 2, 3, 4, 5, 21, 22, 23]))
        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = "prescriber_pharmacy_collusion"

    used_indices.update(collusion_indices)
    print(f"    2. Prescriber-Pharmacy Collusion: {len(collusion_indices):,} claims")

    # ---- 3. Doctor Shopping (15%) ----
    n_shopping = anomaly_counts["doctor_shopping"]
    # Use known doctor shoppers for most, random members for some
    shopping_candidates = df.index[~df.index.isin(used_indices)].to_numpy()
    shopping_indices = rng.choice(shopping_candidates, size=min(n_shopping, len(shopping_candidates)), replace=False)

    all_prescriber_ids = prescribers_df["prescriber_id"].values
    all_pharmacy_ids = pharmacies_df["pharmacy_id"].values

    for i, idx in enumerate(shopping_indices):
        # Assign to doctor shopper members
        if len(doctor_shoppers) > 0:
            df.at[idx, "member_id"] = rng.choice(doctor_shoppers)
        # Force controlled substance
        ctrl_class = rng.choice(controlled_classes)
        ctrl_drugs = drug_by_class.get(ctrl_class, [])
        if ctrl_drugs:
            drug = ctrl_drugs[rng.integers(0, len(ctrl_drugs))]
            df.at[idx, "ndc"] = drug["ndc"]
            df.at[idx, "drug_name"] = drug["drug_name"]
            df.at[idx, "generic_name"] = drug["generic_name"]
            df.at[idx, "therapeutic_class"] = drug["therapeutic_class"]
            df.at[idx, "dea_schedule"] = drug["dea_schedule"]
            df.at[idx, "mme_factor"] = drug["mme_factor"]
            df.at[idx, "is_brand"] = drug["is_brand"]
        # Vary prescribers (shopping means many different doctors)
        df.at[idx, "prescriber_id"] = rng.choice(all_prescriber_ids)
        # Vary pharmacies too (avoid getting caught at one pharmacy)
        df.at[idx, "pharmacy_id"] = rng.choice(all_pharmacy_ids)
        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = "doctor_shopping"

    used_indices.update(shopping_indices)
    print(f"    3. Doctor Shopping: {len(shopping_indices):,} claims")

    # ---- 4. Therapeutic Duplication (10%) ----
    n_dup = anomaly_counts["therapeutic_duplication"]
    dup_candidates = df.index[~df.index.isin(used_indices)].to_numpy()
    dup_indices = rng.choice(dup_candidates, size=min(n_dup, len(dup_candidates)), replace=False)

    # Classes with multiple drugs (where duplication can happen)
    dup_classes = ["STATIN", "ACE_INHIBITOR", "ARB", "SSRI", "SNRI", "PPI",
                   "OPIOID_ANALGESIC", "BENZODIAZEPINE", "BETA_BLOCKER"]

    for idx in dup_indices:
        # Pick a class with multiple drugs and assign a different drug from same class
        tc = rng.choice(dup_classes)
        class_drugs = drug_by_class.get(tc, [])
        if len(class_drugs) >= 2:
            drug = class_drugs[rng.integers(0, len(class_drugs))]
            df.at[idx, "ndc"] = drug["ndc"]
            df.at[idx, "drug_name"] = drug["drug_name"]
            df.at[idx, "generic_name"] = drug["generic_name"]
            df.at[idx, "therapeutic_class"] = drug["therapeutic_class"]
            df.at[idx, "dea_schedule"] = drug["dea_schedule"]
            df.at[idx, "mme_factor"] = drug["mme_factor"]
            df.at[idx, "is_brand"] = drug["is_brand"]
            # Use a different prescriber (the patient is seeing multiple docs)
            df.at[idx, "prescriber_id"] = rng.choice(all_prescriber_ids)
        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = "therapeutic_duplication"

    used_indices.update(dup_indices)
    print(f"    4. Therapeutic Duplication: {len(dup_indices):,} claims")

    # ---- 5. Phantom Billing (15%) ----
    n_phantom = anomaly_counts["phantom_billing"]
    phantom_candidates = df.index[~df.index.isin(used_indices)].to_numpy()
    phantom_indices = rng.choice(phantom_candidates, size=min(n_phantom, len(phantom_candidates)), replace=False)

    # Wrong diagnosis codes for the drug (mismatched drug-diagnosis)
    wrong_diag_map = {
        "OPIOID_ANALGESIC": ("E11.9", "Type 2 diabetes"),           # diabetes diag for opioid
        "STATIN": ("M54.5", "Low back pain"),                        # back pain diag for statin
        "ACE_INHIBITOR": ("F32.1", "Major depressive disorder"),     # depression diag for BP med
        "SSRI": ("I10", "Essential hypertension"),                   # hypertension diag for SSRI
        "BENZODIAZEPINE": ("E78.5", "Hyperlipidemia"),               # cholesterol diag for benzo
        "DIABETES_ORAL": ("J45.20", "Mild intermittent asthma"),     # asthma diag for diabetes
        "PPI": ("F41.1", "Generalized anxiety disorder"),            # anxiety diag for PPI
        "BRONCHODILATOR": ("K21.0", "GERD with esophagitis"),        # GERD diag for inhaler
    }

    for idx in phantom_indices:
        tc = df.at[idx, "therapeutic_class"]
        # Assign a wrong diagnosis code
        if tc in wrong_diag_map:
            df.at[idx, "diagnosis_code"] = wrong_diag_map[tc][0]
            df.at[idx, "diagnosis_desc"] = wrong_diag_map[tc][1]
        else:
            # Random unrelated diagnosis
            df.at[idx, "diagnosis_code"] = "Z76.89"
            df.at[idx, "diagnosis_desc"] = "Other specified encounter"
        # Phantom billing often uses suspicious pharmacies
        if len(suspicious_pharmacies) > 0 and rng.random() < 0.5:
            df.at[idx, "pharmacy_id"] = rng.choice(suspicious_pharmacies)
        # Billing spikes: cluster some phantom claims on specific dates
        if rng.random() < 0.3:
            spike_day = rng.choice([1, 15, 28])  # common billing dates
            current_date = df.at[idx, "service_date"]
            new_date = current_date.replace(day=spike_day)
            df.at[idx, "service_date"] = new_date
        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = "phantom_billing"

    used_indices.update(phantom_indices)
    print(f"    5. Phantom Billing: {len(phantom_indices):,} claims")

    # ---- 6. Upcoding / Price Manipulation (15%) ----
    n_upcoding = anomaly_counts["upcoding"]
    upcoding_candidates = df.index[~df.index.isin(used_indices)].to_numpy()
    upcoding_indices = rng.choice(upcoding_candidates, size=min(n_upcoding, len(upcoding_candidates)), replace=False)

    # Get brand/generic pairs for substitution
    brand_generic_pairs = get_brand_generic_pairs()

    # Build lookup: generic drug name → brand drug info
    drug_name_lookup = {d["drug_name"]: d for d in DRUG_REFERENCE}

    # Compounding drugs for price inflation
    compounding_drugs = drug_by_class.get("COMPOUNDING", [])

    for i, idx in enumerate(upcoding_indices):
        manipulation_type = rng.random()

        if manipulation_type < 0.35 and brand_generic_pairs:
            # Type A: Brand dispensed when generic is available
            pair = brand_generic_pairs[rng.integers(0, len(brand_generic_pairs))]
            brand_name, generic_name = pair
            brand_drug = drug_name_lookup.get(brand_name)
            if brand_drug:
                df.at[idx, "ndc"] = brand_drug["ndc"]
                df.at[idx, "drug_name"] = brand_drug["drug_name"]
                df.at[idx, "generic_name"] = brand_drug["generic_name"]
                df.at[idx, "therapeutic_class"] = brand_drug["therapeutic_class"]
                df.at[idx, "is_brand"] = True
                # Brand pricing is higher
                df.at[idx, "ingredient_cost"] = round(
                    brand_drug["awp"] * df.at[idx, "quantity"] * rng.uniform(0.90, 1.0), 2
                )
                df.at[idx, "total_cost"] = round(
                    df.at[idx, "ingredient_cost"] + df.at[idx, "dispensing_fee"], 2
                )
                df.at[idx, "plan_paid"] = round(
                    max(df.at[idx, "total_cost"] - df.at[idx, "copay"], 0), 2
                )

        elif manipulation_type < 0.70:
            # Type B: Ingredient cost inflated above AWP (1.5x-3x normal)
            matching = [d for d in DRUG_REFERENCE if d["ndc"] == df.at[idx, "ndc"]]
            if matching:
                awp = matching[0]["awp"]
                inflation_factor = rng.uniform(1.5, 3.0)
                df.at[idx, "ingredient_cost"] = round(
                    awp * df.at[idx, "quantity"] * inflation_factor, 2
                )
                df.at[idx, "total_cost"] = round(
                    df.at[idx, "ingredient_cost"] + df.at[idx, "dispensing_fee"], 2
                )
                df.at[idx, "plan_paid"] = round(
                    max(df.at[idx, "total_cost"] - df.at[idx, "copay"], 0), 2
                )

        else:
            # Type C: Compounding pharmacy charges 5x-10x normal
            if compounding_drugs:
                drug = compounding_drugs[rng.integers(0, len(compounding_drugs))]
                df.at[idx, "ndc"] = drug["ndc"]
                df.at[idx, "drug_name"] = drug["drug_name"]
                df.at[idx, "generic_name"] = drug["generic_name"]
                df.at[idx, "therapeutic_class"] = "COMPOUNDING"
                df.at[idx, "is_brand"] = drug["is_brand"]
                inflation = rng.uniform(5.0, 10.0)
                df.at[idx, "ingredient_cost"] = round(drug["awp"] * df.at[idx, "quantity"] * inflation, 2)
                df.at[idx, "dispensing_fee"] = round(rng.uniform(15.0, 50.0), 2)
                df.at[idx, "total_cost"] = round(
                    df.at[idx, "ingredient_cost"] + df.at[idx, "dispensing_fee"], 2
                )
                df.at[idx, "plan_paid"] = round(
                    max(df.at[idx, "total_cost"] - df.at[idx, "copay"], 0), 2
                )
                # Route to suspicious compounding pharmacy
                comp_pharms = pharmacies_df[
                    (pharmacies_df["pharmacy_type"] == "compounding") & pharmacies_df["is_suspicious"]
                ]["pharmacy_id"].values
                if len(comp_pharms) > 0:
                    df.at[idx, "pharmacy_id"] = rng.choice(comp_pharms)

        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = "upcoding"

    used_indices.update(upcoding_indices)
    print(f"    6. Upcoding / Price Manipulation: {len(upcoding_indices):,} claims")

    # ---- 7. Refill Too Soon (10%) ----
    n_refill = anomaly_counts["refill_too_soon"]
    refill_candidates = df.index[~df.index.isin(used_indices)].to_numpy()
    refill_indices = rng.choice(refill_candidates, size=min(n_refill, len(refill_candidates)), replace=False)

    for idx in refill_indices:
        # Set a high refill number (indicating frequent refills)
        df.at[idx, "refill_number"] = int(rng.integers(5, 15))
        # Reduce days_supply to create an early-refill pattern
        original_ds = df.at[idx, "days_supply"]
        # Claim filled before 75% of days_supply elapsed
        df.at[idx, "days_supply"] = max(7, int(original_ds * rng.uniform(0.3, 0.6)))
        # Prefer controlled substances for refill-too-soon
        if rng.random() < 0.65:
            ctrl_class = rng.choice(controlled_classes)
            ctrl_drugs = drug_by_class.get(ctrl_class, [])
            if ctrl_drugs:
                drug = ctrl_drugs[rng.integers(0, len(ctrl_drugs))]
                df.at[idx, "ndc"] = drug["ndc"]
                df.at[idx, "drug_name"] = drug["drug_name"]
                df.at[idx, "generic_name"] = drug["generic_name"]
                df.at[idx, "therapeutic_class"] = drug["therapeutic_class"]
                df.at[idx, "dea_schedule"] = drug["dea_schedule"]
                df.at[idx, "mme_factor"] = drug["mme_factor"]
                df.at[idx, "is_brand"] = drug["is_brand"]
        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = "refill_too_soon"

    used_indices.update(refill_indices)
    print(f"    7. Refill Too Soon: {len(refill_indices):,} claims")

    # ---- Summary ----
    total_injected = df["is_anomaly"].sum()
    actual_rate = total_injected / len(df) * 100
    print(f"  Total anomalies: {total_injected:,} ({actual_rate:.2f}%)")
    print(f"  Anomaly breakdown:")
    for atype, count in df[df["is_anomaly"]]["anomaly_type"].value_counts().items():
        print(f"    {atype}: {count:,}")

    return df
