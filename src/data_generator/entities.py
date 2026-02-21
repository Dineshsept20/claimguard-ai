"""Entity generation: pharmacies, prescribers, and members/patients.

Generates realistic healthcare entities with attributes that mirror
real-world PBM data. A configurable percentage of entities are flagged
as potential bad actors for anomaly injection.
"""

import numpy as np
import pandas as pd
from faker import Faker

from .reference_data import (
    CONDITION_DRUG_CLASSES,
    PHARMACY_CHAINS,
    PHARMACY_TYPES,
    PHARMACY_TYPE_WEIGHTS,
    SPECIALTIES,
    STATE_WEIGHTS,
    US_STATES,
)

fake = Faker()
Faker.seed(42)


def _weighted_state_choice(rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Pick states weighted by population."""
    states = list(STATE_WEIGHTS.keys())
    probs = np.array(list(STATE_WEIGHTS.values()))
    probs = probs / probs.sum()  # normalize
    return rng.choice(states, size=size, p=probs)


def _generate_npi(rng: np.random.Generator) -> str:
    """Generate a realistic 10-digit NPI number."""
    return str(rng.integers(1_000_000_000, 9_999_999_999))


def _generate_dea_number(rng: np.random.Generator, last_name_initial: str) -> str:
    """Generate a realistic DEA number (2 letters + 7 digits)."""
    prefix_letter = rng.choice(list("ABFM"))
    digits = str(rng.integers(1_000_000, 9_999_999))
    return f"{prefix_letter}{last_name_initial}{digits}"


def generate_pharmacies(
    num_pharmacies: int = 500,
    suspicious_rate: float = 0.07,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate pharmacy entities.

    Args:
        num_pharmacies: Number of pharmacies to generate.
        suspicious_rate: Fraction flagged as potential bad actors (5-10%).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with pharmacy_id, name, npi, type, address, state,
        chain_flag, chain_name, is_suspicious (hidden label).
    """
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    num_suspicious = int(num_pharmacies * suspicious_rate)
    is_suspicious = np.array(
        [True] * num_suspicious + [False] * (num_pharmacies - num_suspicious)
    )
    rng.shuffle(is_suspicious)

    states = _weighted_state_choice(rng, num_pharmacies)

    # Pharmacy type distribution — suspicious ones lean toward compounding/specialty
    pharmacy_types = []
    for i in range(num_pharmacies):
        if is_suspicious[i] and rng.random() < 0.5:
            pharmacy_types.append(rng.choice(["compounding", "specialty"]))
        else:
            pharmacy_types.append(
                rng.choice(PHARMACY_TYPES, p=PHARMACY_TYPE_WEIGHTS)
            )

    # Chain vs independent
    chain_flags = []
    chain_names = []
    for i in range(num_pharmacies):
        ptype = pharmacy_types[i]
        if ptype == "retail" and rng.random() < 0.60:
            chain_flags.append(True)
            chain_names.append(rng.choice(PHARMACY_CHAINS))
        else:
            chain_flags.append(False)
            chain_names.append("")

    pharmacies = []
    for i in range(num_pharmacies):
        name = chain_names[i] if chain_flags[i] else f"{fake.last_name()} Pharmacy"
        if pharmacy_types[i] == "compounding":
            name = f"{fake.last_name()} Compounding" if not chain_flags[i] else name
        elif pharmacy_types[i] == "specialty":
            name = f"{fake.last_name()} Specialty Pharmacy" if not chain_flags[i] else name

        pharmacies.append({
            "pharmacy_id": f"PHR{i+1:05d}",
            "pharmacy_name": name,
            "npi": _generate_npi(rng),
            "pharmacy_type": pharmacy_types[i],
            "address": fake.street_address(),
            "city": fake.city(),
            "state": states[i],
            "zip_code": fake.zipcode(),
            "chain_flag": chain_flags[i],
            "chain_name": chain_names[i],
            "is_suspicious": is_suspicious[i],
        })

    return pd.DataFrame(pharmacies)


def generate_prescribers(
    num_prescribers: int = 2000,
    suspicious_rate: float = 0.04,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate prescriber entities.

    Args:
        num_prescribers: Number of prescribers to generate.
        suspicious_rate: Fraction with suspicious patterns (3-5%).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with prescriber_id, name, npi, dea_number, specialty,
        state, practice_type, is_suspicious (hidden label).
    """
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    num_suspicious = int(num_prescribers * suspicious_rate)
    is_suspicious = np.array(
        [True] * num_suspicious + [False] * (num_prescribers - num_suspicious)
    )
    rng.shuffle(is_suspicious)

    states = _weighted_state_choice(rng, num_prescribers)

    # Specialty distribution — weighted toward primary care
    specialty_weights = np.ones(len(SPECIALTIES))
    # Family Medicine and Internal Medicine are most common
    specialty_weights[0] = 5.0  # Family Medicine
    specialty_weights[1] = 4.0  # Internal Medicine
    specialty_weights[2] = 2.0  # Pain Management
    specialty_weights[4] = 2.5  # Psychiatry
    specialty_weights[11] = 2.0  # Cardiology
    specialty_weights = specialty_weights / specialty_weights.sum()

    # Suspicious prescribers lean toward pain management
    specialties_normal = rng.choice(SPECIALTIES, size=num_prescribers, p=specialty_weights)
    specialties = []
    for i in range(num_prescribers):
        if is_suspicious[i] and rng.random() < 0.6:
            specialties.append(rng.choice(["Pain Management", "Family Medicine", "Internal Medicine"]))
        else:
            specialties.append(specialties_normal[i])

    practice_types = ["solo", "group", "hospital", "clinic"]
    practice_type_weights = [0.20, 0.35, 0.25, 0.20]

    prescribers = []
    for i in range(num_prescribers):
        first_name = fake.first_name()
        last_name = fake.last_name()
        prescribers.append({
            "prescriber_id": f"PRE{i+1:05d}",
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"Dr. {first_name} {last_name}",
            "npi": _generate_npi(rng),
            "dea_number": _generate_dea_number(rng, last_name[0]),
            "specialty": specialties[i],
            "state": states[i],
            "practice_type": rng.choice(practice_types, p=practice_type_weights),
            "is_suspicious": is_suspicious[i],
        })

    return pd.DataFrame(prescribers)


def generate_members(
    num_members: int = 10000,
    doctor_shopper_rate: float = 0.025,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate member/patient entities with chronic conditions.

    Args:
        num_members: Number of members/patients to generate.
        doctor_shopper_rate: Fraction who are doctor shoppers (2-3%).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with member_id, demographics, plan_type,
        chronic_conditions, is_doctor_shopper (hidden label).
    """
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    num_shoppers = int(num_members * doctor_shopper_rate)
    is_doctor_shopper = np.array(
        [True] * num_shoppers + [False] * (num_members - num_shoppers)
    )
    rng.shuffle(is_doctor_shopper)

    states = _weighted_state_choice(rng, num_members)

    # Age distribution — skewed toward older adults (more pharmacy claims)
    ages = np.clip(
        rng.normal(loc=55, scale=18, size=num_members).astype(int),
        18, 95
    )

    # Gender
    genders = rng.choice(["M", "F"], size=num_members, p=[0.48, 0.52])

    # Plan types
    plan_types = ["Commercial", "Medicare", "Medicaid", "FEP"]
    plan_type_weights = [0.45, 0.30, 0.15, 0.10]

    # Assign plan based on age (Medicare for 65+)
    plans = []
    for i in range(num_members):
        if ages[i] >= 65:
            plans.append(rng.choice(["Medicare", "Commercial", "FEP"], p=[0.70, 0.20, 0.10]))
        elif ages[i] < 25:
            plans.append(rng.choice(["Commercial", "Medicaid"], p=[0.65, 0.35]))
        else:
            plans.append(rng.choice(plan_types, p=plan_type_weights))

    # Chronic conditions — age-correlated, drives prescribing patterns
    all_conditions = list(CONDITION_DRUG_CLASSES.keys())

    # Condition probability by age
    def _assign_conditions(age: int, rng: np.random.Generator) -> list[str]:
        conditions = []
        base_prob = min(age / 200, 0.4)  # older = more conditions

        # Age-dependent condition probabilities
        condition_probs = {
            "hypertension": 0.15 + (age - 40) * 0.005 if age > 40 else 0.03,
            "diabetes_type2": 0.10 + (age - 45) * 0.004 if age > 45 else 0.02,
            "diabetes_type1": 0.02,
            "hyperlipidemia": 0.12 + (age - 40) * 0.004 if age > 40 else 0.02,
            "chronic_pain": 0.08 + (age - 35) * 0.003 if age > 35 else 0.02,
            "anxiety": 0.10,
            "depression": 0.12,
            "adhd": 0.08 if age < 40 else 0.03,
            "asthma": 0.07,
            "copd": 0.05 + (age - 50) * 0.003 if age > 50 else 0.01,
            "rheumatoid_arthritis": 0.03,
            "hypothyroidism": 0.06,
            "gerd": 0.10,
            "bipolar": 0.02,
            "schizophrenia": 0.01,
        }

        for cond, prob in condition_probs.items():
            prob = max(0, min(prob, 0.5))  # clamp
            if rng.random() < prob:
                conditions.append(cond)

        # Everyone should have at least 1 condition (they're pharmacy patients)
        if not conditions:
            conditions.append(rng.choice(["hypertension", "hyperlipidemia", "gerd"]))

        return conditions

    members = []
    for i in range(num_members):
        first_name = fake.first_name_male() if genders[i] == "M" else fake.first_name_female()
        last_name = fake.last_name()
        conditions = _assign_conditions(ages[i], rng)

        # Doctor shoppers tend to have chronic pain
        if is_doctor_shopper[i] and "chronic_pain" not in conditions:
            conditions.append("chronic_pain")

        members.append({
            "member_id": f"MEM{i+1:06d}",
            "first_name": first_name,
            "last_name": last_name,
            "age": int(ages[i]),
            "gender": genders[i],
            "state": states[i],
            "plan_type": plans[i],
            "chronic_conditions": ",".join(sorted(conditions)),
            "num_conditions": len(conditions),
            "is_doctor_shopper": is_doctor_shopper[i],
        })

    return pd.DataFrame(members)
