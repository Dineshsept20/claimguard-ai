"""Drug reference data, NDC codes, therapeutic classes, and pricing.

Contains realistic pharmacy reference tables mirroring real-world drug data
used in PBM claims adjudication systems (RxClaim, FEP, EDF).
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Drug Reference Table — 55 drugs across 10 therapeutic categories
# Fields mirror real PBM adjudication: NDC, GPI, AWP, DEA schedule, etc.
# ---------------------------------------------------------------------------

DRUG_REFERENCE = [
    # --- OPIOIDS (Schedule II) — High fraud target ---
    {"drug_name": "Oxycodone 30mg", "generic_name": "oxycodone", "ndc": "00406-0230-01",
     "therapeutic_class": "OPIOID_ANALGESIC", "gpi_code": "65100020", "dea_schedule": "II",
     "awp": 8.50, "typical_quantity": 120, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 1.5},
    {"drug_name": "OxyContin 80mg", "generic_name": "oxycodone ER", "ndc": "59011-0480-20",
     "therapeutic_class": "OPIOID_ANALGESIC", "gpi_code": "65100025", "dea_schedule": "II",
     "awp": 32.00, "typical_quantity": 60, "typical_days_supply": 30,
     "route": "oral", "is_brand": True, "mme_factor": 3.0},
    {"drug_name": "Hydrocodone/APAP 10-325mg", "generic_name": "hydrocodone/acetaminophen", "ndc": "00591-0540-01",
     "therapeutic_class": "OPIOID_ANALGESIC", "gpi_code": "65160030", "dea_schedule": "II",
     "awp": 4.20, "typical_quantity": 120, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 1.0},
    {"drug_name": "Fentanyl Patch 75mcg", "generic_name": "fentanyl transdermal", "ndc": "00591-3765-01",
     "therapeutic_class": "OPIOID_ANALGESIC", "gpi_code": "65200040", "dea_schedule": "II",
     "awp": 55.00, "typical_quantity": 5, "typical_days_supply": 30,
     "route": "transdermal", "is_brand": False, "mme_factor": 7.2},
    {"drug_name": "Morphine Sulfate ER 60mg", "generic_name": "morphine ER", "ndc": "00406-8360-01",
     "therapeutic_class": "OPIOID_ANALGESIC", "gpi_code": "65100045", "dea_schedule": "II",
     "awp": 6.80, "typical_quantity": 60, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 1.0},

    # --- BENZODIAZEPINES (Schedule IV) — Abuse target ---
    {"drug_name": "Alprazolam 2mg", "generic_name": "alprazolam", "ndc": "00093-0502-01",
     "therapeutic_class": "BENZODIAZEPINE", "gpi_code": "57100010", "dea_schedule": "IV",
     "awp": 1.80, "typical_quantity": 90, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Diazepam 10mg", "generic_name": "diazepam", "ndc": "00603-2580-21",
     "therapeutic_class": "BENZODIAZEPINE", "gpi_code": "57100020", "dea_schedule": "IV",
     "awp": 1.20, "typical_quantity": 90, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Lorazepam 1mg", "generic_name": "lorazepam", "ndc": "00591-2474-01",
     "therapeutic_class": "BENZODIAZEPINE", "gpi_code": "57100030", "dea_schedule": "IV",
     "awp": 0.90, "typical_quantity": 90, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Clonazepam 1mg", "generic_name": "clonazepam", "ndc": "00093-0833-01",
     "therapeutic_class": "BENZODIAZEPINE", "gpi_code": "57100040", "dea_schedule": "IV",
     "awp": 0.70, "typical_quantity": 60, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},

    # --- STIMULANTS (Schedule II) — Abuse/diversion target ---
    {"drug_name": "Adderall XR 30mg", "generic_name": "amphetamine/dextroamphetamine ER", "ndc": "54092-0391-01",
     "therapeutic_class": "STIMULANT", "gpi_code": "61100010", "dea_schedule": "II",
     "awp": 12.50, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Methylphenidate ER 36mg", "generic_name": "methylphenidate ER", "ndc": "00591-2715-01",
     "therapeutic_class": "STIMULANT", "gpi_code": "61100020", "dea_schedule": "II",
     "awp": 8.00, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},

    # --- SPECIALTY / HIGH-COST DRUGS — Waste/quantity manipulation target ---
    {"drug_name": "Humira 40mg Pen", "generic_name": "adalimumab", "ndc": "00074-4339-02",
     "therapeutic_class": "BIOLOGIC_IMMUNOLOGY", "gpi_code": "66400010", "dea_schedule": "NONE",
     "awp": 2850.00, "typical_quantity": 2, "typical_days_supply": 28,
     "route": "subcutaneous", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Enbrel 50mg", "generic_name": "etanercept", "ndc": "58406-0425-04",
     "therapeutic_class": "BIOLOGIC_IMMUNOLOGY", "gpi_code": "66400020", "dea_schedule": "NONE",
     "awp": 1550.00, "typical_quantity": 4, "typical_days_supply": 28,
     "route": "subcutaneous", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Stelara 45mg", "generic_name": "ustekinumab", "ndc": "57894-0060-03",
     "therapeutic_class": "BIOLOGIC_IMMUNOLOGY", "gpi_code": "66400030", "dea_schedule": "NONE",
     "awp": 13500.00, "typical_quantity": 1, "typical_days_supply": 84,
     "route": "subcutaneous", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Revlimid 25mg", "generic_name": "lenalidomide", "ndc": "59572-0425-00",
     "therapeutic_class": "ONCOLOGY", "gpi_code": "21200010", "dea_schedule": "NONE",
     "awp": 850.00, "typical_quantity": 21, "typical_days_supply": 28,
     "route": "oral", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Harvoni", "generic_name": "ledipasvir/sofosbuvir", "ndc": "61958-1801-01",
     "therapeutic_class": "HEPATITIS_C", "gpi_code": "12300010", "dea_schedule": "NONE",
     "awp": 1125.00, "typical_quantity": 28, "typical_days_supply": 28,
     "route": "oral", "is_brand": True, "mme_factor": 0},

    # --- STATINS (Maintenance — normal baseline) ---
    {"drug_name": "Atorvastatin 40mg", "generic_name": "atorvastatin", "ndc": "00378-2040-77",
     "therapeutic_class": "STATIN", "gpi_code": "39400010", "dea_schedule": "NONE",
     "awp": 0.45, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Rosuvastatin 20mg", "generic_name": "rosuvastatin", "ndc": "00591-3762-30",
     "therapeutic_class": "STATIN", "gpi_code": "39400020", "dea_schedule": "NONE",
     "awp": 0.55, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Simvastatin 40mg", "generic_name": "simvastatin", "ndc": "00093-7154-98",
     "therapeutic_class": "STATIN", "gpi_code": "39400030", "dea_schedule": "NONE",
     "awp": 0.25, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Pravastatin 40mg", "generic_name": "pravastatin", "ndc": "00591-0446-01",
     "therapeutic_class": "STATIN", "gpi_code": "39400040", "dea_schedule": "NONE",
     "awp": 0.35, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Lipitor 40mg", "generic_name": "atorvastatin", "ndc": "00071-0157-23",
     "therapeutic_class": "STATIN", "gpi_code": "39400010", "dea_schedule": "NONE",
     "awp": 7.50, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": True, "mme_factor": 0},

    # --- BLOOD PRESSURE MEDS (Maintenance — normal baseline) ---
    {"drug_name": "Lisinopril 20mg", "generic_name": "lisinopril", "ndc": "00093-1040-01",
     "therapeutic_class": "ACE_INHIBITOR", "gpi_code": "36100010", "dea_schedule": "NONE",
     "awp": 0.20, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Losartan 100mg", "generic_name": "losartan", "ndc": "00093-7367-98",
     "therapeutic_class": "ARB", "gpi_code": "36200010", "dea_schedule": "NONE",
     "awp": 0.30, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Amlodipine 10mg", "generic_name": "amlodipine", "ndc": "00093-3171-98",
     "therapeutic_class": "CALCIUM_CHANNEL_BLOCKER", "gpi_code": "34100010", "dea_schedule": "NONE",
     "awp": 0.15, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Metoprolol Succinate ER 100mg", "generic_name": "metoprolol ER", "ndc": "00378-1054-01",
     "therapeutic_class": "BETA_BLOCKER", "gpi_code": "33200010", "dea_schedule": "NONE",
     "awp": 0.40, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Valsartan 160mg", "generic_name": "valsartan", "ndc": "00093-7392-98",
     "therapeutic_class": "ARB", "gpi_code": "36200020", "dea_schedule": "NONE",
     "awp": 0.50, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},

    # --- DIABETES MEDS (Maintenance — normal baseline) ---
    {"drug_name": "Metformin 1000mg", "generic_name": "metformin", "ndc": "00093-1048-01",
     "therapeutic_class": "DIABETES_ORAL", "gpi_code": "27200010", "dea_schedule": "NONE",
     "awp": 0.12, "typical_quantity": 60, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Glipizide 10mg", "generic_name": "glipizide", "ndc": "00093-0321-01",
     "therapeutic_class": "DIABETES_ORAL", "gpi_code": "27200020", "dea_schedule": "NONE",
     "awp": 0.15, "typical_quantity": 60, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Januvia 100mg", "generic_name": "sitagliptin", "ndc": "00006-0277-31",
     "therapeutic_class": "DIABETES_ORAL", "gpi_code": "27200030", "dea_schedule": "NONE",
     "awp": 16.50, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Lantus Solostar 100u/ml", "generic_name": "insulin glargine", "ndc": "00088-2220-05",
     "therapeutic_class": "DIABETES_INSULIN", "gpi_code": "27100010", "dea_schedule": "NONE",
     "awp": 45.00, "typical_quantity": 5, "typical_days_supply": 30,
     "route": "subcutaneous", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Ozempic 1mg", "generic_name": "semaglutide", "ndc": "00169-4132-12",
     "therapeutic_class": "DIABETES_GLP1", "gpi_code": "27300010", "dea_schedule": "NONE",
     "awp": 935.00, "typical_quantity": 1, "typical_days_supply": 28,
     "route": "subcutaneous", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Trulicity 1.5mg", "generic_name": "dulaglutide", "ndc": "00002-1474-80",
     "therapeutic_class": "DIABETES_GLP1", "gpi_code": "27300020", "dea_schedule": "NONE",
     "awp": 890.00, "typical_quantity": 4, "typical_days_supply": 28,
     "route": "subcutaneous", "is_brand": True, "mme_factor": 0},

    # --- ANTIDEPRESSANTS / MENTAL HEALTH ---
    {"drug_name": "Sertraline 100mg", "generic_name": "sertraline", "ndc": "00093-7198-05",
     "therapeutic_class": "SSRI", "gpi_code": "58200010", "dea_schedule": "NONE",
     "awp": 0.35, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Escitalopram 20mg", "generic_name": "escitalopram", "ndc": "00093-5852-56",
     "therapeutic_class": "SSRI", "gpi_code": "58200020", "dea_schedule": "NONE",
     "awp": 0.40, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Duloxetine 60mg", "generic_name": "duloxetine", "ndc": "00002-3235-30",
     "therapeutic_class": "SNRI", "gpi_code": "58300010", "dea_schedule": "NONE",
     "awp": 0.60, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Bupropion XL 300mg", "generic_name": "bupropion XL", "ndc": "00093-7198-98",
     "therapeutic_class": "ANTIDEPRESSANT_OTHER", "gpi_code": "58400010", "dea_schedule": "NONE",
     "awp": 0.80, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Quetiapine 200mg", "generic_name": "quetiapine", "ndc": "00591-3525-01",
     "therapeutic_class": "ANTIPSYCHOTIC", "gpi_code": "59200010", "dea_schedule": "NONE",
     "awp": 1.50, "typical_quantity": 60, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},

    # --- RESPIRATORY ---
    {"drug_name": "Albuterol HFA Inhaler", "generic_name": "albuterol", "ndc": "00093-3174-68",
     "therapeutic_class": "BRONCHODILATOR", "gpi_code": "44200010", "dea_schedule": "NONE",
     "awp": 8.50, "typical_quantity": 1, "typical_days_supply": 30,
     "route": "inhalation", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Symbicort 160/4.5", "generic_name": "budesonide/formoterol", "ndc": "00186-0372-20",
     "therapeutic_class": "RESPIRATORY_COMBO", "gpi_code": "44300010", "dea_schedule": "NONE",
     "awp": 350.00, "typical_quantity": 1, "typical_days_supply": 30,
     "route": "inhalation", "is_brand": True, "mme_factor": 0},
    {"drug_name": "Montelukast 10mg", "generic_name": "montelukast", "ndc": "00093-7398-56",
     "therapeutic_class": "LEUKOTRIENE_INHIBITOR", "gpi_code": "44400010", "dea_schedule": "NONE",
     "awp": 0.30, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},

    # --- PROTON PUMP INHIBITORS (GI) ---
    {"drug_name": "Omeprazole 40mg", "generic_name": "omeprazole", "ndc": "00093-5289-98",
     "therapeutic_class": "PPI", "gpi_code": "49270010", "dea_schedule": "NONE",
     "awp": 0.25, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Pantoprazole 40mg", "generic_name": "pantoprazole", "ndc": "00093-0108-98",
     "therapeutic_class": "PPI", "gpi_code": "49270020", "dea_schedule": "NONE",
     "awp": 0.20, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},

    # --- THYROID ---
    {"drug_name": "Levothyroxine 100mcg", "generic_name": "levothyroxine", "ndc": "00378-1810-01",
     "therapeutic_class": "THYROID", "gpi_code": "28100010", "dea_schedule": "NONE",
     "awp": 0.30, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Synthroid 100mcg", "generic_name": "levothyroxine", "ndc": "00074-6624-13",
     "therapeutic_class": "THYROID", "gpi_code": "28100010", "dea_schedule": "NONE",
     "awp": 2.80, "typical_quantity": 30, "typical_days_supply": 30,
     "route": "oral", "is_brand": True, "mme_factor": 0},

    # --- ANTIBIOTICS (Acute — short courses) ---
    {"drug_name": "Amoxicillin 500mg", "generic_name": "amoxicillin", "ndc": "00093-3109-01",
     "therapeutic_class": "ANTIBIOTIC", "gpi_code": "01200010", "dea_schedule": "NONE",
     "awp": 0.10, "typical_quantity": 30, "typical_days_supply": 10,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Azithromycin 250mg", "generic_name": "azithromycin", "ndc": "00093-7169-56",
     "therapeutic_class": "ANTIBIOTIC", "gpi_code": "01200020", "dea_schedule": "NONE",
     "awp": 0.80, "typical_quantity": 6, "typical_days_supply": 5,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Ciprofloxacin 500mg", "generic_name": "ciprofloxacin", "ndc": "00093-0862-01",
     "therapeutic_class": "ANTIBIOTIC", "gpi_code": "01200030", "dea_schedule": "NONE",
     "awp": 0.50, "typical_quantity": 20, "typical_days_supply": 10,
     "route": "oral", "is_brand": False, "mme_factor": 0},

    # --- COMPOUNDING (High fraud risk — price manipulation) ---
    {"drug_name": "Custom Pain Cream", "generic_name": "compounded topical", "ndc": "99999-0001-01",
     "therapeutic_class": "COMPOUNDING", "gpi_code": "99000010", "dea_schedule": "NONE",
     "awp": 180.00, "typical_quantity": 1, "typical_days_supply": 30,
     "route": "topical", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Custom Scar Cream", "generic_name": "compounded topical", "ndc": "99999-0002-01",
     "therapeutic_class": "COMPOUNDING", "gpi_code": "99000020", "dea_schedule": "NONE",
     "awp": 250.00, "typical_quantity": 1, "typical_days_supply": 30,
     "route": "topical", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Custom Hormone Cream", "generic_name": "compounded hormone", "ndc": "99999-0003-01",
     "therapeutic_class": "COMPOUNDING", "gpi_code": "99000030", "dea_schedule": "NONE",
     "awp": 320.00, "typical_quantity": 1, "typical_days_supply": 30,
     "route": "topical", "is_brand": False, "mme_factor": 0},

    # --- MUSCLE RELAXANTS ---
    {"drug_name": "Cyclobenzaprine 10mg", "generic_name": "cyclobenzaprine", "ndc": "00591-5528-01",
     "therapeutic_class": "MUSCLE_RELAXANT", "gpi_code": "62100010", "dea_schedule": "NONE",
     "awp": 0.15, "typical_quantity": 90, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Gabapentin 300mg", "generic_name": "gabapentin", "ndc": "00591-2658-01",
     "therapeutic_class": "ANTICONVULSANT", "gpi_code": "72100010", "dea_schedule": "V",
     "awp": 0.25, "typical_quantity": 90, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
    {"drug_name": "Pregabalin 150mg", "generic_name": "pregabalin", "ndc": "00071-1015-68",
     "therapeutic_class": "ANTICONVULSANT", "gpi_code": "72100020", "dea_schedule": "V",
     "awp": 4.50, "typical_quantity": 60, "typical_days_supply": 30,
     "route": "oral", "is_brand": False, "mme_factor": 0},
]


# ---------------------------------------------------------------------------
# Drug → Diagnosis (ICD-10) Mapping
# Maps therapeutic classes to realistic diagnosis codes for claim validation
# ---------------------------------------------------------------------------

DRUG_DIAGNOSIS_MAP = {
    "OPIOID_ANALGESIC": [
        ("M54.5", "Low back pain"),
        ("M54.2", "Cervicalgia"),
        ("G89.29", "Other chronic pain"),
        ("M25.511", "Pain in right shoulder"),
        ("G89.4", "Chronic pain syndrome"),
    ],
    "BENZODIAZEPINE": [
        ("F41.1", "Generalized anxiety disorder"),
        ("F41.0", "Panic disorder"),
        ("G47.00", "Insomnia, unspecified"),
        ("G40.909", "Epilepsy, unspecified"),
    ],
    "STIMULANT": [
        ("F90.0", "ADHD, predominantly inattentive"),
        ("F90.1", "ADHD, predominantly hyperactive"),
        ("F90.2", "ADHD, combined type"),
        ("G47.419", "Narcolepsy without cataplexy"),
    ],
    "BIOLOGIC_IMMUNOLOGY": [
        ("M06.9", "Rheumatoid arthritis, unspecified"),
        ("L40.0", "Psoriasis vulgaris"),
        ("K50.90", "Crohn's disease, unspecified"),
        ("M45.9", "Ankylosing spondylitis"),
    ],
    "ONCOLOGY": [
        ("C90.00", "Multiple myeloma"),
        ("C91.10", "Chronic lymphocytic leukemia"),
    ],
    "HEPATITIS_C": [
        ("B18.2", "Chronic viral hepatitis C"),
    ],
    "STATIN": [
        ("E78.5", "Hyperlipidemia, unspecified"),
        ("E78.0", "Pure hypercholesterolemia"),
        ("E78.2", "Mixed hyperlipidemia"),
    ],
    "ACE_INHIBITOR": [
        ("I10", "Essential hypertension"),
        ("I50.9", "Heart failure, unspecified"),
        ("I25.10", "Atherosclerotic heart disease"),
    ],
    "ARB": [
        ("I10", "Essential hypertension"),
        ("I50.9", "Heart failure, unspecified"),
    ],
    "CALCIUM_CHANNEL_BLOCKER": [
        ("I10", "Essential hypertension"),
        ("I20.9", "Angina pectoris, unspecified"),
    ],
    "BETA_BLOCKER": [
        ("I10", "Essential hypertension"),
        ("I48.91", "Atrial fibrillation"),
        ("I50.9", "Heart failure, unspecified"),
    ],
    "DIABETES_ORAL": [
        ("E11.9", "Type 2 diabetes without complications"),
        ("E11.65", "Type 2 diabetes with hyperglycemia"),
    ],
    "DIABETES_INSULIN": [
        ("E10.9", "Type 1 diabetes without complications"),
        ("E11.9", "Type 2 diabetes without complications"),
        ("E11.65", "Type 2 diabetes with hyperglycemia"),
    ],
    "DIABETES_GLP1": [
        ("E11.9", "Type 2 diabetes without complications"),
        ("E11.65", "Type 2 diabetes with hyperglycemia"),
        ("E66.01", "Morbid obesity due to excess calories"),
    ],
    "SSRI": [
        ("F32.1", "Major depressive disorder, moderate"),
        ("F33.1", "Major depressive disorder, recurrent moderate"),
        ("F41.1", "Generalized anxiety disorder"),
    ],
    "SNRI": [
        ("F32.1", "Major depressive disorder, moderate"),
        ("M79.7", "Fibromyalgia"),
        ("G89.29", "Other chronic pain"),
    ],
    "ANTIDEPRESSANT_OTHER": [
        ("F32.1", "Major depressive disorder, moderate"),
        ("F17.210", "Nicotine dependence, cigarettes"),
    ],
    "ANTIPSYCHOTIC": [
        ("F20.9", "Schizophrenia, unspecified"),
        ("F31.9", "Bipolar disorder, unspecified"),
        ("F32.2", "Major depressive disorder, severe"),
    ],
    "BRONCHODILATOR": [
        ("J45.20", "Mild intermittent asthma"),
        ("J44.1", "COPD with acute exacerbation"),
    ],
    "RESPIRATORY_COMBO": [
        ("J45.40", "Moderate persistent asthma"),
        ("J44.0", "COPD with acute lower respiratory infection"),
    ],
    "LEUKOTRIENE_INHIBITOR": [
        ("J45.20", "Mild intermittent asthma"),
        ("J30.1", "Allergic rhinitis due to pollen"),
    ],
    "PPI": [
        ("K21.0", "GERD with esophagitis"),
        ("K25.9", "Gastric ulcer, unspecified"),
        ("K29.70", "Gastritis, unspecified"),
    ],
    "THYROID": [
        ("E03.9", "Hypothyroidism, unspecified"),
    ],
    "ANTIBIOTIC": [
        ("J06.9", "Acute upper respiratory infection"),
        ("J02.9", "Acute pharyngitis, unspecified"),
        ("N39.0", "Urinary tract infection"),
        ("J18.9", "Pneumonia, unspecified"),
    ],
    "COMPOUNDING": [
        ("M79.3", "Panniculitis, unspecified"),
        ("L90.5", "Scar conditions"),
        ("M54.5", "Low back pain"),
    ],
    "MUSCLE_RELAXANT": [
        ("M62.830", "Muscle spasm of back"),
        ("M54.5", "Low back pain"),
    ],
    "ANTICONVULSANT": [
        ("G40.909", "Epilepsy, unspecified"),
        ("M79.7", "Fibromyalgia"),
        ("G89.29", "Other chronic pain"),
    ],
}


# ---------------------------------------------------------------------------
# Condition → Drug Class Mapping
# Drives realistic prescribing: patients with condition X get drug class Y
# ---------------------------------------------------------------------------

CONDITION_DRUG_CLASSES = {
    "hypertension": ["ACE_INHIBITOR", "ARB", "CALCIUM_CHANNEL_BLOCKER", "BETA_BLOCKER"],
    "diabetes_type2": ["DIABETES_ORAL", "DIABETES_INSULIN", "DIABETES_GLP1"],
    "diabetes_type1": ["DIABETES_INSULIN"],
    "hyperlipidemia": ["STATIN"],
    "chronic_pain": ["OPIOID_ANALGESIC", "MUSCLE_RELAXANT", "ANTICONVULSANT", "SNRI"],
    "anxiety": ["BENZODIAZEPINE", "SSRI"],
    "depression": ["SSRI", "SNRI", "ANTIDEPRESSANT_OTHER"],
    "adhd": ["STIMULANT"],
    "asthma": ["BRONCHODILATOR", "RESPIRATORY_COMBO", "LEUKOTRIENE_INHIBITOR"],
    "copd": ["BRONCHODILATOR", "RESPIRATORY_COMBO"],
    "rheumatoid_arthritis": ["BIOLOGIC_IMMUNOLOGY"],
    "hypothyroidism": ["THYROID"],
    "gerd": ["PPI"],
    "bipolar": ["ANTIPSYCHOTIC"],
    "schizophrenia": ["ANTIPSYCHOTIC"],
}


# ---------------------------------------------------------------------------
# US States for entity distribution
# ---------------------------------------------------------------------------

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

# Population-weighted state probabilities (top states get more entities)
STATE_WEIGHTS = {
    "CA": 0.12, "TX": 0.09, "FL": 0.07, "NY": 0.06, "PA": 0.04,
    "IL": 0.04, "OH": 0.04, "GA": 0.03, "NC": 0.03, "MI": 0.03,
    "NJ": 0.03, "VA": 0.03, "WA": 0.02, "AZ": 0.02, "MA": 0.02,
    "TN": 0.02, "IN": 0.02, "MO": 0.02, "MD": 0.02, "WI": 0.02,
    "CO": 0.02, "MN": 0.02, "SC": 0.02, "AL": 0.015, "LA": 0.015,
    "KY": 0.014, "OR": 0.013, "OK": 0.012, "CT": 0.011, "UT": 0.010,
    "IA": 0.010, "NV": 0.010, "AR": 0.009, "MS": 0.009, "KS": 0.009,
    "NM": 0.006, "NE": 0.006, "ID": 0.006, "WV": 0.005, "HI": 0.004,
    "NH": 0.004, "ME": 0.004, "RI": 0.003, "MT": 0.003, "DE": 0.003,
    "SD": 0.003, "ND": 0.002, "AK": 0.002, "VT": 0.002, "WY": 0.002,
}

# Prescriber specialties
SPECIALTIES = [
    "Family Medicine", "Internal Medicine", "Pain Management",
    "Orthopedics", "Psychiatry", "Neurology", "Oncology",
    "Rheumatology", "Endocrinology", "Pulmonology",
    "Gastroenterology", "Cardiology", "Dermatology",
    "General Surgery", "Emergency Medicine", "Pediatrics",
    "OB/GYN", "Urology", "Ophthalmology", "Dentistry",
]

# Specialty → typical drug class mappings (for specialty_match feature)
SPECIALTY_DRUG_CLASSES = {
    "Pain Management": ["OPIOID_ANALGESIC", "MUSCLE_RELAXANT", "ANTICONVULSANT", "COMPOUNDING"],
    "Psychiatry": ["BENZODIAZEPINE", "SSRI", "SNRI", "ANTIDEPRESSANT_OTHER", "ANTIPSYCHOTIC", "STIMULANT"],
    "Oncology": ["ONCOLOGY", "OPIOID_ANALGESIC"],
    "Rheumatology": ["BIOLOGIC_IMMUNOLOGY"],
    "Endocrinology": ["DIABETES_ORAL", "DIABETES_INSULIN", "DIABETES_GLP1", "THYROID"],
    "Pulmonology": ["BRONCHODILATOR", "RESPIRATORY_COMBO", "LEUKOTRIENE_INHIBITOR"],
    "Gastroenterology": ["PPI", "HEPATITIS_C"],
    "Cardiology": ["STATIN", "ACE_INHIBITOR", "ARB", "BETA_BLOCKER", "CALCIUM_CHANNEL_BLOCKER"],
    "Neurology": ["ANTICONVULSANT", "BENZODIAZEPINE"],
    "Dermatology": ["COMPOUNDING"],
    "Orthopedics": ["OPIOID_ANALGESIC", "MUSCLE_RELAXANT"],
    "Family Medicine": None,    # Can prescribe anything
    "Internal Medicine": None,  # Can prescribe anything
    "Pediatrics": ["STIMULANT", "ANTIBIOTIC", "BRONCHODILATOR"],
    "General Surgery": ["OPIOID_ANALGESIC", "ANTIBIOTIC"],
    "Emergency Medicine": ["OPIOID_ANALGESIC", "ANTIBIOTIC", "BENZODIAZEPINE"],
    "Dentistry": ["OPIOID_ANALGESIC", "ANTIBIOTIC"],
    "OB/GYN": ["ANTIBIOTIC", "THYROID", "SSRI"],
    "Urology": ["ANTIBIOTIC"],
    "Ophthalmology": ["ANTIBIOTIC"],
}

# Pharmacy types
PHARMACY_TYPES = ["retail", "mail_order", "specialty", "compounding"]
PHARMACY_TYPE_WEIGHTS = [0.70, 0.10, 0.10, 0.10]

# Major pharmacy chains
PHARMACY_CHAINS = [
    "CVS Pharmacy", "Walgreens", "Rite Aid", "Walmart Pharmacy",
    "Kroger Pharmacy", "Costco Pharmacy", "Sam's Club Pharmacy",
    "Publix Pharmacy", "H-E-B Pharmacy", "Safeway Pharmacy",
]


def get_drug_reference_df() -> pd.DataFrame:
    """Return the drug reference table as a pandas DataFrame."""
    return pd.DataFrame(DRUG_REFERENCE)


def get_drugs_by_class(therapeutic_class: str) -> list[dict]:
    """Return all drugs belonging to a specific therapeutic class."""
    return [d for d in DRUG_REFERENCE if d["therapeutic_class"] == therapeutic_class]


def get_diagnosis_for_drug(therapeutic_class: str, rng: np.random.Generator = None) -> tuple[str, str]:
    """Return a random (icd_code, description) for a given drug's therapeutic class."""
    if rng is None:
        rng = np.random.default_rng()
    diagnoses = DRUG_DIAGNOSIS_MAP.get(therapeutic_class, [("Z76.89", "Other encounter")])
    idx = rng.integers(0, len(diagnoses))
    return diagnoses[idx]


def get_brand_generic_pairs() -> list[tuple[str, str]]:
    """Return pairs of (brand_drug, generic_drug) for upcoding detection.

    Identifies cases where a brand drug has a cheaper generic equivalent.
    """
    pairs = []
    generic_by_gpi = {}
    brand_by_gpi = {}
    for drug in DRUG_REFERENCE:
        gpi = drug["gpi_code"]
        if drug["is_brand"]:
            brand_by_gpi.setdefault(gpi, []).append(drug)
        else:
            generic_by_gpi.setdefault(gpi, []).append(drug)

    for gpi in brand_by_gpi:
        if gpi in generic_by_gpi:
            for brand in brand_by_gpi[gpi]:
                for generic in generic_by_gpi[gpi]:
                    pairs.append((brand["drug_name"], generic["drug_name"]))
    return pairs
