# ClaimGuard AI — Anomaly Patterns Guide

> This document describes the 7 fraud/waste/abuse patterns injected into
> the synthetic claims dataset. Each pattern mirrors real-world schemes
> detected by PBMs and health plan SIU teams.

---

## Overview

| # | Pattern | % of Anomalies | Real-World Prevalence |
|---|---------|---------------|-----------------------|
| 1 | Quantity Manipulation | 15% | Common in compounding & specialty |
| 2 | Prescriber-Pharmacy Collusion | 20% | Highest-impact fraud scheme |
| 3 | Doctor Shopping | 15% | Most common member-driven abuse |
| 4 | Therapeutic Duplication | 10% | Waste / uncoordinated care |
| 5 | Phantom Billing | 15% | Pharmacy-driven fraud |
| 6 | Upcoding / Price Manipulation | 15% | Billing fraud |
| 7 | Refill Too Soon | 10% | Abuse / diversion signal |

**Target anomaly rate:** 3–5% of total claims (realistic for a health plan).

---

## 1. Quantity Manipulation

**What it is:** Pharmacy dispenses significantly more units than the
prescription calls for, then bills the plan for the inflated quantity.

**How we inject it:**
- Quantity inflated to **3×–5× the typical amount** for that drug
- Days supply may or may not change (mismatch = stronger signal)
- Cost recalculated based on inflated quantity
- **60% routed through suspicious pharmacies**

**Detection signals:**
- `quantity / typical_quantity > 3.0`
- `quantity / days_supply` ratio deviates from drug's normal ratio
- Same pharmacy repeatedly shows inflated quantities
- High `ingredient_cost` relative to AWP × typical quantity

**Real-world example:** A compounding pharmacy bills for 120g of a
topical cream when the standard prescription is 30g.

---

## 2. Prescriber-Pharmacy Collusion

**What it is:** A prescriber and pharmacy work together to generate
fraudulent or inflated claims. The prescriber sends an unusual
concentration of prescriptions to a single pharmacy.

**How we inject it:**
- **3–10 collusion pairs** created from suspicious entities
- All claims in a pair funnel through the same prescriber→pharmacy pipeline
- **70% involve controlled substances** (opioids, benzos, stimulants)
- **50% submitted during off-hours** (late night / early morning)

**Detection signals:**
- Prescriber sends >50% of all scripts to one pharmacy
- High controlled substance ratio for the pair
- Geographic mismatch (prescriber in one state, pharmacy in another)
- After-hours submission pattern
- Prescriber-pharmacy pair volume is outlier vs. peers

**Real-world example:** A pain management doctor sends 85% of opioid
scripts to a single independent pharmacy owned by a relative.

---

## 3. Doctor Shopping

**What it is:** A member visits multiple prescribers to obtain the same
controlled substance, using different pharmacies to avoid detection.

**How we inject it:**
- Claims assigned to **known doctor-shopper members** (2.5% of members)
- Drugs forced to **controlled substances** (opioids, benzos, stimulants)
- **Different prescribers** assigned to each claim (shopping behavior)
- **Different pharmacies** used (avoidance behavior)

**Detection signals:**
- Member has ≥5 unique prescribers for same drug class in 90 days
- Member uses ≥3 unique pharmacies in 90 days
- High percentage of controlled substance claims
- Elevated MME (morphine milligram equivalent) per member
- Overlapping fills (multiple active prescriptions for same class)

**Real-world example:** A patient visits 7 different doctors across 3
counties to obtain OxyContin, filling at different CVS locations.

---

## 4. Therapeutic Duplication

**What it is:** A member receives two or more drugs from the same
therapeutic class simultaneously, typically from different prescribers
who are unaware of each other's prescriptions.

**How we inject it:**
- Claims assigned to drug classes with **multiple drug options**
  (statins, SSRIs, ACE inhibitors, ARBs, PPIs, opioids, benzos)
- **Different prescribers** assigned (uncoordinated care)
- Same member, same therapeutic class, overlapping time periods

**Detection signals:**
- Same `member_id` + same `therapeutic_class` + different `prescriber_id`
  within a 30-day window
- Two active prescriptions from same drug class
- Increased total cost per member for duplicated class

**Real-world example:** Patient takes Lipitor from cardiologist and
Crestor from PCP simultaneously — both are statins.

---

## 5. Phantom Billing

**What it is:** Pharmacy bills for drugs that the member doesn't
actually need, or bills using diagnosis codes that don't match the
drug prescribed.

**How we inject it:**
- **Drug-diagnosis mismatch:** e.g., opioid billed with diabetes
  diagnosis, statin billed with back pain diagnosis
- **50% routed through suspicious pharmacies**
- **30% clustered on specific billing dates** (1st, 15th, 28th)

**Detection signals:**
- `diagnosis_code` doesn't match expected diagnoses for `therapeutic_class`
- Volume spikes on specific calendar dates
- Pharmacy has high rate of mismatched drug-diagnosis pairs
- Claims for members with no history of the billed condition

**Real-world example:** Pharmacy bills for Humira (rheumatoid arthritis
drug) using a common cold diagnosis code.

---

## 6. Upcoding / Price Manipulation

**What it is:** Pharmacy bills at inflated prices or dispenses a more
expensive version of a drug when a cheaper alternative exists.

**Three sub-patterns:**

### 6a. Brand When Generic Available (35% of upcoding)
- Brand drug dispensed and billed when a cheaper generic equivalent exists
- Uses brand/generic pairs from the drug reference table (same GPI code)

### 6b. Price Inflation (35% of upcoding)
- `ingredient_cost` inflated to **1.5×–3× the AWP** for the drug
- Same drug, same quantity, just higher price

### 6c. Compounding Markup (30% of upcoding)
- Compounding pharmacy charges **5×–10× normal** for a compound
- Dispensing fee also inflated ($15–$50 vs normal $1–$6)
- Routed through suspicious compounding pharmacies

**Detection signals:**
- `ingredient_cost / (awp × quantity) > 1.5` (cost vs. AWP ratio)
- Brand dispensed when generic available for same GPI
- Compounding pharmacy cost per claim >> peer average
- Dispensing fee significantly above normal range

**Real-world example:** Compounding pharmacy bills $3,000 for a
topical pain cream that costs $50 at a retail pharmacy.

---

## 7. Refill Too Soon

**What it is:** Member refills a prescription before the previous
supply should be exhausted, suggesting diversion (selling pills)
or abuse.

**How we inject it:**
- **High refill numbers** (5–15, indicating frequent refills)
- **Shortened days supply** (30%–60% of normal) creating overlap
- **65% involve controlled substances** (opioids, benzos, stimulants)

**Detection signals:**
- `refill_number` ≥ 5 within the observation period
- Days between fills < 75% of `days_supply`
- Controlled substance with accelerating refill frequency
- Same drug, same member, multiple fills within days_supply window

**Real-world example:** Patient fills 30-day OxyContin prescription
every 15 days, suggesting they're consuming double the prescribed
dose or diverting pills.

---

## Feature Engineering Implications

Each anomaly pattern creates specific feature signals:

| Pattern | Key Features to Engineer |
|---------|------------------------|
| Quantity Manipulation | `qty_vs_typical_ratio`, `cost_vs_awp_ratio`, `qty_days_supply_mismatch` |
| Collusion | `prescriber_top_pharmacy_pct`, `pharmacy_top_prescriber_pct`, `after_hours_ratio` |
| Doctor Shopping | `member_unique_prescribers_90d`, `member_unique_pharmacies_90d`, `member_controlled_pct` |
| Therapeutic Duplication | `member_duplicate_class_count`, `member_class_prescriber_count` |
| Phantom Billing | `drug_diagnosis_match_score`, `billing_date_spike_flag` |
| Upcoding | `cost_vs_awp_ratio`, `brand_generic_available_flag`, `dispensing_fee_percentile` |
| Refill Too Soon | `days_since_last_fill`, `refill_frequency`, `early_refill_count` |

---

## Anomaly Injection Process

1. **Normal claims generated first** — all claims start with `is_anomaly=False`
2. **Random selection** — target indices chosen (non-overlapping per type)
3. **Pattern-specific modifications** — fields modified to create the anomaly
4. **Labels set** — `is_anomaly=True`, `anomaly_type` = pattern name
5. **Validation** — FK checks, cost sanity, distribution checks

This labeled dataset enables both **supervised** (XGBoost with labels) and
**unsupervised** (Isolation Forest without labels) model training.
