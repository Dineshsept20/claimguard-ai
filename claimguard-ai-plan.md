# ClaimGuard AI — Complete Build Plan for Local Mac

## Project Overview

**What:** AI-powered pharmacy claims anomaly detection system that identifies fraud, waste, and abuse patterns in prescription drug claims.

**Why it matters:** Pharmacy fraud/waste/abuse costs the US healthcare system $100B+ annually. Every PBM (CVS Caremark, Express Scripts, OptumRx), health plan, and CMS needs this capability.

**Your unique angle:** You understand real pharmacy claims adjudication (RxClaim, FEP, EDF systems). Most ML engineers build generic fraud detection — you'll build one that understands pharmacy-specific patterns like therapeutic duplication, quantity manipulation, and prescriber-pharmacy collusion.

---

## Prerequisites — Mac Setup (Day 0)

### 1. Python Environment

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+ (if not already)
brew install python@3.11

# Create project directory
mkdir -p ~/Dinesh/claimguard-ai
cd ~/Dinesh/claimguard-ai

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install pandas numpy scikit-learn xgboost lightgbm
pip install shap matplotlib seaborn plotly
pip install streamlit
pip install faker
pip install sqlalchemy
pip install pytest
pip install jupyter
```

### 2. Project Structure

```
claimguard-ai/
├── README.md                    # Project case study (critical for portfolio)
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/                     # Generated synthetic data
│   ├── processed/               # Feature-engineered data
│   └── reference/               # Drug reference tables, NDC codes
├── src/
│   ├── __init__.py
│   ├── data_generator/          # Synthetic claims data engine
│   │   ├── __init__.py
│   │   ├── generator.py         # Main generator orchestrator
│   │   ├── entities.py          # Pharmacies, prescribers, patients
│   │   ├── claims.py            # Claim generation logic
│   │   ├── anomalies.py         # Anomaly injection patterns
│   │   └── reference_data.py    # Drug names, NDCs, pricing
│   ├── features/                # Feature engineering
│   │   ├── __init__.py
│   │   ├── claim_features.py    # Per-claim features
│   │   ├── prescriber_features.py  # Prescriber behavior profiles
│   │   ├── pharmacy_features.py    # Pharmacy behavior profiles
│   │   └── network_features.py     # Prescriber-pharmacy relationships
│   ├── models/                  # ML models
│   │   ├── __init__.py
│   │   ├── isolation_forest.py  # Unsupervised anomaly detection
│   │   ├── xgboost_model.py     # Supervised classification
│   │   └── ensemble.py          # Hybrid model combining both
│   ├── explainability/          # SHAP + business rules
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   └── rule_engine.py       # Domain-specific business rules
│   └── utils/
│       ├── __init__.py
│       └── metrics.py           # Custom evaluation metrics
├── app/
│   ├── streamlit_app.py         # Main dashboard
│   ├── pages/
│   │   ├── 01_overview.py       # Summary dashboard
│   │   ├── 02_claims_explorer.py # Drill into flagged claims
│   │   ├── 03_prescriber_profile.py # Prescriber risk scoring
│   │   ├── 04_model_performance.py  # Model metrics
│   │   └── 05_explainability.py     # SHAP visualizations
│   └── components/
│       └── charts.py            # Reusable chart components
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
├── tests/
│   ├── test_generator.py
│   ├── test_features.py
│   └── test_models.py
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    └── anomaly_patterns.md
```

### 3. Git Setup

```bash
cd ~/Dinesh/claimguard-ai
git init
echo "venv/\n__pycache__/\n*.pyc\ndata/raw/\n.DS_Store\n*.pkl" > .gitignore
git add .
git commit -m "Initial project structure"
```

---

## Week 1: Synthetic Data Generator (The Foundation)

> **Why this matters:** This component alone demonstrates deep domain knowledge. A realistic synthetic claims dataset is harder to build than the ML model — and interviewers know it.

### Day 1-2: Reference Data & Entity Generation

**File: `src/data_generator/reference_data.py`**

Build reference tables that mirror real pharmacy data:

```python
# Drug Reference Table
# Include: NDC (National Drug Code), drug_name, generic_name,
#          therapeutic_class (GPI code), DEA_schedule,
#          AWP (Average Wholesale Price), typical_days_supply,
#          typical_quantity, route_of_administration

# Example categories to include:
# - Opioids (Schedule II) — high fraud target
# - Specialty drugs (high cost) — waste/abuse target
# - Maintenance meds (statins, BP meds) — baseline normal claims
# - Controlled substances (benzos, stimulants) — abuse patterns
# - Injectables (Humira, Enbrel) — quantity manipulation target
```

**File: `src/data_generator/entities.py`**

```python
# Generate realistic entities:
# 1. Pharmacies (500-1000)
#    - pharmacy_id, name, npi, type (retail/mail/specialty/compounding)
#    - address, chain_flag, state
#    - Some pharmacies will be "bad actors" (5-10%)

# 2. Prescribers (2000-5000)
#    - prescriber_id, name, npi, specialty, DEA_number
#    - state, practice_type
#    - Some prescribers will have suspicious patterns (3-5%)

# 3. Members/Patients (10000-20000)
#    - member_id, age, gender, plan_type, state
#    - chronic_conditions (drives realistic prescribing)
#    - Some members will be "doctor shoppers" (2-3%)
```

### Day 3-4: Claims Generation Engine

**File: `src/data_generator/claims.py`**

Generate 500K-1M claims over a 12-month period:

```python
# Each claim should have:
# - claim_id, service_date, pharmacy_id, prescriber_id, member_id
# - ndc, drug_name, quantity, days_supply, refill_number
# - ingredient_cost, dispensing_fee, copay, plan_paid
# - diagnosis_code (ICD-10), prior_auth_flag
# - claim_status (paid/reversed/rejected)
# - submit_time (time of day — suspicious if always after hours)

# Normal patterns to encode:
# - Members with diabetes get metformin, insulin
# - Members with hypertension get ACE inhibitors, ARBs
# - Refill patterns follow 30/90 day cycles
# - Cost follows AWP with normal variation
# - Most claims during business hours
```

### Day 5-6: Anomaly Injection

**File: `src/data_generator/anomalies.py`**

This is where your domain knowledge shines. Inject realistic fraud patterns:

```python
# ANOMALY TYPE 1: Quantity Manipulation (15% of anomalies)
# - Dispensing 120 tablets when normal is 30
# - Days supply doesn't match quantity
# - Pattern: specific pharmacies doing this repeatedly

# ANOMALY TYPE 2: Prescriber-Pharmacy Collusion (20% of anomalies)
# - One prescriber sends 80%+ of scripts to one pharmacy
# - Both in different geographic areas (unusual)
# - High volume of controlled substances

# ANOMALY TYPE 3: Doctor Shopping (15% of anomalies)
# - Same member visiting 5+ prescribers for same drug class
# - Multiple pharmacies used to avoid detection
# - Focus on opioids and controlled substances

# ANOMALY TYPE 4: Therapeutic Duplication (10% of anomalies)
# - Same member getting two drugs from same therapeutic class
# - From different prescribers (who don't know about each other)

# ANOMALY TYPE 5: Phantom Billing (15% of anomalies)
# - Claims for members who don't have recent diagnosis
# - Unusual drug-diagnosis combinations
# - Billing spikes on specific dates

# ANOMALY TYPE 6: Upcoding/Price Manipulation (15% of anomalies)
# - Brand dispensed when generic available + cheaper
# - Ingredient cost significantly above AWP
# - Compounding pharmacy charges 10x normal

# ANOMALY TYPE 7: Refill Too Soon (10% of anomalies)
# - Refill before 75% of days supply consumed
# - Pattern of early refills for controlled substances
```

### Day 7: Data Validation & Documentation

```python
# Validate generated data:
# - Claim volumes follow realistic distribution (weekday > weekend)
# - Cost distributions are realistic
# - Anomaly rate is 3-5% of total claims (realistic)
# - All foreign keys are valid
# - No impossible combinations (e.g., pediatric patient on geriatric drug)

# Generate data_dictionary.md documenting every field
# Generate anomaly_patterns.md explaining each pattern

# Run generator and save to data/raw/
```

**Deliverable:** A synthetic data generator that produces realistic pharmacy claims with known anomalies. This becomes a reusable asset for Projects 2 and 3.

---

## Week 2: Feature Engineering & Model Training

### Day 8-9: Claim-Level Features

**File: `src/features/claim_features.py`**

```python
# Per-claim features:
# - cost_vs_awp_ratio: ingredient_cost / AWP (>1.5 = suspicious)
# - quantity_vs_typical: quantity / typical_quantity for that drug
# - days_supply_quantity_mismatch: flag if days_supply * daily_dose != quantity
# - refill_days_early: how many days before expected refill
# - is_controlled_substance: binary
# - is_specialty_drug: binary
# - submit_hour: hour of claim submission
# - is_weekend: binary
# - cost_percentile: where this claim falls in cost distribution for this drug
```

### Day 10-11: Entity-Level Features (Behavioral Profiles)

**File: `src/features/prescriber_features.py`**

```python
# Prescriber behavioral profile (rolling 90-day window):
# - total_claims_count
# - unique_patients_count
# - controlled_substance_ratio: % of claims that are controlled
# - unique_pharmacies_count
# - top_pharmacy_concentration: % of claims to most-used pharmacy
# - avg_cost_per_claim
# - specialty_match_ratio: % of drugs matching prescriber specialty
# - weekend_prescribing_ratio
# - opioid_mme_per_patient: morphine milligram equivalents (critical metric)
```

**File: `src/features/pharmacy_features.py`**

```python
# Pharmacy behavioral profile (rolling 90-day window):
# - total_claims_count
# - unique_prescribers_count
# - unique_patients_count
# - controlled_substance_ratio
# - avg_cost_per_claim vs peer pharmacies in same state
# - reversal_rate: % of claims later reversed
# - brand_when_generic_available_ratio
# - after_hours_claim_ratio
# - geographic_dispersion_of_patients: avg distance of patients
```

**File: `src/features/network_features.py`**

```python
# Relationship/network features:
# - prescriber_pharmacy_exclusivity: how concentrated is this relationship
# - patient_prescriber_count_for_drug_class: doctor shopping signal
# - patient_pharmacy_count_for_controlled: pharmacy shopping signal
# - prescriber_peer_cost_deviation: how different from peers in same specialty
# - pharmacy_peer_volume_deviation: volume vs similar pharmacies
```

### Day 12-13: Model Training

**File: `src/models/isolation_forest.py`** (Unsupervised)

```python
# Why Isolation Forest:
# - Works well for anomaly detection without labels
# - Handles high-dimensional data
# - Fast training on large datasets
# - Good baseline that doesn't need labeled anomalies

# Train on claim-level + entity-level features
# Tune contamination parameter (set to ~0.05 matching known anomaly rate)
# Evaluate on known anomalies from generator
```

**File: `src/models/xgboost_model.py`** (Supervised)

```python
# Why XGBoost:
# - Handles imbalanced data well (fraud is rare)
# - Feature importance built-in
# - Strong with tabular data
# - Industry standard for fraud detection

# Handle class imbalance:
# - scale_pos_weight parameter
# - SMOTE for training data
# - Stratified cross-validation

# Hyperparameter tuning:
# - Use Optuna or RandomizedSearchCV
# - Focus on: max_depth, learning_rate, n_estimators, min_child_weight
# - Optimize for F1-score (balance precision and recall)
```

**File: `src/models/ensemble.py`** (Hybrid)

```python
# Combine Isolation Forest + XGBoost + Business Rules:
# 1. Isolation Forest anomaly score (0-1)
# 2. XGBoost fraud probability (0-1)
# 3. Business rule flags (binary)
# 4. Weighted ensemble with configurable thresholds

# Business rules (from domain knowledge):
# - Opioid MME > 90 per patient per day = flag
# - Prescriber sending >60% to single pharmacy = flag
# - Days supply mismatch > 20% = flag
# - Cost > 3x AWP = flag
# - Patient visiting >3 prescribers for same drug class in 90 days = flag
```

### Day 14: Model Evaluation

```python
# Metrics to compute and display:
# - Precision, Recall, F1-Score (per anomaly type)
# - ROC-AUC curve
# - Precision-Recall curve (more useful for imbalanced data)
# - Confusion matrix
# - Detection rate per anomaly type

# Compare: Isolation Forest alone vs XGBoost alone vs Ensemble
# Document findings in notebook 03_model_training.ipynb
```

---

## Week 3: Explainability & Streamlit Dashboard

### Day 15-16: SHAP Explainability

**File: `src/explainability/shap_explainer.py`**

```python
# SHAP (SHapley Additive exPlanations):
# - Global feature importance: which features matter most overall
# - Local explanations: why THIS specific claim was flagged
# - Interaction effects: which feature combinations are suspicious

# Generate:
# 1. SHAP summary plot (global importance)
# 2. SHAP waterfall plot (per-claim explanation)
# 3. SHAP dependence plots (feature interactions)
# 4. Human-readable explanation text:
#    "This claim was flagged because:
#     - Quantity dispensed (360) is 4x the typical quantity (90)
#     - This prescriber sends 78% of scripts to this pharmacy
#     - The ingredient cost is 2.3x above AWP"
```

### Day 17-19: Streamlit Dashboard

**File: `app/streamlit_app.py`** — Main multi-page app

**Page 1: Executive Overview (`pages/01_overview.py`)**
```
- Total claims analyzed: 750,000
- Flagged claims: 28,500 (3.8%)
- Estimated savings: $4.2M
- Anomaly breakdown by type (pie chart)
- Trend over 12 months (line chart)
- Top 10 highest-risk pharmacies
- Top 10 highest-risk prescribers
```

**Page 2: Claims Explorer (`pages/02_claims_explorer.py`)**
```
- Filterable table of flagged claims
- Filters: date range, anomaly type, risk score, pharmacy, prescriber
- Click any claim → see SHAP explanation
- Export flagged claims to CSV
```

**Page 3: Prescriber Risk Profile (`pages/03_prescriber_profile.py`)**
```
- Search by prescriber NPI or name
- Risk score with breakdown
- Prescribing pattern visualizations
- Network graph: prescriber ↔ pharmacy relationships
- Peer comparison (how does this prescriber compare to similar ones)
```

**Page 4: Model Performance (`pages/04_model_performance.py`)**
```
- ROC curves for each model
- Precision-Recall curves
- Confusion matrices
- Detection rate by anomaly type
- Model comparison table
```

**Page 5: Explainability Deep Dive (`pages/05_explainability.py`)**
```
- Global SHAP feature importance
- Select any flagged claim → waterfall plot
- Natural language explanation of why claim was flagged
- "What-if" analysis: change a feature value and see how risk changes
```

### Day 20-21: Polish & Connect

```
- Connect all dashboard pages to the trained models
- Add caching (@st.cache_data) for performance
- Style the dashboard (professional color scheme, clear layout)
- Add sidebar navigation
- Test all pages end-to-end
```

---

## Week 4: Documentation, Testing & Portfolio Packaging

### Day 22-23: Testing

```python
# tests/test_generator.py
# - Generated data has correct schema
# - Anomaly rate is within expected range
# - No impossible data combinations
# - Foreign key integrity

# tests/test_features.py
# - Feature values are in expected ranges
# - No NaN values in critical features
# - Feature computation is deterministic

# tests/test_models.py
# - Model trains without errors
# - Predictions are in [0, 1] range
# - Model performance meets minimum thresholds
# - Ensemble beats individual models
```

### Day 24-25: README as Case Study

This is the **most important file in your repo**. It's what hiring managers read.

```markdown
# Structure your README.md as:

## 1. Problem Statement (3-4 sentences)
   - What is pharmacy claims fraud/waste/abuse
   - Why it matters ($100B+ impact)
   - Why current approaches fall short

## 2. Solution Architecture
   - Architecture diagram (use draw.io or Mermaid)
   - Data pipeline flow
   - Model ensemble approach
   - Explainability layer

## 3. Key Technical Decisions (show your thinking)
   - Why synthetic data (privacy + controlled anomalies)
   - Why hybrid model (unsupervised + supervised + rules)
   - Why SHAP for healthcare (explainability is non-negotiable)
   - Why Streamlit (rapid prototyping, easy demo)

## 4. Results
   - Model performance metrics with charts
   - Detection rates per anomaly type
   - Example SHAP explanations
   - Screenshot of dashboard

## 5. Domain Insights
   - What pharmacy fraud patterns look like
   - Why ML alone isn't enough (business rules matter)
   - How this would integrate into a real PBM workflow

## 6. Tech Stack
   - Full list with rationale for each choice

## 7. How to Run
   - One-command setup instructions
   - Demo walkthrough

## 8. Future Enhancements (shows vision)
   - Real-time streaming detection
   - Graph neural networks for network analysis
   - Integration with FormularyGPT for policy queries
   - Cloud deployment architecture (GCP Vertex AI)
```

### Day 26-27: Demo Video & LinkedIn Post

**5-minute demo video:**
```
- 0:00 — Problem introduction (30 sec)
- 0:30 — Architecture overview (60 sec)
- 1:30 — Live demo: Overview dashboard (60 sec)
- 2:30 — Live demo: Drill into a flagged claim, show SHAP explanation (90 sec)
- 4:00 — Model performance comparison (30 sec)
- 4:30 — Key learnings and what's next (30 sec)
```

Record with QuickTime (built into Mac) or OBS (free). Upload to YouTube (unlisted).

**LinkedIn Post:**
```
Title: "I Built an AI System to Detect Pharmacy Claims Fraud — Here's What I Learned"

Content:
- Hook: The $100B problem most people don't know about
- What you built (with screenshot)
- 3 surprising insights from the project
- Link to GitHub repo and demo video
- Tag relevant hashtags: #HealthcareAI #MachineLearning #PBM
```

### Day 28: Final Cleanup

```
- All code has docstrings
- requirements.txt is accurate (pip freeze > requirements.txt)
- .gitignore is complete
- All notebooks run end-to-end
- Dashboard starts with one command: streamlit run app/streamlit_app.py
- Push to GitHub with clean commit history
```

---

## Running Commands (Mac Quick Reference)

```bash
# Activate environment
cd ~/Dinesh/claimguard-ai
source venv/bin/activate

# Generate synthetic data
python -m src.data_generator.generator --num-claims 750000 --output data/raw/

# Run feature engineering
python -m src.features.claim_features
python -m src.features.prescriber_features
python -m src.features.pharmacy_features

# Train models
python -m src.models.isolation_forest
python -m src.models.xgboost_model
python -m src.models.ensemble

# Run tests
pytest tests/ -v

# Launch dashboard
streamlit run app/streamlit_app.py

# Jupyter notebooks (for exploration)
jupyter notebook notebooks/
```

---

## Interview-Ready Talking Points

After completing this project, you should be able to confidently answer:

**"Walk me through your approach to anomaly detection."**
> "I used a hybrid ensemble — Isolation Forest for unsupervised anomaly scoring, XGBoost for supervised classification, and domain-specific business rules. The ensemble outperformed any single approach by 15% on F1-score because each method catches different fraud patterns."

**"How did you handle the class imbalance?"**
> "Fraud is typically 3-5% of claims. I used SMOTE for training, scale_pos_weight in XGBoost, and optimized for F1/PR-AUC rather than accuracy. I also stratified cross-validation to ensure each fold had representative fraud cases."

**"Why is explainability important here?"**
> "In healthcare, a black-box model isn't deployable. Pharmacists and SIU investigators need to understand WHY a claim was flagged before taking action. I used SHAP values to generate both visual explanations and natural language summaries for each flagged claim."

**"How would you deploy this in production?"**
> "Batch scoring pipeline on GCP — Cloud Functions triggered by new claims data in BigQuery, model served via Vertex AI endpoints, results written to a dashboard in Looker. For real-time, I'd add Pub/Sub for streaming claims and serve the model with low-latency endpoints."

**"What would you do differently with real data?"**
> "Three things: First, I'd incorporate temporal patterns — fraud evolves, so models need retraining pipelines. Second, I'd add graph-based features using prescriber-pharmacy-patient networks. Third, I'd implement a human-in-the-loop feedback system where investigator decisions retrain the model."

---

## Timeline Summary

| Week | Focus | Deliverable |
|------|-------|------------|
| Week 1 | Synthetic Data Generator | 750K realistic claims with 7 anomaly types |
| Week 2 | Features + Model Training | Hybrid ensemble with >0.85 F1-score |
| Week 3 | SHAP + Streamlit Dashboard | 5-page interactive dashboard |
| Week 4 | Docs + Testing + Portfolio | README case study, demo video, LinkedIn post |

**Daily rhythm:** 5:00-5:30 AM — focused learning on that day's topic. 5:30-6:30 AM — hands-on coding. Commit and push every day.

---

## Cloud Deployment (After MVP — Phase 2)

Once the local MVP is solid, we'll deploy on GCP:
- **Data:** BigQuery for claims data warehouse
- **Model Serving:** Vertex AI Model Registry + Endpoints
- **Pipeline:** Cloud Composer (Airflow) for scheduled retraining
- **Dashboard:** Cloud Run for Streamlit OR migrate to Looker
- **Monitoring:** Vertex AI Model Monitoring for drift detection
- **CI/CD:** Cloud Build for automated testing and deployment

This will be planned in detail once the local MVP is complete and demo-ready.
