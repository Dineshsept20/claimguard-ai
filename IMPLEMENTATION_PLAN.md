# ClaimGuard AI — Step-by-Step Implementation Plan

> **Created:** 21 Feb 2026
> **Location:** `/Users/329228/Dinesh/claimguard-ai/`
> **Environment:** macOS, Conda available, Homebrew available
> **Status:** Awaiting Approval

---

## Analysis of Requirements

The project is an **AI-powered pharmacy claims anomaly detection system** with:
- Synthetic data generation (500K+ claims with 7 fraud patterns)
- Feature engineering (claim-level, prescriber, pharmacy, network)
- ML models (Isolation Forest + XGBoost + Rules-based Ensemble)
- SHAP explainability layer
- Streamlit multi-page dashboard (5 pages)
- Full test suite and documentation

### Changes Made to Original Plan

| Change | Detail |
|--------|--------|
| Path fix | All references updated from `~/projects/claimguard-ai` to `~/Dinesh/claimguard-ai` |

### Key Decisions for Local Mac Build

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python version | 3.11 via Conda | System has 3.9.6 (too old for some deps). Conda already installed |
| Environment | Conda env (not venv) | Conda already available, handles native deps (numpy, scipy) better on Mac |
| Data size | 500K claims (start) | Lighter for development, can scale to 750K once pipeline works |
| Dashboard | Streamlit (local) | No cloud needed, runs on localhost |

---

## PHASE 1: Environment and Project Scaffolding

**Estimated time:** ~15 minutes

**What we will do:**

1. Create a Conda environment `claimguard` with Python 3.11
2. Install all dependencies: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, streamlit, faker, plotly, matplotlib, seaborn, sqlalchemy, pytest, jupyter, imbalanced-learn (SMOTE)
3. Create the full project directory structure as specified in the plan
4. Initialize Git repo with proper `.gitignore`
5. Create `requirements.txt`, `setup.py`, and all `__init__.py` files

**Files created:**

- `.gitignore`
- `requirements.txt`
- `setup.py`
- `src/__init__.py`
- `src/data_generator/__init__.py`
- `src/features/__init__.py`
- `src/models/__init__.py`
- `src/explainability/__init__.py`
- `src/utils/__init__.py`
- `app/pages/` (empty directory)
- `app/components/` (empty directory)
- `data/raw/`, `data/processed/`, `data/reference/` (empty directories)
- `tests/` (empty directory)
- `notebooks/` (empty directory)
- `docs/` (empty directory)

**Approval needed:** YES / NO

---

## PHASE 2: Reference Data and Entity Generation (Synthetic Data Part 1)

**Estimated time:** ~30 minutes

**What we will do:**

1. Build `src/data_generator/reference_data.py` — Drug reference table with:
   - 50+ realistic drugs across categories (opioids, specialty, maintenance, controlled, injectables)
   - NDC codes, AWP pricing, typical quantities, therapeutic classes, DEA schedules
   - Drug-to-diagnosis mappings (ICD-10 codes)

2. Build `src/data_generator/entities.py` — Generate:
   - 500 pharmacies (5-10% flagged as suspicious) with NPI, type, address, chain flag
   - 2,000 prescribers (3-5% with suspicious profiles) with NPI, specialty, DEA number
   - 10,000 members/patients (2-3% doctor shoppers) with demographics, chronic conditions

3. Build `src/data_generator/claims.py` — Claims generation engine:
   - 500K claims over 12 months
   - Realistic distributions (weekday > weekend, business hours, seasonal)
   - Correct drug-diagnosis mappings and refill cycles (30/90 day)
   - Proper pricing based on AWP with normal variation

**Files created:**

- `src/data_generator/reference_data.py`
- `src/data_generator/entities.py`
- `src/data_generator/claims.py`

**Approval needed:** YES / NO

---

## PHASE 3: Anomaly Injection and Data Generator Orchestrator (Synthetic Data Part 2)

**Estimated time:** ~30 minutes

**What we will do:**

1. Build `src/data_generator/anomalies.py` — 7 anomaly types:
   - Quantity Manipulation (15% of anomalies)
   - Prescriber-Pharmacy Collusion (20% of anomalies)
   - Doctor Shopping (15% of anomalies)
   - Therapeutic Duplication (10% of anomalies)
   - Phantom Billing (15% of anomalies)
   - Upcoding/Price Manipulation (15% of anomalies)
   - Refill Too Soon (10% of anomalies)

2. Build `src/data_generator/generator.py` — Orchestrator that:
   - Generates entities, then normal claims, then injects anomalies
   - Validates data integrity (FK checks, realistic distributions)
   - Saves output to `data/raw/` as CSV files
   - Adds `is_anomaly` and `anomaly_type` labels
   - CLI interface: `python -m src.data_generator.generator --num-claims 500000`

3. Generate data and validate:
   - Run generator to produce 500K claims
   - Verify anomaly rate is approximately 3-5%
   - Verify no impossible data combinations

4. Create documentation:
   - `docs/data_dictionary.md` documenting every field
   - `docs/anomaly_patterns.md` explaining each fraud pattern

**Files created:**

- `src/data_generator/anomalies.py`
- `src/data_generator/generator.py`
- `data/raw/claims.csv`
- `data/raw/pharmacies.csv`
- `data/raw/prescribers.csv`
- `data/raw/members.csv`
- `docs/data_dictionary.md`
- `docs/anomaly_patterns.md`

**Approval needed:** YES / NO

---

## PHASE 4: Feature Engineering and Model Training

**Estimated time:** ~45 minutes

**What we will do:**

1. Build feature engineering pipeline:
   - `src/features/claim_features.py` — Per-claim features: cost_vs_awp_ratio, quantity_vs_typical, days_supply_mismatch, refill_days_early, submit_hour, is_weekend, cost_percentile
   - `src/features/prescriber_features.py` — 90-day rolling window profiles: total claims, unique patients, controlled substance ratio, top pharmacy concentration, opioid MME per patient
   - `src/features/pharmacy_features.py` — Pharmacy profiles: reversal rate, brand_when_generic_ratio, after_hours_ratio, geographic dispersion
   - `src/features/network_features.py` — Relationship features: prescriber-pharmacy exclusivity, doctor shopping signals, peer deviation scores

2. Build ML models:
   - `src/models/isolation_forest.py` — Unsupervised anomaly detection baseline
   - `src/models/xgboost_model.py` — Supervised classification with SMOTE and scale_pos_weight
   - `src/models/ensemble.py` — Hybrid: Isolation Forest score + XGBoost probability + business rules

3. Build supporting modules:
   - `src/utils/metrics.py` — Custom metrics: per-anomaly-type detection rate, PR-AUC
   - `src/explainability/rule_engine.py` — Domain business rules (opioid MME > 90, prescriber concentration > 60%, etc.)

4. Train all models and save artifacts to `models/` directory
5. Save feature-engineered data to `data/processed/`

**Files created:**

- `src/features/claim_features.py`
- `src/features/prescriber_features.py`
- `src/features/pharmacy_features.py`
- `src/features/network_features.py`
- `src/models/isolation_forest.py`
- `src/models/xgboost_model.py`
- `src/models/ensemble.py`
- `src/utils/metrics.py`
- `src/explainability/rule_engine.py`
- `models/` (saved model artifacts: .pkl files)
- `data/processed/` (feature-engineered CSVs)

**Approval needed:** YES / NO

---

## PHASE 5: SHAP Explainability and Streamlit Dashboard

**Estimated time:** ~45 minutes

**What we will do:**

1. Build `src/explainability/shap_explainer.py`:
   - Global SHAP feature importance
   - Per-claim waterfall explanations
   - Natural language explanation generator (human-readable text for each flagged claim)
   - Dependence plots for feature interactions

2. Build Streamlit multi-page dashboard:
   - `app/streamlit_app.py` — Main app with sidebar navigation and professional styling
   - `app/pages/01_overview.py` — Executive summary: total claims, flagged count, estimated savings, anomaly breakdown pie chart, trend line chart, top 10 risky pharmacies and prescribers
   - `app/pages/02_claims_explorer.py` — Filterable table of flagged claims with drill-down, filters by date/anomaly type/risk score/pharmacy/prescriber, CSV export
   - `app/pages/03_prescriber_profile.py` — Search by NPI/name, risk score breakdown, prescribing pattern visualizations, peer comparison
   - `app/pages/04_model_performance.py` — ROC curves, PR curves, confusion matrices, detection rate by anomaly type, model comparison table
   - `app/pages/05_explainability.py` — Global SHAP importance, per-claim waterfall, natural language explanations, what-if analysis

3. Build `app/components/charts.py` — Reusable Plotly chart components

4. Wire everything: dashboard loads saved models, scores claims, shows explanations

**Files created:**

- `src/explainability/shap_explainer.py`
- `app/streamlit_app.py`
- `app/pages/01_overview.py`
- `app/pages/02_claims_explorer.py`
- `app/pages/03_prescriber_profile.py`
- `app/pages/04_model_performance.py`
- `app/pages/05_explainability.py`
- `app/components/charts.py`

**Approval needed:** YES / NO

---

## PHASE 6: Testing and Notebooks

**Estimated time:** ~30 minutes

**What we will do:**

1. Build test suite:
   - `tests/test_generator.py` — Data schema validation, anomaly rate checks, FK integrity, no impossible combinations
   - `tests/test_features.py` — Feature value ranges, no NaN values in critical features, deterministic computation
   - `tests/test_models.py` — Model trains without errors, predictions in [0,1], minimum performance thresholds, ensemble beats individuals

2. Create Jupyter notebooks:
   - `notebooks/01_data_exploration.ipynb` — EDA on generated claims data
   - `notebooks/02_feature_engineering.ipynb` — Feature analysis and distribution plots
   - `notebooks/03_model_training.ipynb` — Model comparison and evaluation
   - `notebooks/04_shap_analysis.ipynb` — SHAP deep-dive visualizations

3. Run all tests with `pytest tests/ -v` and verify passing

**Files created:**

- `tests/test_generator.py`
- `tests/test_features.py`
- `tests/test_models.py`
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_feature_engineering.ipynb`
- `notebooks/03_model_training.ipynb`
- `notebooks/04_shap_analysis.ipynb`

**Approval needed:** YES / NO

---

## PHASE 7: Documentation and Final Polish

**Estimated time:** ~20 minutes

**What we will do:**

1. Create comprehensive `README.md` in portfolio case study format:
   - Problem statement (pharmacy fraud costs $100B+ annually)
   - Solution architecture with Mermaid diagram
   - Key technical decisions with rationale
   - Results with metrics and charts
   - Domain insights
   - Tech stack
   - How to run (one-command setup)
   - Future enhancements

2. Create `docs/architecture.md` with Mermaid diagrams showing data flow and system architecture

3. Add docstrings to all Python modules

4. Final `requirements.txt` freeze with exact versions

5. Clean Git history with meaningful commits

6. End-to-end validation:
   - Data generation runs successfully
   - All tests pass with pytest
   - Dashboard starts with `streamlit run app/streamlit_app.py`

**Files created/updated:**

- `README.md`
- `docs/architecture.md`

**Approval needed:** YES / NO

---

## Quick-Start Commands (After All Phases Complete)

```bash
# Navigate to project
cd ~/Dinesh/claimguard-ai

# Activate environment
conda activate claimguard

# Generate synthetic data (500K claims)
python -m src.data_generator.generator --num-claims 500000 --output data/raw/

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
```

---

## Tech Stack Summary

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Environment | Conda |
| Data Processing | pandas, numpy |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Synthetic Data | Faker |
| Testing | pytest |
| Notebooks | Jupyter |
| Version Control | Git |

---

## Important Notes

- All runs are local on your Mac — no cloud services needed for MVP
- Data stays local — synthetic data only, no real PHI/PII
- Each phase is independent — we can pause and resume between phases
- I will ask for your approval before starting each phase
- Cloud deployment (GCP) is planned as a future phase after MVP is complete

---

## Ready to Start?

Please review this plan and reply with:
- **Approve all** — I will execute phases sequentially, pausing for approval between each
- **Approve Phase 1** — I will start with environment setup only
- **Changes needed** — Tell me what to adjust
