# ClaimGuard AI — System Architecture

## High-Level Architecture

```mermaid
graph LR
    subgraph "Data Layer"
        A1[Reference Data<br/>53 drugs · 27 classes]
        A2[Entity Generator<br/>pharmacies · prescribers · members]
        A3[Claims Engine<br/>500K claims · 12 months]
        A4[Anomaly Injector<br/>7 fraud patterns · 4% rate]
    end

    subgraph "Feature Layer"
        B1[Claim Features<br/>14 features]
        B2[Prescriber Features<br/>11 features]
        B3[Pharmacy Features<br/>13 features]
        B4[Network Features<br/>11 features]
    end

    subgraph "Model Layer"
        C1[Isolation Forest<br/>Unsupervised · 25%]
        C2[XGBoost + SMOTE<br/>Supervised · 55%]
        C3[Rule Engine<br/>10 rules · 20%]
        C4[Hybrid Ensemble<br/>Weighted fusion]
    end

    subgraph "Explainability Layer"
        D1[SHAP TreeExplainer<br/>Global importance]
        D2[Per-Claim Waterfall<br/>Local explanations]
        D3[NL Generator<br/>Human-readable text]
    end

    subgraph "Presentation Layer"
        E1[Streamlit Dashboard<br/>5 interactive pages]
        E2[Jupyter Notebooks<br/>4 analysis notebooks]
    end

    A1 --> A3
    A2 --> A3
    A3 --> A4
    A4 --> B1 & B2 & B3 & B4
    B1 & B2 & B3 & B4 --> C1 & C2 & C3
    C1 & C2 & C3 --> C4
    C4 --> D1 & D2 & D3
    D1 & D2 & D3 --> E1
    C4 --> E1
    C4 --> E2
```

---

## Data Generation Pipeline

```mermaid
sequenceDiagram
    participant G as generator.py
    participant R as reference_data.py
    participant E as entities.py
    participant C as claims.py
    participant A as anomalies.py

    G->>R: Load DRUG_REFERENCE (53 drugs)
    G->>E: generate_pharmacies(500)
    G->>E: generate_prescribers(2000)
    G->>E: generate_members(10000)
    G->>C: generate_claims(500K)
    Note over C: Pre-computed lookups<br/>drug_class → prescriber map<br/>pharmacy arrays by type
    C-->>G: baseline claims DataFrame
    G->>A: inject_anomalies(claims, rate=0.04)
    Note over A: 7 patterns: quantity_manipulation,<br/>prescriber_pharmacy_collusion,<br/>doctor_shopping, therapeutic_duplication,<br/>phantom_billing, upcoding, refill_too_soon
    A-->>G: claims with is_anomaly labels
    G->>G: Save CSVs to data/raw/
```

---

## Feature Engineering Pipeline

```mermaid
flowchart TB
    RAW[Raw Claims CSV<br/>500K × 27 columns] --> CF[Claim Features<br/>14 features per claim]
    RAW --> PF[Prescriber Features<br/>11 prescriber profiles]
    RAW --> PHF[Pharmacy Features<br/>13 pharmacy profiles]
    RAW --> NF[Network Features<br/>11 relationship features]

    CF --> MERGE[Merge All Features<br/>via left join on claim_id]
    PF --> MERGE
    PHF --> MERGE
    NF --> MERGE

    MERGE --> RE[Rule Engine<br/>10 business rules → rule_flags_count]
    RE --> OUT[claims_features.csv<br/>500K × 98 columns]

    subgraph "Per-Claim Features"
        CF1[cost_vs_awp_ratio]
        CF2[quantity_vs_typical]
        CF3[is_controlled / is_specialty]
        CF4[is_weekend / is_after_hours]
        CF5[cost_percentile]
        CF6[opioid_mme_daily]
    end

    subgraph "Behavioral Profiles"
        PF1[presc_controlled_ratio]
        PF2[presc_cost_peer_zscore]
        PHF1[pharm_controlled_ratio]
        PHF2[pharm_cost_peer_zscore]
    end

    subgraph "Network Signals"
        NF1[pair_presc_exclusivity]
        NF2[doctor_shopping_signal]
        NF3[pharmacy_shopping_signal]
    end
```

---

## Model Training Pipeline

```mermaid
flowchart LR
    subgraph "Step 1: Load"
        L[Load raw CSVs<br/>+ reference data]
    end

    subgraph "Step 2: Features"
        F[Build 49+ features<br/>claim + prescriber +<br/>pharmacy + network + rules]
    end

    subgraph "Step 3: Train IF"
        IF[Isolation Forest<br/>n_estimators=200<br/>contamination=0.04]
    end

    subgraph "Step 4: Train XGB"
        XGB[XGBoost Classifier<br/>SMOTE oversampling<br/>scale_pos_weight=24]
    end

    subgraph "Step 5: Ensemble"
        ENS[Weighted fusion<br/>IF: 0.25 · XGB: 0.55<br/>Rules: 0.20]
    end

    subgraph "Step 6: Save"
        S[Save models + scores<br/>to models/ and data/processed/]
    end

    L --> F --> IF & XGB
    IF & XGB --> ENS --> S
```

---

## Ensemble Scoring

```mermaid
flowchart TD
    CLAIM[Input Claim<br/>49+ features] --> IF_SCORE[IF Anomaly Score<br/>range: 0.0 - 1.0]
    CLAIM --> XGB_PROB[XGB Probability<br/>range: 0.0 - 1.0]
    CLAIM --> RULE_SCORE[Rule Score<br/>flags / 10]

    IF_SCORE --> |× 0.25| WEIGHTED
    XGB_PROB --> |× 0.55| WEIGHTED
    RULE_SCORE --> |× 0.20| WEIGHTED

    WEIGHTED[Weighted Sum] --> THRESHOLD{Score ≥ 0.5?}
    THRESHOLD --> |Yes| FLAGGED[🚨 Flagged as Anomaly]
    THRESHOLD --> |No| NORMAL[✅ Normal Claim]

    FLAGGED --> TIER{Risk Tier}
    TIER --> |≥ 0.8| CRIT[🔴 Critical]
    TIER --> |≥ 0.6| HIGH[🟠 High]
    TIER --> |≥ 0.4| MED[🟡 Medium]
    TIER --> |< 0.4| LOW[🟢 Low]
```

---

## Dashboard Architecture

```mermaid
graph TD
    APP[streamlit_app.py<br/>Main entry point] --> NAV[Sidebar Navigation<br/>Radio button routing]

    NAV --> P1[p01_overview.py<br/>Executive KPIs]
    NAV --> P2[p02_claims_explorer.py<br/>Filterable claims table]
    NAV --> P3[p03_prescriber_profile.py<br/>Prescriber risk analysis]
    NAV --> P4[p04_model_performance.py<br/>ROC/PR curves]
    NAV --> P5[p05_explainability.py<br/>SHAP explanations]

    DL[data_loader.py<br/>@st.cache_data] --> P1 & P2 & P3 & P4 & P5
    CH[charts.py<br/>14 Plotly components] --> P1 & P2 & P3 & P4 & P5

    subgraph "Cached Data Sources"
        D1[claims_scored.csv]
        D2[claims_features.csv]
        D3[pharmacies.csv]
        D4[prescribers.csv]
        D5[xgboost_model.pkl]
        D6[shap_values.pkl]
    end

    DL --> D1 & D2 & D3 & D4 & D5 & D6
```

---

## SHAP Explainability Flow

```mermaid
sequenceDiagram
    participant U as User / SIU Investigator
    participant D as Dashboard
    participant S as shap_explainer.py
    participant M as XGBoost Model

    U->>D: Select claim to investigate
    D->>S: get_claim_explanation(claim_id)
    S->>M: TreeExplainer.shap_values()
    M-->>S: SHAP values array (36 features)
    S->>S: Sort by |SHAP value|
    S->>S: generate_natural_language_explanation()
    S-->>D: Top factors + NL text
    D-->>U: Waterfall chart + explanation text

    Note over U,D: "This claim was flagged because<br/>the prescriber has 87% exclusivity<br/>with one pharmacy and the quantity<br/>is 8.5x typical for this drug."
```

---

## Data Model (Entity Relationships)

```mermaid
erDiagram
    MEMBERS {
        string member_id PK
        string first_name
        string last_name
        int age
        string gender
        string state
        string plan_type
        string chronic_conditions
        int num_conditions
        bool is_doctor_shopper
    }

    PRESCRIBERS {
        string prescriber_id PK
        string full_name
        string npi
        string dea_number
        string specialty
        string state
        string practice_type
        bool is_suspicious
    }

    PHARMACIES {
        string pharmacy_id PK
        string pharmacy_name
        string npi
        string pharmacy_type
        string state
        bool chain_flag
        bool is_suspicious
    }

    CLAIMS {
        string claim_id PK
        date service_date
        string member_id FK
        string prescriber_id FK
        string pharmacy_id FK
        string ndc
        string drug_name
        float quantity
        int days_supply
        float total_cost
        bool is_anomaly
        string anomaly_type
    }

    MEMBERS ||--o{ CLAIMS : "submits"
    PRESCRIBERS ||--o{ CLAIMS : "prescribes"
    PHARMACIES ||--o{ CLAIMS : "dispenses"
```
