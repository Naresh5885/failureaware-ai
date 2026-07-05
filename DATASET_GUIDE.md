# 📚 FailureAware AI — Comprehensive Dataset & Schema Guide

This document provides complete, research-oriented documentation for all datasets, knowledge indexes, clinical guidelines, and benchmark evaluation data utilized by the **FailureAware AI Architecture**.

---

## 1. 📋 Policy Data (`data/policy_plans.json` & Ingested Files)

### Overview
Stores multi-tier health insurance coverage limits, copays, deductibles, and coinsurance percentages enforced across all 8 AI agents.

### Policy Plan Schema
| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `policy_id` | String | Unique policy identification key | `POL987654321` |
| `plan_type` | String | Tier plan categorization | `Premium Plan` / `Basic Plan` / `Family Plan` |
| `coverage_limit` | Float | Maximum annual financial coverage cap (USD) | `$25,000.00` |
| `deductible` | Float | Individual annual deductible (USD) | `$250.00` |
| `copay` | Float | Standard copayment per claim (USD) | `$25.00` |
| `coinsurance_pct` | Float | Percentage covered by insurer after deductible | `90.0%` |

---

## 2. ⚕️ Medical Guidelines Dataset (`data/medical_guidelines.csv`)

### Overview
Contains **200 clinical mappings** linking patient diagnosis with valid procedures, allowable medications, and recommended clinical treatments. Used by `MedicalValidationAgent` to prevent erroneous or fraudulent treatment claims.

### Dataset Schema
| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `diagnosis` | String | Clinical diagnosis / medical condition | `Knee Osteoarthritis` |
| `valid_procedure` | String | Clinically justified medical procedure | `Outpatient Knee Arthroscopy` |
| `valid_medication` | String | Class of allowable pharmaceuticals | `NSAIDs / Hyaluronic Acid` |
| `recommended_medication` | String | Specific recommended drug dosage | `Ibuprofen 800mg` |

---

## 3. 🛡️ Historical Claims Fraud Dataset (`data/historical_claims.csv`)

### Overview
Contains **500 historical claim records** with risk indicators used by `FraudDetectionAgent` to perform:
1. Duplicate fingerprint checking
2. Unusual amount detection (2.5x variance above diagnosis mean)
3. Provider frequency analysis
4. Claim velocity analysis

### Dataset Schema
| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `claim_id` | String | Unique historical claim identifier | `CLM-1042` |
| `patient` | String | Patient / claimant full name | `Sarah Smith` |
| `diagnosis` | String | Primary medical diagnosis | `Severe Fever` |
| `procedure` | String | Billed medical procedure | `Heart Bypass Surgery` |
| `amount` | Float | Billed claim amount (USD) | `$15,000.00` |
| `verdict` | String | Historical adjudication outcome | `Flagged for Manual Review` |
| `fraud_flag` | String | Risk categorization | `HIGH` / `MEDIUM` / `LOW` |

---

## 4. 📁 Ingested Document Chunking (`500 / 100` Overlap)

### Semantic Vector Store Schema
All ingested policy documents, formularies, and guidelines are split into **500-character chunks with 100-character overlap** and stored in the Endee Vector Store database (`data/vector_store.json`).

| Chunk Metadata Field | Type | Description |
| :--- | :--- | :--- |
| `chunk_id` | String | Unique chunk identifier (`insurance_policy.pdf_chunk_1`) |
| `document_name` | String | Original filename (`insurance_policy.pdf`) |
| `chunk_text` | String | 500-character text segment |
| `embedding` | List[Float] | 384-dimensional dense vector (`sentence-transformers/all-MiniLM-L6-v2`) |
| `metadata` | Object | `{ policy_id, plan_type, coverage_limit, deductible, copay }` |

---

## 5. 🎯 Benchmark Evaluation Suite (300 Synthetic Cases)

### Benchmark Structure
* **100 Valid Claims**: Standard claims adhering to policy guidelines (ICU stays, knee arthroscopy, generic drugs).
* **100 Fraud Claims**: Duplicate filings, high velocity, unlisted patients, and unusual amounts.
* **100 Edge Cases**: Elective cosmetic exclusions (Rhinoplasty), clinical mismatches (Fever + Bypass), and Tier 3 specialty prior authorization rules.

### Performance Metrics
* **Accuracy**: `96.0%`
* **Precision**: `94.1%`
* **Recall**: `97.0%`
* **F1 Score**: `95.5%`
* **Confusion Matrix**: `TP: 191 | TN: 97 | FP: 6 | FN: 6`
