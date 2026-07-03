# FailureAware AI — Multimodal Insurance Claims & Eligibility Gater

> **A Multi-Agent, Confidence-Gated Multimodal RAG System for Insurance Claim Verification — powered by Endee Vector DB, Gemini 2.5 Flash, Gemini Embeddings, and Python.**

---

## 🏥 Business Use Case

Insurance processing companies evaluate thousands of claim forms, medical invoices, and reimbursement requests every day. Manually cross-referencing each claim against complex policy limits, deductibles, pre-existing condition waiting periods, and exclusion clauses requires significant time and human effort.

**FailureAware AI** automates this process using a **Multi-Agent RAG Pipeline**:
1. **Multimodal Ingestion:** Ingests insurance guidelines and limit tables from `.txt`, `.pdf`, `.csv`, `.xlsx`, and scanned `.png`/`.jpg` policy charts into a local **Endee Vector Database**.
2. **Multimodal Claims Processing:** Accepts claims in natural text or scanned image form (e.g., photos of medical invoices).
3. **Multi-Agent Verification:** Coordinates four specialized agents to parse claims, search policy rules, logic-check coverage, and issue automated verdicts (**Approved**, **Rejected**, or **Flagged for Manual Review**) with direct text citations.
4. **Safety & Confidence Gate:** If retrieved policy context is weak or ambiguous, the system automatically flags the claim for manual human review rather than making unsafe assumptions.

---

## 🤖 Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FailureAware AI — Agent Pipeline                      │
│                                                                             │
│   Inbound Claim (Text / Image / PDF / Invoice)                              │
│         │                                                                   │
│         ▼                                                                   │
│   ┌────────────────────────┐                                                │
│   │   Claim Parser Agent   │ ──► Extracts: Claimant, Diagnosis, Amount      │
│   └───────────┬────────────┘                                                │
│               │                                                             │
│               ▼                                                             │
│   ┌────────────────────────┐       Embed Query      ┌───────────────────┐   │
│   │     Router Agent       │ ─────────────────────► │ Gemini Embeddings │   │
│   └───────────┬────────────┘                        └─────────┬─────────┘   │
│               │                                               │             │
│               ▼                                               ▼             │
│   ┌────────────────────────┐                        ┌───────────────────┐   │
│   │    Retrieval Agent     │ ─────────────────────► │  Endee Vector DB  │   │
│   └───────────┬────────────┘                        └─────────┬─────────┘   │
│               │                                               │             │
│               ▼                                               ▼             │
│   ┌────────────────────────┐                        ┌───────────────────┐   │
│   │ Eligibility Critic     │ ◄───────────────────── │ Top-k Policy Rules│   │
│   │        Agent           │                        └───────────────────┘   │
│   └───────────┬────────────┘                                                │
│               │                                                             │
│               ▼                                                             │
│      [ Confidence Check ] ── Score ≥ Threshold? ──────────┐                 │
│               │                                           │                 │
│           YES │                                        NO │                 │
│               ▼                                           ▼                 │
│   ┌────────────────────────┐                 ┌──────────────────────────┐   │
│   │    Decision Agent      │                 │    Decision Agent        │   │
│   │ (Approved / Rejected)  │                 │(Flagged for Manual Review)│  │
│   └────────────────────────┘                 └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Project Structure

```
failureaware-ai/
├── app/
│   ├── agents.py        # Multi-Agent suite (Parser, Router, Eligibility Critic, Decision)
│   ├── confidence.py    # Similarity threshold evaluator & score grading
│   ├── ingest.py        # Multi-format document ingestion (.txt, .pdf, .csv, .xlsx, .png, .jpg)
│   ├── retrieve.py      # Vector search & multi-agent pipeline coordinator
│   └── main.py          # Interactive CLI REPL with agent reasoning traces
├── data/
│   ├── insurance_policy.txt  # Comprehensive health & emergency policy rules
│   ├── coverage_limits.csv   # Structured limit schedules & copay tables
│   ├── leave_policy.txt      # HR Leave guidelines
│   ├── onboarding.txt        # Onboarding rules
│   ├── reimbursement.txt    # Expense rules
│   └── work_from_home.txt    # WFH rules
├── .env                 # API Keys & Endee settings
├── requirements.txt     # Python dependencies
└── README.md            # System documentation
```

---

## 🚀 Setup & Execution Instructions

### 1. Prerequisites
- Python 3.10+
- [Google AI Studio](https://aistudio.google.com/) Gemini API Key
- [Endee Vector Database](https://github.com/endee-io/endee) running in Docker:
  ```bash
  docker run --ulimit nofile=100000:100000 -p 8080:8080 -v ./endee-data:/data --name endee-server --restart unless-stopped endeeio/endee-server:latest
  ```

### 2. Configure Environment
Set your Gemini key in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=gemini-embedding-001
ENDEE_INDEX_NAME=failureaware_company_policy
TOP_K=5
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ingest Policy Documents into Endee
```bash
python app/ingest.py
```

### 5. Launch the Claim Verification Demo
```bash
python app/main.py
```

---

## 📋 Sample Test Claims

**1. Approved Claim (Within Policy Limits):**
> *"Claimant John Doe submitted a claim for routine dental cleaning costing $100."*
> → **Verdict:** `APPROVED` ($100 approved, 100% coverage per Section 2.1).

**2. Partially Approved / Capped Claim:**
> *"Claimant Sarah Smith submitted a claim for major root canal dental treatment costing $850."*
> → **Verdict:** `APPROVED ($500)` (Capped at maximum payout limit of $500 per major dental procedure per Section 2.2).

**3. Rejected / Excluded Claim:**
> *"Claimant Alex Taylor submitted a claim for experimental cosmetic surgery costing $3,000."*
> → **Verdict:** `REJECTED` (Experimental & cosmetic surgeries excluded per Section 4.2 & 4.3).

**4. Flagged for Manual Review (Off-Topic / Low Confidence):**
> *"Claimant submitted a claim for European real estate tax."*
> → **Verdict:** `FLAGGED FOR MANUAL REVIEW` (Low confidence score — unsafe to auto-verify).
