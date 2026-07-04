# FailureAware AI — Multi-Agent Insurance Verification & Anti-Fraud Platform

FailureAware AI is an enterprise-grade Multi-Agent Insurance Claim Verification and Fraud Detection Platform engineered with Python, FastAPI, Gemini 2.5 Flash, Endee Vector Database, and Agentic RAG.

## 🚀 Key Features

- **Multi-Agent Architecture**: Parser Agent, Router Agent, Eligibility Critic Agent, Fraud Detection Agent, and Decision Synthesizer Agent.
- **Sub-5ms Execution Engine**: Fast deterministic policy checks with full multi-row CSV/Excel batch support.
- **10 Real-World Insurance Policy Rules**: Automated coverage caps for Routine Dental, ICU stays, Outpatient Surgery, Tier 1-3 Pharmacy Formularies, Emergency Ambulances, and Cosmetic Exclusions.
- **Anti-Fraud & Member Registry**: Server-side duplicate claim fingerprint tracking and member enrollment database checks.
- **Glassmorphism React SaaS Interface**: Single-Page Application with drag-and-drop file dropzone, live log console, KPI summary cards, and CSV export.
- **100-Claim Synthetic Benchmark**: Integrated evaluation suite (`evaluate_batch.py`) computing Accuracy, Precision, Recall, F1 Score, Latency, and Hallucination metrics.
- **Production Containerization**: Multi-stage Dockerfile, docker-compose orchestration, and GitHub Actions CI/CD pipeline.

## 💻 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application Server
```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to: **http://localhost:8000**

### 3. Run Synthetic Benchmark Evaluation
```bash
python evaluate_batch.py
```

## 🌐 Production Cloud URL
- **Live SaaS App**: [https://failureaware-ai.onrender.com](https://failureaware-ai.onrender.com)
- **GitHub Repository**: [https://github.com/Naresh5885/failureaware-ai](https://github.com/Naresh5885/failureaware-ai)
