"""
app/fraud.py
------------
Fraud Detection Agent for FailureAware AI Platform.
Analyzes claims for duplicates, missing required metadata, abnormal financial ratios,
and policy abuse anomalies. Returns LOW, MEDIUM, or HIGH risk classification.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

# Global in-memory claim registry for duplicate detection
_PROCESSED_CLAIM_FINGERPRINTS: Set[str] = set()


class FraudAgent:
    """Agent specialized in Fraud, Waste, and Abuse (FWA) detection."""

    def analyze_claim_risk(
        self,
        claim_data: Dict[str, Any],
        critic_eval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive risk scoring across multiple fraud vectors."""
        name = str(claim_data.get("claimant_name", "")).strip().lower()
        reason = str(claim_data.get("claim_reason", "")).strip().lower()
        claimed_amt = float(claim_data.get("claimed_amount", 0.0))
        policy_id = str(claim_data.get("policy_id", "")).strip()
        verdict = critic_eval.get("verdict", "Approved")

        risk_flags: List[str] = []
        risk_score = 0.0

        # 1. Duplicate Claim Fingerprint Check
        fingerprint = f"{name}___{reason}___{claimed_amt:.2f}"
        if fingerprint in _PROCESSED_CLAIM_FINGERPRINTS:
            risk_flags.append("Duplicate claim filing detected for identical claimant and amount.")
            risk_score += 0.45
        else:
            _PROCESSED_CLAIM_FINGERPRINTS.add(fingerprint)

        # 2. Missing or Suspicious Metadata Check
        if not policy_id or policy_id in ("N/A", "Unknown", "POL123456789"):
            risk_flags.append("Unverified or generic Policy ID provided.")
            risk_score += 0.20

        if name in ("unknown", "hii", "test", "fake", "none"):
            risk_flags.append("Unregistered or synthetic claimant name detected.")
            risk_score += 0.40

        # 3. Financial Outlier / Excessive Amount Check
        if claimed_amt > 10000.0:
            risk_flags.append("Claimed amount exceeds $10,000 threshold requiring senior audit.")
            risk_score += 0.25

        # 4. Policy Rejection / High Risk Verdict Linkage
        if verdict == "Rejected":
            risk_flags.append("Critic Agent issued policy rejection.")
            risk_score += 0.50
        elif verdict == "Flagged for Manual Review":
            risk_flags.append("Specialty drug or prior authorization condition matched.")
            risk_score += 0.30

        # Risk Classification
        if risk_score >= 0.40 or verdict == "Rejected":
            risk_level = "HIGH"
        elif risk_score >= 0.20 or verdict == "Flagged for Manual Review":
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_score": min(1.0, round(risk_score, 2)),
            "risk_flags": risk_flags,
            "is_duplicate": "Duplicate claim" in " ".join(risk_flags)
        }
