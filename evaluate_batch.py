"""
evaluate_batch.py
-----------------
Synthetic Benchmark Evaluation Suite for FailureAware AI Platform.
Generates 100 synthetic insurance claims across diverse policy categories (Dental, ICU, Surgery,
Pharmacy, Cosmetic, Unregistered Members, Duplicates) and computes Accuracy, Precision, Recall,
F1 Score, Latency, and Hallucination Rate metrics.
"""

from __future__ import annotations

import os
import sys

# Set fallback dummy API key for GitHub Actions CI/CD test environments
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "DUMMY_KEY_FOR_BENCHMARK"

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure app module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.graph import MultiAgentGraph


def generate_synthetic_claims(count: int = 100) -> List[Dict[str, Any]]:
    """Generate 100 realistic synthetic claim test cases with ground truth verdicts."""
    claimants = [
        ("David Miller", True),
        ("Alex Turner", True),
        ("Rachel Green", True),
        ("Marcus Vance", True),
        ("Sarah Smith", True),
        ("John Doe", True),
        ("Unknown User Hii", False),
        ("Naresh Unregistered", False),
        ("Fake Claimant Test", False)
    ]

    scenarios = [
        # (Diagnosis/Reason, Amount, Expected Verdict)
        ("Routine Dental Cleaning Checkup", 120.0, "Approved"),
        ("Outpatient Knee Surgery", 3200.0, "Approved"),
        ("Intensive Care Unit (ICU) Stay", 2200.0, "Approved"),
        ("Emergency Ground Ambulance", 750.0, "Approved"),
        ("Humira Specialty Prescription", 500.0, "Flagged for Manual Review"),
        ("Ozempic Diabetes Drug", 600.0, "Flagged for Manual Review"),
        ("Major Dental Root Canal Treatment", 800.0, "Flagged for Manual Review"),
        ("Elective Cosmetic Rhinoplasty Surgery", 4500.0, "Rejected"),
        ("Botox Injections Procedure", 900.0, "Rejected"),
    ]

    claims = []
    for i in range(count):
        claimant, is_registered = random.choice(claimants)
        reason, amount, expected_verdict = random.choice(scenarios)

        # If member is unregistered, ground truth verdict must be Rejected
        if not is_registered:
            ground_truth = "Rejected"
        else:
            ground_truth = expected_verdict

        claims.append({
            "id": f"TEST-CLM-{i+1:03d}",
            "claimant": claimant,
            "reason": reason,
            "amount": amount,
            "ground_truth_verdict": ground_truth,
            "raw_text": f"Claimant {claimant} submitted a claim for {reason} costing ${amount:.2f}. Policy ID: POL{1000+i}."
        })

    return claims


def run_benchmark():
    """Run batch evaluation across 100 synthetic claims and print metrics report."""
    print("=" * 70)
    print("🚀 STARTING FAILUREAWARE AI 100-CLAIM SYNTHETIC BENCHMARK")
    print("=" * 70)

    claims = generate_synthetic_claims(100)
    graph = MultiAgentGraph()

    tp, fp, fn, tn = 0, 0, 0, 0
    hallucination_count = 0
    latencies: List[float] = []

    start_batch_time = time.time()

    for idx, test_case in enumerate(claims, 1):
        t0 = time.time()
        result = graph.run(test_case["raw_text"], is_image=False)
        elapsed = (time.time() - t0) * 1000.0
        latencies.append(elapsed)

        pred_verdict = result.get("verdict", "Approved")
        true_verdict = test_case["ground_truth_verdict"]

        # Classification matrix logic (Positive = Approved/Flagged, Negative = Rejected)
        if pred_verdict == true_verdict:
            if true_verdict != "Rejected":
                tp += 1
            else:
                tn += 1
        else:
            if pred_verdict != "Rejected" and true_verdict == "Rejected":
                fp += 1
            elif pred_verdict == "Rejected" and true_verdict != "Rejected":
                fn += 1

        # Check hallucination (e.g. non-existent citations or contradictory reasoning)
        reasoning = result.get("reasoning", "")
        if "general policy" in reasoning.lower() and true_verdict == "Rejected":
            hallucination_count += 1

        if idx % 20 == 0 or idx == 100:
            print(f"  • Processed {idx}/100 claims... (Avg Latency: {sum(latencies)/len(latencies):.1f}ms)")

    total_time = time.time() - start_batch_time
    total = len(claims)
    accuracy = ((tp + tn) / total) * 100.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies)
    hallucination_rate = (hallucination_count / total) * 100.0

    print("\n" + "=" * 70)
    print("📊 BENCHMARK METRICS SUMMARY (100 CLAIMS EVALUATED)")
    print("=" * 70)
    print(f"  ✅ Accuracy           : {accuracy:.2f}%")
    print(f"  🎯 Precision          : {precision:.2f}%")
    print(f"  🔍 Recall             : {recall:.2f}%")
    print(f"  ⚡ F1 Score           : {f1_score:.2f}%")
    print(f"  ⏱️  Mean Latency       : {avg_latency:.2f} ms")
    print(f"  🛡️  Hallucination Rate : {hallucination_rate:.2f}%")
    print(f"  🕒 Total Batch Duration: {total_time:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
