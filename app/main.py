"""
main.py
-------
Entry point for the FailureAware Multimodal Insurance Eligibility Checker.
Accepts text claim descriptions or image file paths, executes the Multi-Agent
RAG pipeline, and displays step-by-step reasoning traces and final verdicts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from retrieve import process_claim_pipeline

_LINE_WIDTH = 75


def _print_report(report: dict) -> None:
    """Print a formatted insurance eligibility report to stdout."""
    border = "=" * _LINE_WIDTH
    print(f"\n{border}")
    print("  AUTOMATED INSURANCE ELIGIBILITY REPORT")
    print(border)
    print(f"Claimant Name   : {report.get('claimant', 'Unknown')}")
    print(f"Claim Reason    : {report.get('claim_reason', 'N/A')}")
    print(f"Claimed Amount  : ${report.get('claimed_amount', 0.0):,.2f}")
    
    verdict = report.get('verdict', 'Flagged for Manual Review')
    print(f"\nVERDICT         : [ {verdict.upper()} ]")
    print(f"Approved Amount : ${report.get('approved_amount', 0.0):,.2f}")
    print(f"Confidence Tier : {report.get('confidence_tier', 'N/A')}")
    print(f"\nReasoning       :\n{report.get('reasoning', 'N/A')}")

    print("\nPolicy Citations & Matching Chunks:")
    citations = report.get("citations", [])
    if not citations:
        print("  (No relevant policy citations retrieved)")
    else:
        for idx, cit in enumerate(citations, start=1):
            print(f"  {idx}. [Score: {cit['similarity_score']:.4f}] {cit['source']}")
            print(f"     \"{cit['snippet']}\"\n")
            
    print(f"{border}\n")


def _interactive_loop() -> None:
    """Run an interactive REPL for testing claims."""
    print("=" * _LINE_WIDTH)
    print("  FailureAware AI — Multimodal Insurance Eligibility Gater")
    print("  Enter a claim description OR an image file path (e.g. claims/invoice.png)")
    print("  Type 'exit' to quit.")
    print("=" * _LINE_WIDTH + "\n")

    while True:
        try:
            raw = input("Enter Claim or Image Path: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Insurance Verification System. Goodbye!")
            break

        if not raw:
            print("  ⚠  Please enter a claim description or file path first.\n")
            continue

        if raw.lower() in {"exit", "quit", "q"}:
            print("Session ended.")
            break

        is_image = False
        if (raw.endswith(".png") or raw.endswith(".jpg") or raw.endswith(".jpeg")) and os.path.exists(raw):
            is_image = True
            print(f"\n[Parser Agent] Detected image claim file: {raw}")
        else:
            print(f"\n[Parser Agent] Processing text claim input...")

        try:
            print("[Router Agent] Formulating vector search queries...")
            print("[Retrieval Agent] Querying Endee vector index...")
            print("[Critic Agent] Evaluating policy rules vs claim details...")
            print("[Decision Agent] Generating verdict report...\n")

            report = process_claim_pipeline(raw, is_image=is_image)
            _print_report(report)
        except Exception as err:
            print(f"\n  [ERROR] {err}\n")


if __name__ == "__main__":
    _interactive_loop()