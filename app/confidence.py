"""
confidence.py
-------------
Evaluates how trustworthy the retrieved search results are before
forming an answer. Three tiers are used: High, Medium, and Low.
If nothing was retrieved at all, a special "Insufficient evidence"
state is returned so the caller can handle it gracefully.
"""

from __future__ import annotations

from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# Thresholds — adjust these to tune how strict the confidence gate is
# --------------------------------------------------------------------------- #
HIGH_TOP   = 0.80   # minimum top-1 score for High confidence
HIGH_AVG   = 0.75   # minimum avg-of-top-3 for High confidence
MED_TOP    = 0.65   # minimum top-1 score for Medium confidence
MED_AVG    = 0.60   # minimum avg-of-top-3 for Medium confidence


def _extract_score(record: Dict[str, Any]) -> float:
    """Pull a numeric relevance score out of a result record.

    Endee can surface the score under different keys depending on the
    query mode, so we check both 'similarity' and 'score'.
    """
    for key in ("similarity", "score"):
        value = record.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _make_result(level: str, explanation: str, best: float, avg: float) -> Dict[str, Any]:
    """Bundle a confidence verdict into a consistent dict."""
    return {
        "confidence": level,
        "reason":     explanation,
        "top_score":  round(best, 4),
        "avg_top3":   round(avg, 4),
    }


def evaluate_retrieval_confidence(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide how much to trust a list of retrieved chunks.

    Parameters
    ----------
    hits : list of result dicts returned by the vector index

    Returns
    -------
    dict with keys: confidence, reason, top_score, avg_top3
    """
    if not hits:
        return _make_result(
            "Insufficient evidence",
            "The vector search returned no matching documents.",
            0.0,
            0.0,
        )

    all_scores  = [_extract_score(h) for h in hits]
    best_score  = all_scores[0]
    sample      = all_scores[:3]
    mean_top3   = sum(sample) / len(sample)

    if best_score >= HIGH_TOP and mean_top3 >= HIGH_AVG:
        return _make_result(
            "High",
            "The top results align closely with the query — strong evidence found.",
            best_score,
            mean_top3,
        )

    if best_score >= MED_TOP and mean_top3 >= MED_AVG:
        return _make_result(
            "Medium",
            "Partial matches were found, but the overall signal is not conclusive.",
            best_score,
            mean_top3,
        )

    return _make_result(
        "Low",
        "Similarity scores are below acceptable thresholds — answer may be unreliable.",
        best_score,
        mean_top3,
    )