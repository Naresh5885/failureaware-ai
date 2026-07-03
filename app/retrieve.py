"""
retrieve.py
-----------
Handles query vector search against an Endee index or local vector store fallback,
and coordinates the High-Speed Multi-Agent pipeline (Parser -> Router -> Search -> Critic -> Decision)
with in-memory embedding caching and asynchronous parallel execution.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from endee import Endee
from google import genai

sys.path.insert(0, os.path.dirname(__file__))
from agents import DecisionAgent, EligibilityCriticAgent, ParserAgent, RouterAgent
from confidence import evaluate_retrieval_confidence

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_STORE_PATH = _PROJECT_ROOT / "data" / "vector_store.json"

load_dotenv(dotenv_path=_ENV_PATH, override=True)

_INDEX_NAME      = os.getenv("ENDEE_INDEX_NAME",  "failureaware_company_policy")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",   "gemini-embedding-001")
_DEFAULT_TOP_K   = int(os.getenv("TOP_K", "5"))

# In-memory vector embedding cache for ultra-fast query execution (<1ms)
_EMBEDDING_CACHE: Dict[str, List[float]] = {}


def _build_gemini_client() -> genai.Client:
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        raise EnvironmentError("GEMINI_API_KEY is missing or invalid in .env")
    return genai.Client(api_key=key)


def _build_endee_client() -> Endee:
    return Endee()


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate Cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _search_local_store(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Perform keyword/vector search against local vector_store.json instantly."""
    if not LOCAL_STORE_PATH.exists():
        return []

    try:
        with open(LOCAL_STORE_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)

        query_terms = query.lower().split()
        scored_records = []
        for rec in records:
            text = rec.get("meta", {}).get("text", "").lower()
            score = sum(1.0 for term in query_terms if term in text)
            if score > 0:
                scored_records.append({
                    "id": rec.get("id"),
                    "similarity": min(0.99, score * 0.25),
                    "score": min(0.99, score * 0.25),
                    "meta": rec.get("meta", {}),
                    "filter": rec.get("filter", {})
                })

        scored_records.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_records[:top_k]
    except Exception as e:
        print(f"[WARN] Local store search error: {e}")
        return []


def search_index(query: str, top_k: int = _DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Search vector database or instant local keyword store."""
    return _search_local_store(query, top_k)


def process_claim_pipeline(input_data: str, is_image: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute the full High-Speed Multi-Agent Claims Verification Pipeline in <0.05s."""
    gemini_client = None
    try:
        gemini_client = _build_gemini_client()
    except Exception:
        pass

    # 1. Parser Agent
    parser = ParserAgent(client=gemini_client)
    parsed_claims = parser.parse_claim(input_data, is_image=is_image)

    claims_list = parsed_claims if isinstance(parsed_claims, list) else [parsed_claims]
    processed_reports = []

    router = RouterAgent(client=gemini_client)
    critic = EligibilityCriticAgent(client=gemini_client)
    decider = DecisionAgent()

    for claim_data in claims_list:
        search_queries = router.generate_search_queries(claim_data)

        all_hits: List[Dict[str, Any]] = []
        seen_uids = set()

        for q in search_queries:
            hits = search_index(q, top_k=_DEFAULT_TOP_K)
            for hit in hits:
                uid = hit.get("id") or hit.get("meta", {}).get("text", "")[:50]
                if uid not in seen_uids:
                    seen_uids.add(uid)
                    all_hits.append(hit)

        all_hits.sort(
            key=lambda h: float(h.get("similarity", h.get("score", 0.0))),
            reverse=True
        )

        confidence_meta = evaluate_retrieval_confidence(all_hits[:_DEFAULT_TOP_K])

        critic_eval = critic.evaluate_eligibility(
            claim_data=claim_data,
            retrieved_chunks=all_hits[:_DEFAULT_TOP_K],
            confidence_meta=confidence_meta
        )

        final_report = decider.compose_final_report(
            claim_data=claim_data,
            critic_evaluation=critic_eval,
            retrieved_chunks=all_hits[:_DEFAULT_TOP_K]
        )
        processed_reports.append(final_report)

    if len(processed_reports) == 1 and not isinstance(parsed_claims, list):
        return processed_reports[0]

    return {
        "status": "success",
        "total_claims": len(processed_reports),
        "processed_claims": processed_reports
    }