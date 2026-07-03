"""
api.py
------
FastAPI Application Entry Point for FailureAware AI Platform.
Exposes REST endpoints for claims verification (text & multimodal invoice files),
knowledge base ingestion, and synthetic benchmark evaluation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_STATIC_DIR = _APP_DIR / "static"
_DATA_DIR = _PROJECT_ROOT / "data"

sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from app.retrieve import process_claim_pipeline, search_index
from app.ingest import ingest_all_documents
from app.graph import MultiAgentGraph

app = FastAPI(
    title="FailureAware AI — Insurance Verification Platform",
    description="Multi-Agent Multimodal Claim Verification & Anti-Fraud Engine",
    version="2.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


class TextClaimRequest(BaseModel):
    claim_text: str = Field(..., description="Raw text describing claim details")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page React SaaS dashboard."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post("/api/verify-claim/text")
async def verify_text_claim(req: TextClaimRequest):
    """Verify single claim text via LangGraph Multi-Agent Engine."""
    if not req.claim_text.strip():
        raise HTTPException(status_code=400, detail="claim_text cannot be empty")

    graph = MultiAgentGraph()
    report = graph.run(req.claim_text, is_image=False)
    return JSONResponse(status_code=200, content=report)


@app.post("/api/verify-claim/image")
async def verify_image_claim(file: UploadFile = File(...)):
    """Verify single uploaded invoice file (PDF, PNG, JPG, CSV, XLSX)."""
    allowed_exts = {".pdf", ".xlsx", ".xls", ".csv", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"File format {ext} not supported.")

    temp_path = _DATA_DIR / f"temp_{file.filename}"
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        graph = MultiAgentGraph()
        report = graph.run(str(temp_path), is_image=True)

        if isinstance(report, dict) and "processed_claims" in report:
            for item in report["processed_claims"]:
                item["filename"] = file.filename
            return JSONResponse(status_code=200, content=report)
        else:
            report["filename"] = file.filename
            return JSONResponse(status_code=200, content=report)
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/verify-claims/batch")
async def verify_batch_claims(files: List[UploadFile] = File(...)):
    """Verify multiple uploaded invoice files in batch."""
    processed = []
    for file in files:
        temp_path = _DATA_DIR / f"temp_batch_{file.filename}"
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)

            graph = MultiAgentGraph()
            report = graph.run(str(temp_path), is_image=True)
            if isinstance(report, dict) and "processed_claims" in report:
                for item in report["processed_claims"]:
                    item["filename"] = file.filename
                    processed.append(item)
            else:
                report["filename"] = file.filename
                processed.append(report)
        finally:
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    return JSONResponse(status_code=200, content={
        "status": "success",
        "total_claims": len(processed),
        "processed_claims": processed
    })


@app.post("/api/ingest")
async def trigger_ingestion():
    """Trigger vector ingestion across all policy documents into Endee DB."""
    try:
        count = ingest_all_documents()
        return JSONResponse(status_code=200, content={"status": "success", "indexed_chunks": count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
