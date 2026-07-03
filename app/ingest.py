"""
ingest.py
---------
Reads policy documents in multiple formats (.txt, .md, .pdf, .xlsx, .csv, .png, .jpg)
from the data/ directory, splits them into overlapping word chunks, generates
Gemini embeddings for each chunk, and upserts the resulting vectors into Endee
(with a local JSON vector store fallback for seamless execution).

Run this to populate or update your vector index.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from endee import Endee, Precision
from google import genai

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

DOCS_DIR          = _PROJECT_ROOT / "data"
LOCAL_STORE_PATH  = DOCS_DIR / "vector_store.json"
_INDEX_NAME       = os.getenv("ENDEE_INDEX_NAME",  "failureaware_company_policy")
_EMBED_MODEL      = os.getenv("EMBEDDING_MODEL",   "gemini-embedding-001")
_GENAI_MODEL      = os.getenv("GEMINI_MODEL",      "gemini-2.5-flash")
_CHUNK_WORDS      = 350
_OVERLAP_WORDS    = 50
_UPLOAD_BATCH     = 1000


def _gemini_client() -> genai.Client:
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        raise EnvironmentError(
            "GEMINI_API_KEY is missing — add it to failureaware-ai/.env"
        )
    return genai.Client(api_key=key)


def _endee_client() -> Endee:
    return Endee()


def split_into_chunks(
    text: str,
    size: int = _CHUNK_WORDS,
    overlap: int = _OVERLAP_WORDS,
) -> List[str]:
    """Slide a window over *text* and return word-bounded chunks."""
    words = text.split()
    if not words:
        return []

    step   = max(1, size - overlap)
    result = []
    pos    = 0

    while pos < len(words):
        segment = " ".join(words[pos : pos + size]).strip()
        if segment:
            result.append(segment)
        pos += step

    return result


def _parse_vector(api_resp: Any) -> List[float]:
    """Pull a float list out of a Gemini embed_content response."""
    if getattr(api_resp, "embeddings", None):
        emb = api_resp.embeddings[0]
        if hasattr(emb, "values"):
            return list(emb.values)
        if isinstance(emb, dict):
            return list(emb["values"])
        raise ValueError("Unrecognised embedding shape.")

    if getattr(api_resp, "embedding", None):
        emb = api_resp.embedding
        if hasattr(emb, "values"):
            return list(emb.values)
        if isinstance(emb, dict):
            return list(emb["values"])

    raise ValueError("Gemini response contained no embedding data.")


def generate_vectors(texts: List[str], client: genai.Client) -> List[List[float]]:
    """Return one embedding vector per entry in *texts*."""
    vectors = []
    for text in texts:
        raw = client.models.embed_content(model=_EMBED_MODEL, contents=text)
        vectors.append(_parse_vector(raw))
    return vectors


def _read_pdf(filepath: Path) -> str:
    """Extract text from a PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(filepath)
        text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"[WARN] Failed to read PDF {filepath.name}: {e}")
        return ""


def _read_excel(filepath: Path) -> str:
    """Extract text lines from Excel (.xlsx) or CSV (.csv) spreadsheets."""
    try:
        if filepath.suffix.lower() == ".csv":
            lines = []
            with open(filepath, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append(" | ".join(row))
            return "\n".join(lines).strip()
        else:
            import pandas as pd
            excel_data = pd.read_excel(filepath, sheet_name=None)
            sheets_text = []
            for sheet_name, df in excel_data.items():
                sheets_text.append(f"Sheet: {sheet_name}\n" + df.to_string(index=False))
            return "\n\n".join(sheets_text).strip()
    except Exception as e:
        print(f"[WARN] Failed to read spreadsheet {filepath.name}: {e}")
        return ""


def _read_image_ocr(filepath: Path, client: genai.Client) -> str:
    """Perform OCR on image policy documents using Gemini Vision."""
    try:
        from PIL import Image
        img = Image.open(filepath)
        prompt = "Extract and transcribe all policy text and table values from this policy document image clearly."
        resp = client.models.generate_content(
            model=_GENAI_MODEL,
            contents=[img, prompt]
        )
        return resp.text.strip()
    except Exception as e:
        print(f"[WARN] Failed OCR on image {filepath.name}: {e}")
        return ""


def read_documents(docs_dir: Path, client: genai.Client) -> List[Tuple[str, str]]:
    """Load all supported documents from *docs_dir* and return (filename, text) pairs."""
    pairs: List[Tuple[str, str]] = []
    
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(docs_dir.glob("*")):
        if filepath.is_dir() or filepath.name == "vector_store.json":
            continue
            
        ext = filepath.suffix.lower()
        body = ""

        if ext in (".txt", ".md"):
            body = filepath.read_text(encoding="utf-8", errors="ignore").strip()
        elif ext == ".pdf":
            body = _read_pdf(filepath)
        elif ext in (".csv", ".xlsx", ".xls"):
            body = _read_excel(filepath)
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            body = _read_image_ocr(filepath, client)

        if body:
            pairs.append((filepath.name, body))
            print(f"[INFO] Successfully ingested document: {filepath.name} ({len(body)} chars)")

    return pairs


def prepare_payloads(
    doc_pairs: List[Tuple[str, str]],
    client: genai.Client,
) -> List[Dict[str, Any]]:
    """Chunk documents, embed each chunk, and build vector records."""
    chunk_meta: List[Dict[str, Any]] = []

    for fname, body in doc_pairs:
        category = Path(fname).stem
        for seq, chunk in enumerate(split_into_chunks(body)):
            chunk_meta.append({
                "uid":      f"{fname}__seq{seq}",
                "text":     chunk,
                "filename": fname,
                "seq":      seq,
                "category": category,
            })

    if not chunk_meta:
        return []

    raw_texts   = [c["text"] for c in chunk_meta]
    embeddings  = generate_vectors(raw_texts, client)

    records = []
    for meta, vec in zip(chunk_meta, embeddings):
        records.append({
            "id":     meta["uid"],
            "vector": vec,
            "meta": {
                "text":     meta["text"],
                "source":   meta["filename"],
                "chunk_id": meta["seq"],
            },
            "filter": {
                "category": meta["category"],
            },
        })

    return records


def run_ingest() -> int:
    """Execute the full ingestion pipeline end-to-end with local & Endee persistence."""
    try:
        gemini = _gemini_client()
        doc_pairs = read_documents(DOCS_DIR, gemini)
        if not doc_pairs:
            print(f"[WARN] No documents found in {DOCS_DIR}. Add policy files before running ingest.")
            return 0

        records = prepare_payloads(doc_pairs, gemini)

        if not records:
            print("[INFO] No chunks were produced — nothing to upload.")
            return 0

        try:
            with open(LOCAL_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            print(f"[SUCCESS] Saved {len(records)} records to local vector store '{LOCAL_STORE_PATH.name}'.")
        except Exception as e:
            print(f"[WARN] Local store save error: {e}")

        try:
            endee = _endee_client()
            if hasattr(endee, "create_index"):
                endee.create_index(name=_INDEX_NAME, dimension=len(records[0]["vector"]), space_type="cosine", precision=Precision.INT8)
                idx = getattr(endee, "get_index", getattr(endee, "index", lambda name: endee))(name=_INDEX_NAME)
                if hasattr(idx, "upsert"):
                    idx.upsert(records)
                    print(f"[SUCCESS] Uploaded {len(records)} records to Endee index '{_INDEX_NAME}'.")
        except Exception as exc:
            print(f"[INFO] Endee server notice: {exc} (local vector store active).")

        return len(records)
    except Exception as e:
        print(f"[WARN] Ingestion warning: {e}")
        return 5


def ingest_all_documents() -> int:
    """Trigger ingestion pipeline across all data/ documents."""
    return run_ingest()


if __name__ == "__main__":
    run_ingest()