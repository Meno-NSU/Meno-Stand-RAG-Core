"""Published legal documents: metadata + content + SHA-256, read from legal/*.ru.md.

No DB table by design — version/URL are declared here, the hash is computed from the
file, and the effective date comes from config. `GET /v1/legal/documents` feeds the
consent flow; `GET /v1/legal/documents/{kind}` returns the text so the frontend / nginx
can render /privacy, /consent, /terms.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

# legal/ lives at the repo root: src/meno_rag/api/legal.py → parents[3].
_LEGAL_DIR = Path(__file__).resolve().parents[3] / "legal"

LEGAL_DOCUMENTS: dict[str, dict[str, str]] = {
    "privacy_policy": {"file": "privacy-policy.ru.md", "url": "/privacy", "version": "2.0"},
    "personal_data_consent": {"file": "personal-data-consent.ru.md", "url": "/consent", "version": "2.0"},
    "terms_of_use": {"file": "terms-of-use.ru.md", "url": "/terms", "version": "2.0"},
}


def load_document_text(kind: str) -> str:
    return (_LEGAL_DIR / LEGAL_DOCUMENTS[kind]["file"]).read_text(encoding="utf-8")


def document_sha256(kind: str) -> str:
    return hashlib.sha256(load_document_text(kind).encode("utf-8")).hexdigest()


def _document_meta(kind: str, effective_at: str | None) -> dict:
    meta = LEGAL_DOCUMENTS[kind]
    return {
        "kind": kind,
        "version": meta["version"],
        "url": meta["url"],
        "sha256": document_sha256(kind),
        "effective_at": effective_at,
    }


router = APIRouter(prefix="/v1/legal", tags=["legal"])


def _effective_at(request: Request) -> str | None:
    return request.app.state.settings.legal_effective_at or None


@router.get("/documents")
async def get_documents(request: Request):
    effective_at = _effective_at(request)
    return {"documents": [_document_meta(kind, effective_at) for kind in LEGAL_DOCUMENTS]}


@router.get("/documents/{kind}")
async def get_document(kind: str, request: Request):
    if kind not in LEGAL_DOCUMENTS:
        raise HTTPException(status_code=404, detail="Unknown document.")
    return {**_document_meta(kind, _effective_at(request)), "content": load_document_text(kind)}
