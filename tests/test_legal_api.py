from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meno_rag.api import legal
from meno_rag.config import Settings


@pytest.fixture
def client():
    app = FastAPI()
    app.state.settings = Settings()
    app.include_router(legal.router)
    with TestClient(app) as c:
        yield c


def test_lists_the_three_documents_with_hashes(client):
    body = client.get("/v1/legal/documents").json()
    docs = {d["kind"]: d for d in body["documents"]}
    assert set(docs) == {"privacy_policy", "personal_data_consent", "terms_of_use"}
    assert docs["privacy_policy"]["url"] == "/privacy"
    assert docs["personal_data_consent"]["url"] == "/consent"
    assert docs["terms_of_use"]["url"] == "/terms"
    for d in docs.values():
        assert d["version"] == "2.0"
        assert len(d["sha256"]) == 64  # hex SHA-256


def test_returns_document_content_and_matching_hash(client):
    listed = {d["kind"]: d for d in client.get("/v1/legal/documents").json()["documents"]}
    r = client.get("/v1/legal/documents/privacy_policy")
    assert r.status_code == 200
    doc = r.json()
    assert "Политика обработки персональных данных" in doc["content"]
    assert doc["sha256"] == listed["privacy_policy"]["sha256"]


def test_unknown_document_is_404(client):
    assert client.get("/v1/legal/documents/nope").status_code == 404
