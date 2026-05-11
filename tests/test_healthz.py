"""healthz returns structured backend-readiness info; request_id propagates to logs."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from meno_rag.api.main import app

    with TestClient(app) as c:
        yield c


def test_healthz_returns_structured_status(client):
    response = client.get("/healthz")
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "rag_ready" in body
    assert "db" in body
    assert "redis" in body
    assert "embedder_device" in body


def test_request_id_header_is_echoed(client):
    response = client.get("/healthz", headers={"X-Request-Id": "test-req-id"})
    assert response.headers.get("x-request-id") == "test-req-id"


def test_request_id_is_generated_when_absent(client):
    response = client.get("/healthz")
    assert response.headers.get("x-request-id")
