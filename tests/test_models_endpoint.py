from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_or(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    from meno_rag.config import get_settings

    get_settings.cache_clear()
    from meno_rag.api.main import app

    with TestClient(app) as c:
        # Inject mocked registries to avoid real network calls
        app.state.vllm_registry.list_models = AsyncMock(
            return_value=[
                {"id": "menon-1", "endpoint": "http://v", "created": 100, "object": "model", "owned_by": "vllm"}
            ]
        )
        app.state.openrouter_registry = AsyncMock()
        app.state.openrouter_registry.list_models = AsyncMock(
            return_value=[
                {
                    "id": "d/c:free",
                    "display_name": "DeepSeek V3 (free)",
                    "context_length": 65536,
                    "featured": True,
                    "provider": "openrouter",
                }
            ]
        )
        yield c


def test_models_endpoint_returns_provider_and_status_for_each(client_with_or):
    r = client_with_or.get("/v1/models")
    body = r.json()
    by_id = {m["id"]: m for m in body["data"]}

    vllm_record = by_id["menon-1"]
    assert vllm_record["provider"] == "vllm"
    assert vllm_record["stages"] == ["rewrite", "rerank", "generation"]
    assert vllm_record["status"]["state"] == "available"

    or_record = by_id["d/c:free"]
    assert or_record["provider"] == "openrouter"
    assert or_record["stages"] == ["generation"]
    assert or_record["featured"] is True
    assert or_record["display_name"] == "DeepSeek V3 (free)"


def test_models_endpoint_returns_core_model_id(client_with_or):
    r = client_with_or.get("/v1/models")
    body = r.json()
    assert body["core_model_id"] == "menon-1"


def test_an_openrouter_entry_is_a_menon_10_model_named_by_its_display_name(client_with_or):
    entry = {m["id"]: m for m in client_with_or.get("/v1/models").json()["data"]}["d/c:free"]
    assert entry["stand_id"] == "menon-1.0"
    assert entry["llm_id"] == entry["id"]
    assert entry["llm_name"] == entry["display_name"]
    assert entry["stand_name"] == "Менон 1.0"


def test_a_model_whose_endpoint_is_down_is_listed_as_unreachable(client_with_or):
    from meno_rag.api.main import app

    app.state.vllm_registry.unavailable_models = lambda: [
        {"id": "meno-lite", "endpoint": "http://down", "created": 100, "object": "model", "owned_by": "vllm"}
    ]
    by_id = {m["id"]: m for m in client_with_or.get("/v1/models").json()["data"]}
    assert by_id["menon-1"]["status"]["state"] == "available"
    assert by_id["meno-lite"]["status"]["state"] == "unreachable"
    assert by_id["meno-lite"]["provider"] == "vllm"


def test_a_stale_model_kept_for_routing_is_not_presented_as_available(client_with_or):
    from meno_rag.api.main import app

    app.state.vllm_registry.unavailable_models = lambda: [
        {"id": "menon-1", "endpoint": "http://v", "created": 100, "object": "model", "owned_by": "vllm"}
    ]
    entries = [m for m in client_with_or.get("/v1/models").json()["data"] if m["id"] == "menon-1"]
    assert len(entries) == 1
    assert entries[0]["status"]["state"] == "unreachable"
