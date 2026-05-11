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
