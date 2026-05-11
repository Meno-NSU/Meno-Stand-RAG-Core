from datetime import datetime, timedelta, timezone
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
        # Mock registries
        app.state.vllm_registry.list_models = AsyncMock(return_value=[{"id": "menon-1", "endpoint": "http://v"}])
        app.state.vllm_registry.resolve_model = AsyncMock(return_value=("menon-1", "http://v/v1"))
        app.state.openrouter_registry = AsyncMock()
        app.state.openrouter_registry.list_models = AsyncMock(
            return_value=[{"id": "d/c:free", "provider": "openrouter", "featured": True}]
        )

        # Mock pipeline (required for chat_completions endpoint)
        app.state.pipeline = AsyncMock()
        yield c


def test_pre_flight_429_when_or_model_is_rate_limited(client_with_or):
    import asyncio

    until = datetime.now(timezone.utc) + timedelta(minutes=5)
    asyncio.get_event_loop().run_until_complete(
        client_with_or.app.state.model_status_store.mark_rate_limited(
            "d/c:free", until=until, error="rate_limit_exceeded"
        )
    )
    r = client_with_or.post(
        "/v1/chat/completions", json={"model": "d/c:free", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["code"] == "model_rate_limited"
    assert body["error"]["until"]


def test_503_when_or_model_is_unreachable(client_with_or):
    import asyncio

    asyncio.get_event_loop().run_until_complete(
        client_with_or.app.state.model_status_store.mark_unreachable("d/c:free", error="conn_error")
    )
    r = client_with_or.post(
        "/v1/chat/completions", json={"model": "d/c:free", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "model_unreachable"
