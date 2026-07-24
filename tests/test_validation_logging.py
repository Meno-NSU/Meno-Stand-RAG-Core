from unittest.mock import AsyncMock

from meno_rag.api.validation import sanitize_validation_errors


def test_sanitize_keeps_field_and_reason_strips_raw_input():
    # Shaped like a real Pydantic v2 error for /v1/arena/turn's 40k content cap: the
    # `input` (full answer text) and `ctx` must be dropped, `loc`/`msg`/`type` kept.
    raw = [
        {
            "type": "string_too_long",
            "loc": ("body", "sides", 0, "content"),
            "msg": "String should have at most 40000 characters",
            "input": "a very long generated answer " * 5000,
            "ctx": {"max_length": 40000},
            "url": "https://errors.pydantic.dev/2/v/string_too_long",
        },
    ]
    out = sanitize_validation_errors(raw)
    assert out == [
        {
            "loc": ("body", "sides", 0, "content"),
            "msg": "String should have at most 40000 characters",
            "type": "string_too_long",
        }
    ]
    assert "input" not in out[0]
    assert "ctx" not in out[0]


def test_sanitize_tolerates_missing_keys():
    assert sanitize_validation_errors([{}]) == [{"loc": None, "msg": None, "type": None}]


def _client():
    # Imported lazily: constructing the app + TestClient lifespan pulls the model stack
    # (torch/faiss), which segfaults on macOS — these run on Linux CI. The pure tests
    # above need none of that.
    from fastapi.testclient import TestClient

    from meno_rag.api.main import app

    return TestClient(app)


def test_arena_turn_invalid_body_gets_a_sanitized_422():
    with _client() as c:
        # Missing session_id, and sides has fewer than the required 2 → validation fails
        # before any handler logic. The response detail must be the trimmed shape, no `input`.
        resp = c.post("/v1/arena/turn", json={"question": "q", "sides": []})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail, "expected at least one validation error"
    for err in detail:
        assert set(err.keys()) <= {"loc", "msg", "type"}
        assert "input" not in err


def test_refresh_models_refreshes_vllm_and_openrouter_concurrently():
    from meno_rag.api.main import app

    with _client() as c:
        app.state.vllm_registry.list_models = AsyncMock(return_value=[])
        app.state.vllm_registry.refresh = AsyncMock(return_value=[])
        app.state.openrouter_registry = AsyncMock()
        app.state.openrouter_registry.list_models = AsyncMock(return_value=[])
        app.state.openrouter_registry.discover = AsyncMock(return_value=[])

        resp = c.post("/v1/models/refresh")

    assert resp.status_code == 200
    app.state.vllm_registry.refresh.assert_awaited_once()
    app.state.openrouter_registry.discover.assert_awaited_once()
