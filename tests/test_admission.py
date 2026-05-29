"""Admission control: bound concurrent chat requests, fast-fail beyond the cap."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meno_rag.api.admission import AdmissionController


def test_try_acquire_until_max_then_rejects():
    ac = AdmissionController(2)
    assert ac.try_acquire() is True
    assert ac.try_acquire() is True
    assert ac.try_acquire() is False
    assert ac.active == 2


def test_release_frees_a_slot():
    ac = AdmissionController(1)
    assert ac.try_acquire() is True
    assert ac.try_acquire() is False
    ac.release()
    assert ac.try_acquire() is True


def test_zero_max_disables_limit():
    ac = AdmissionController(0)
    for _ in range(500):
        assert ac.try_acquire() is True


def test_release_never_goes_negative():
    ac = AdmissionController(1)
    ac.release()
    assert ac.active == 0


def test_chat_endpoint_returns_503_overloaded_when_saturated():
    from meno_rag.api import main as main_mod

    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = object()  # non-None so the readiness check passes
        saturated = AdmissionController(1)
        assert saturated.try_acquire() is True  # fill the only slot
        c.app.state.admission = saturated
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "q"}]},
        )
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "overloaded"
    assert r.headers.get("Retry-After")


@pytest.mark.asyncio
async def test_lifespan_installs_admission_controller():
    from meno_rag.api.main import app

    with TestClient(app):
        assert isinstance(app.state.admission, AdmissionController)
