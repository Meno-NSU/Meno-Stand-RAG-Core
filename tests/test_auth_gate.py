# tests/test_auth_gate.py
from __future__ import annotations

from meno_rag.api.auth import requires_auth_for_model


def test_openrouter_blocked_for_anonymous_when_enabled():
    assert requires_auth_for_model("openrouter", auth_enabled=True, authenticated=False) is True


def test_openrouter_allowed_for_authenticated():
    assert requires_auth_for_model("openrouter", auth_enabled=True, authenticated=True) is False


def test_vllm_always_allowed():
    assert requires_auth_for_model("vllm", auth_enabled=True, authenticated=False) is False


def test_no_gate_when_auth_disabled():
    assert requires_auth_for_model("openrouter", auth_enabled=False, authenticated=False) is False
