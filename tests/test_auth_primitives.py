# tests/test_auth_primitives.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from meno_rag.api.auth import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_roundtrip():
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h)
    assert not verify_password("wrong", h)


def test_verify_password_bad_hash_is_false():
    assert verify_password("x", "not-a-bcrypt-hash") is False


def test_token_roundtrip():
    token = create_access_token("u1", secret="s", ttl_hours=1)
    assert decode_access_token(token, secret="s") == "u1"


def test_token_wrong_secret_or_tampered_returns_none():
    token = create_access_token("u1", secret="s", ttl_hours=1)
    assert decode_access_token(token, secret="other") is None
    assert decode_access_token(token + "x", secret="s") is None


def test_token_expired_returns_none():
    past = datetime.now(UTC) - timedelta(hours=2)
    token = create_access_token("u1", secret="s", ttl_hours=1, now=past)
    assert decode_access_token(token, secret="s") is None
