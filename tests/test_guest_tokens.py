from __future__ import annotations

from meno_rag.api.guest_tokens import generate_guest_token, hash_guest_token, verify_guest_token


def test_generate_is_high_entropy_and_unique():
    a = generate_guest_token()
    b = generate_guest_token()
    assert a != b
    assert len(a) >= 43  # 32 random bytes, url-safe base64 → 43 chars


def test_hash_is_deterministic_hex64():
    token = "example-token"
    h = hash_guest_token(token)
    assert h == hash_guest_token(token)
    assert len(h) == 64
    assert h != token


def test_verify_matches_only_the_right_token():
    token = generate_guest_token()
    token_hash = hash_guest_token(token)
    assert verify_guest_token(token, token_hash) is True
    assert verify_guest_token("wrong", token_hash) is False
