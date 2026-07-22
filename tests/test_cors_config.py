# tests/test_cors_config.py
"""Stage 5: CORS is configurable so prod can lock it down (default stays permissive)."""

from __future__ import annotations

from meno_rag.config import parse_cors_origins


def test_empty_is_permissive_wildcard():
    assert parse_cors_origins("") == ["*"]
    assert parse_cors_origins("   ") == ["*"]


def test_single_origin():
    assert parse_cors_origins("https://meno.nsu.ru") == ["https://meno.nsu.ru"]


def test_comma_separated_trimmed():
    assert parse_cors_origins("https://a.ru, https://b.ru ,") == ["https://a.ru", "https://b.ru"]
