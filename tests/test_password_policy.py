from __future__ import annotations

import pytest
from pydantic import ValidationError

from meno_rag.schemas import RegisterRequest


def test_password_over_72_utf8_bytes_rejected():
    # 37 Cyrillic chars = 74 UTF-8 bytes — over the bcrypt 72-byte limit, though
    # only 37 characters. Must be rejected, not silently truncated (ТЗ §5.9).
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="я" * 37)


def test_password_exactly_72_bytes_accepted():
    req = RegisterRequest(email="a@b.com", password="я" * 36)  # 36 * 2 = 72 bytes
    assert len(req.password.encode("utf-8")) == 72


def test_short_password_still_rejected():
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="short")
