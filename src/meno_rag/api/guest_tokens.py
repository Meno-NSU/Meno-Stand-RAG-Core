"""Guest-session token primitives: high-entropy tokens + SHA-256 hashing.

The raw token is a 256-bit URL-safe secret handed to the browser once and never
stored. Only its SHA-256 hash is persisted; lookups compare hashes in constant
time. bcrypt is deliberately NOT used — these tokens are already high-entropy, so
a fast digest is correct and avoids bcrypt's 72-byte limit.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


def generate_guest_token() -> str:
    """Return a fresh URL-safe 256-bit secret (never stored raw)."""
    return secrets.token_urlsafe(32)


def hash_guest_token(token: str) -> str:
    """Return the hex SHA-256 of the token (64 chars)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_guest_token(token: str, token_hash: str) -> bool:
    """Constant-time check that ``token`` hashes to ``token_hash``."""
    return hmac.compare_digest(hash_guest_token(token), token_hash)
