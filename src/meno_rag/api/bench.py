"""Recognition of the developer benchmark token.

nginx already 404s the benchmark path without a valid token, so by the time a
request reaches here the edge has checked it. This module checks it a second
time on purpose: nginx decides *reachability*, the application needs the flag to
decide *behavior* — return the full pipeline trace, write nothing to production
tables, and draw from a separate concurrency budget. It also means an SSH tunnel
straight to 127.0.0.1:9006 gets bench mode with no nginx in the path.
"""

from __future__ import annotations

import hmac

from fastapi import Request


def _bearer_token(request: Request) -> str | None:
    """The Bearer credential from ``Authorization``, or None.

    Deliberately does NOT read ``X-Auth-Token``: that header carries the
    application's JWT, and letting two different kinds of secret share one header
    is how they eventually get mistaken for each other.
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip() or None


def is_bench_request(request: Request) -> bool:
    """True when the request carries the configured benchmark token. Never raises."""
    settings = getattr(request.app.state, "settings", None)
    expected = getattr(settings, "bench_api_token", "") or ""
    if not expected:
        return False
    token = _bearer_token(request)
    if token is None:
        return False
    return hmac.compare_digest(token, expected)
