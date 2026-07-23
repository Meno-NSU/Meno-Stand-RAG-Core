# tests/test_leaderboard_sealed.py
"""The contributor leaderboard is sealed: the module exists, the route does not.

`api/leaderboard.py` is kept so the feature can come back, but it must not be reachable.
Publishing nicknames and per-user activity to every visitor is распространение, which
needs its own consent under art. 10.1 152-ФЗ — and the published Consent says outright
that distribution is not included. This test fails the moment someone mounts it again.
"""

from __future__ import annotations

from meno_rag.api import leaderboard, main


def _routes(app) -> list[str]:
    return [getattr(route, "path", "") for route in app.routes]


def test_the_module_is_still_there():
    # Sealed, not deleted — the router object must remain importable.
    assert leaderboard.router.prefix == "/v1/leaderboard"


def test_the_assembled_app_exposes_no_leaderboard_route():
    paths = _routes(main.app)
    assert not any(path.startswith("/v1/leaderboard") for path in paths), (
        "The contributor board is mounted again. It needs a separate art. 10.1 152-ФЗ "
        "consent plus matching Policy/Consent wording first — see api/leaderboard.py."
    )
    # The arena board is a different thing: anonymous model aggregates, no user data.
    assert "/v1/arena/leaderboard" in paths
