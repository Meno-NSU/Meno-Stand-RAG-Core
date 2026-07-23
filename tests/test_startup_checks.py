"""Runtime safety checks: refuse SQLite in production, warn in dev."""

from __future__ import annotations

import pytest

from meno_rag.api.main import check_runtime_safety
from meno_rag.config import get_settings


def _settings(**overrides):
    return get_settings().model_copy(update=overrides)


def test_production_with_sqlite_raises():
    s = _settings(app_env="production", database_url="sqlite+aiosqlite:///./var/x.sqlite3")
    with pytest.raises(RuntimeError, match="SQLite"):
        check_runtime_safety(s)


def test_production_with_postgres_is_ok():
    s = _settings(
        app_env="production",
        database_url="postgresql+asyncpg://u:p@h:5432/db",
        # Off, so this stays a test of the database check alone. Previews default to on and
        # raise a standing reminder in production — asserted separately below.
        log_content_previews=False,
    )
    assert check_runtime_safety(s) == []


def test_production_warns_while_the_log_carries_content_previews():
    """The reminder is unconditional in production, because no check can verify that the log
    has an approved retention period and restricted access."""
    s = _settings(
        app_env="production",
        database_url="postgresql+asyncpg://u:p@h:5432/db",
        log_content_previews=True,
    )
    assert any("log_content_previews_on" in w for w in check_runtime_safety(s))


def test_dev_with_sqlite_warns_but_does_not_raise():
    s = _settings(app_env="development", database_url="sqlite+aiosqlite:///./var/x.sqlite3")
    warnings = check_runtime_safety(s)
    assert any("sqlite" in w.lower() for w in warnings)
