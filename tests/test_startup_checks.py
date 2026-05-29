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
    s = _settings(app_env="production", database_url="postgresql+asyncpg://u:p@h:5432/db")
    assert check_runtime_safety(s) == []


def test_dev_with_sqlite_warns_but_does_not_raise():
    s = _settings(app_env="development", database_url="sqlite+aiosqlite:///./var/x.sqlite3")
    warnings = check_runtime_safety(s)
    assert any("sqlite" in w.lower() for w in warnings)
