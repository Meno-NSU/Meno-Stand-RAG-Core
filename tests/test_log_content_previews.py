# tests/test_log_content_previews.py
"""`LOG_CONTENT_PREVIEWS` decides whether excerpts of a dialogue reach the log.

Excerpts make a bad answer diagnosable from the log alone, and they are personal data:
lawful to log while the log has an approved retention period, restricted access, and the
processing is described in the published Policy. The switch exists so a deployment where
that does not hold can turn them off without a code change.
"""

from __future__ import annotations

import pytest

from meno_rag import logging_config
from meno_rag.config import Settings
from meno_rag.logging_config import preview_field


@pytest.fixture(autouse=True)
def restore_default():
    original = logging_config._CONTENT_PREVIEWS
    yield
    logging_config._CONTENT_PREVIEWS = original


def _set(enabled: bool) -> None:
    logging_config._CONTENT_PREVIEWS = enabled


def test_enabled_yields_a_truncated_excerpt():
    _set(True)
    assert preview_field("content_preview", "x" * 500, 200) == {"content_preview": "x" * 200}


def test_disabled_yields_no_key_at_all():
    _set(False)
    # Not an empty string, not None — the key must be absent, so a disabled preview
    # leaves nothing behind in the log line.
    assert preview_field("content_preview", "секретный вопрос", 200) == {}


def test_empty_text_yields_no_key():
    _set(True)
    assert preview_field("content_preview", "", 200) == {}


def test_shorter_than_the_limit_is_passed_through():
    _set(True)
    assert preview_field("answer_preview", "короткий ответ", 200) == {"answer_preview": "короткий ответ"}


def test_configure_logging_carries_the_setting():
    logging_config.configure_logging("INFO", content_previews=False)
    assert preview_field("content_preview", "text", 10) == {}
    logging_config.configure_logging("INFO", content_previews=True)
    assert preview_field("content_preview", "text", 10) == {"content_preview": "text"}


def test_default_is_on_so_debugging_keeps_working():
    assert Settings().log_content_previews is True
