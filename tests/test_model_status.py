from datetime import UTC, datetime

from meno_rag.llm.status import ModelStatus, ModelStatusState


def test_model_status_defaults_to_available():
    s = ModelStatus.available()
    assert s.state == ModelStatusState.AVAILABLE
    assert s.until is None
    assert s.last_error is None
    assert s.consecutive_failures == 0
    assert isinstance(s.updated_at, datetime)


def test_model_status_rate_limited_carries_until():
    reset = datetime(2030, 1, 1, tzinfo=UTC)
    s = ModelStatus.rate_limited(until=reset, error="rate_limit_exceeded")
    assert s.state == ModelStatusState.RATE_LIMITED
    assert s.until == reset
    assert s.last_error == "rate_limit_exceeded"


def test_model_status_to_dict_round_trip():
    reset = datetime(2030, 1, 1, tzinfo=UTC)
    s = ModelStatus.rate_limited(until=reset, error="rate_limit_exceeded")
    payload = s.to_dict()
    restored = ModelStatus.from_dict(payload)
    assert restored == s


def test_model_status_unreachable_carries_backoff_info():
    until = datetime(2030, 1, 1, tzinfo=UTC)
    s = ModelStatus.unreachable(until=until, error="connection_timeout", consecutive_failures=3)
    assert s.state == ModelStatusState.UNREACHABLE
    assert s.until == until
    assert s.last_error == "connection_timeout"
    assert s.consecutive_failures == 3
