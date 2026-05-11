from datetime import datetime, timezone

from meno_rag.llm.status import ModelStatus, ModelStatusState


def test_model_status_defaults_to_available():
    s = ModelStatus.available()
    assert s.state == ModelStatusState.AVAILABLE
    assert s.until is None
    assert s.last_error is None
    assert s.consecutive_failures == 0
    assert isinstance(s.updated_at, datetime)


def test_model_status_rate_limited_carries_until():
    reset = datetime(2030, 1, 1, tzinfo=timezone.utc)
    s = ModelStatus.rate_limited(until=reset, error="rate_limit_exceeded")
    assert s.state == ModelStatusState.RATE_LIMITED
    assert s.until == reset
    assert s.last_error == "rate_limit_exceeded"


def test_model_status_to_dict_round_trip():
    reset = datetime(2030, 1, 1, tzinfo=timezone.utc)
    s = ModelStatus.rate_limited(until=reset, error="rate_limit_exceeded")
    payload = s.to_dict()
    restored = ModelStatus.from_dict(payload)
    assert restored == s
