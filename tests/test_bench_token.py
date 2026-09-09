"""Распознавание токена бенчмарк-эндпоинта."""

from __future__ import annotations

from types import SimpleNamespace

from starlette.datastructures import Headers

from meno_rag.api.bench import is_bench_request
from meno_rag.config import Settings


def test_bench_settings_default_to_disabled():
    s = Settings()
    assert s.bench_api_token == ""
    assert s.bench_max_concurrent == 4


def test_bench_settings_read_from_env():
    s = Settings(BENCH_API_TOKEN="secret-token", BENCH_MAX_CONCURRENT=7)
    assert s.bench_api_token == "secret-token"
    assert s.bench_max_concurrent == 7


def _request(headers: dict[str, str], configured_token: str):
    """Минимальный дубль Request: используются только .headers и .app.state.settings."""
    settings = SimpleNamespace(bench_api_token=configured_token)
    app = SimpleNamespace(state=SimpleNamespace(settings=settings))
    return SimpleNamespace(headers=Headers(headers), app=app)


def test_valid_token_is_recognised():
    r = _request({"authorization": "Bearer secret-token"}, "secret-token")
    assert is_bench_request(r) is True


def test_bearer_prefix_is_case_insensitive():
    r = _request({"authorization": "bearer secret-token"}, "secret-token")
    assert is_bench_request(r) is True


def test_wrong_token_is_rejected():
    r = _request({"authorization": "Bearer wrong"}, "secret-token")
    assert is_bench_request(r) is False


def test_empty_setting_disables_the_feature():
    r = _request({"authorization": "Bearer anything"}, "")
    assert is_bench_request(r) is False


def test_missing_header_is_rejected():
    r = _request({}, "secret-token")
    assert is_bench_request(r) is False


def test_empty_bearer_value_is_rejected():
    r = _request({"authorization": "Bearer   "}, "secret-token")
    assert is_bench_request(r) is False


def test_non_bearer_scheme_is_rejected():
    r = _request({"authorization": "Basic secret-token"}, "secret-token")
    assert is_bench_request(r) is False


def test_x_auth_token_header_is_not_accepted():
    """X-Auth-Token принадлежит JWT приложения; bench-токен туда не ходит."""
    r = _request({"x-auth-token": "secret-token"}, "secret-token")
    assert is_bench_request(r) is False


def test_missing_settings_state_is_safe():
    """Никогда не бросает, даже если state ещё не заполнен."""
    r = SimpleNamespace(headers=Headers({"authorization": "Bearer x"}), app=SimpleNamespace(state=SimpleNamespace()))
    assert is_bench_request(r) is False
