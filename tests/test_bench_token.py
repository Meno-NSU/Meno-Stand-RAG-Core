"""Распознавание токена бенчмарк-эндпоинта."""

from __future__ import annotations

from types import SimpleNamespace

from starlette.datastructures import Headers

from meno_rag.api.bench import is_bench_request
from meno_rag.config import Settings


def test_bench_settings_default_to_disabled():
    # M4: `_env_file=None` asserts the code's own defaults, not whatever
    # BENCH_API_TOKEN happens to be set in the machine's real `.env` — this
    # test must pass identically on a prod host or a dev machine that has the
    # feature enabled locally.
    s = Settings(_env_file=None)
    assert s.bench_api_token == ""
    assert s.bench_max_concurrent == 16


def test_bench_settings_read_from_env():
    s = Settings(BENCH_API_TOKEN="secret-token", BENCH_MAX_CONCURRENT=7)
    assert s.bench_api_token == "secret-token"
    assert s.bench_max_concurrent == 7


def test_bench_api_token_is_stripped_of_surrounding_whitespace():
    """M3: a trailing/leading space in `.env` must not silently produce a
    configured token that can never match any client-supplied credential."""
    s = Settings(_env_file=None, BENCH_API_TOKEN="  secret-token  ")
    assert s.bench_api_token == "secret-token"


def _request(headers: dict[str, str], configured_token: str):
    """Минимальный дубль Request: используются только .headers и .app.state.settings."""
    settings = SimpleNamespace(bench_api_token=configured_token)
    app = SimpleNamespace(state=SimpleNamespace(settings=settings))
    return SimpleNamespace(headers=Headers(headers), app=app)


def test_valid_token_is_recognised():
    r = _request({"authorization": "Bearer secret-token"}, "secret-token")
    assert is_bench_request(r) is True


def test_configured_token_with_whitespace_still_matches_clean_client_token():
    """M3, end-to-end: whitespace around the configured token (read through a
    real Settings object, not the SimpleNamespace double `_request` uses)
    must not break matching against a clean client-supplied token."""
    settings = Settings(_env_file=None, BENCH_API_TOKEN="  secret-token  ")
    app = SimpleNamespace(state=SimpleNamespace(settings=settings))
    r = SimpleNamespace(headers=Headers({"authorization": "Bearer secret-token"}), app=app)
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


def test_non_ascii_bearer_credential_is_rejected_not_raised():
    """C1: Starlette decodes headers as latin-1, so a raw byte >= 0x80 in the
    Authorization value surfaces as a non-ASCII str. hmac.compare_digest raises
    TypeError on non-ASCII str operands, and that must never escape
    is_bench_request — its docstring promises "Never raises," and a malformed
    header must never be able to 500 the production chat endpoint.

    httpx refuses to send a non-ASCII str header client-side, so this builds
    the header at the byte level (as an ASGI server would hand it to
    Starlette off the wire) rather than going through httpx or a plain str.
    """
    headers = Headers(raw=[(b"authorization", b"Bearer caf\xe9")])
    settings = SimpleNamespace(bench_api_token="secret-token")
    app = SimpleNamespace(state=SimpleNamespace(settings=settings))
    r = SimpleNamespace(headers=headers, app=app)
    assert is_bench_request(r) is False


def test_ascii_token_still_matches_when_header_built_at_byte_level():
    """Guards the C1 encode/compare refactor: a correct ASCII credential,
    constructed the same raw-bytes way as the regression test above, must
    still be recognised — the fix must not break the happy path."""
    headers = Headers(raw=[(b"authorization", b"Bearer secret-token")])
    settings = SimpleNamespace(bench_api_token="secret-token")
    app = SimpleNamespace(state=SimpleNamespace(settings=settings))
    r = SimpleNamespace(headers=headers, app=app)
    assert is_bench_request(r) is True
