import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_no_or(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    from meno_rag.config import get_settings

    get_settings.cache_clear()
    from meno_rag.api.main import app

    with TestClient(app) as c:
        yield c


def test_openrouter_disabled_appears_in_healthz(client_no_or):
    r = client_no_or.get("/healthz")
    body = r.json()
    assert body["openrouter"]["state"] == "disabled"
