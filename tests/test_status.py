from fastapi.testclient import TestClient

from meno_rag.api.admission import AdmissionController


def test_status_reports_active_and_limit():
    from meno_rag.api import main as main_mod

    with TestClient(main_mod.app) as c:
        c.app.state.admission = AdmissionController(256)
        assert c.app.state.admission.try_acquire() is True  # active -> 1
        r = c.get("/v1/status")

    assert r.status_code == 200
    assert r.json() == {"active_requests": 1, "limit": 256}


def test_status_degrades_when_admission_missing():
    from meno_rag.api import main as main_mod

    with TestClient(main_mod.app) as c:
        c.app.state.admission = None
        r = c.get("/v1/status")

    assert r.status_code == 200
    assert r.json() == {"active_requests": 0, "limit": 0}
