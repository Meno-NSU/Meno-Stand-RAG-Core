"""Leaderboard rows name their version and model; the old fields keep their values."""

from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from meno_rag.api.arena import router
from meno_rag.api.knowledge_bases import LEGACY_KB_ID, V2_KB_ID
from meno_rag.api.stands.registry import StandRegistry
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import ArenaRating
from meno_rag.db.session import Database
from tests.stands._helpers import manifest, stand_handler, write_stands

RAGU = {"id": "ragu", "url": "http://ragu.example:9007/", "api_key": "k-1"}  # no name: the manifest's is used

ROWS = [
    ("gemma-4-12B-it", LEGACY_KB_ID),
    ("gemma-4-12B-it", V2_KB_ID),
    ("gemma", "ragu@nsu-ragu-2026-11"),
    ("gemma", "old@v1"),
    ("some-model", "ragu@nsu-ragu-2026-11"),
]


def _rows(tmp_path):
    db_path = tmp_path / "lb.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as s:
        for model, kb in ROWS:
            s.add(ArenaRating(model=model, knowledge_base=kb))
        s.commit()
    engine.dispose()

    path = write_stands(tmp_path / "stands.json", [RAGU])
    mf = manifest("nsu-ragu-2026-11", name="RAGU")
    http = httpx.AsyncClient(transport=httpx.MockTransport(stand_handler([], manifest=mf)))
    registry = StandRegistry(str(path), http_client=http)

    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings()
    app.state.stand_registry = registry
    app.include_router(router)
    with TestClient(app) as c:
        c.portal.call(registry.refresh)
        data = c.get("/v1/arena/leaderboard").json()["data"]
    return {(r["model"], r["knowledge_base"]): r for r in data}


def _labels(row):
    return row["stand_id"], row["stand_name"], row["stand_version"], row["llm_name"]


def test_rows_name_their_version_and_model(tmp_path):
    rows = _rows(tmp_path)
    assert _labels(rows[("gemma-4-12B-it", LEGACY_KB_ID)]) == ("menon-1.0", "Менон 1.0", None, "Gemma 4 12B")
    assert _labels(rows[("gemma-4-12B-it", V2_KB_ID)]) == ("menon-1.1", "Менон 1.1", None, "Gemma 4 12B")
    assert _labels(rows[("gemma", "ragu@nsu-ragu-2026-11")]) == ("ragu", "RAGU", "nsu-ragu-2026-11", "gemma")
    assert rows[("some-model", "ragu@nsu-ragu-2026-11")]["llm_name"] == "some-model"


def test_a_stand_no_longer_configured_still_gets_a_label(tmp_path):
    row = _rows(tmp_path)[("gemma", "old@v1")]
    assert (row["stand_id"], row["stand_name"], row["stand_version"]) == ("old", "old", "v1")


def test_the_old_fields_are_unchanged(tmp_path):
    rows = _rows(tmp_path)
    row = rows[("gemma", "ragu@nsu-ragu-2026-11")]
    assert {"model", "knowledge_base", "elo", "live"} <= set(row)
    assert row["live"] is True and rows[("gemma", "old@v1")]["live"] is False
    assert rows[("gemma-4-12B-it", LEGACY_KB_ID)]["model"] == "gemma-4-12B-it"
