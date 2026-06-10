from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from meno_rag.api.leaderboard import router
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import ArenaVote, User
from meno_rag.db.session import Database


def test_get_leaderboard(tmp_path):
    db_path = tmp_path / "lb.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as s:
        s.add(User(id="u1", email="a@b.c", password_hash="h", nickname="Alice"))
        s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id="u1"))
        s.commit()
    engine.dispose()

    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.include_router(router)
    with TestClient(app) as c:
        r = c.get("/v1/leaderboard")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data and data[0]["nickname"] == "Alice" and data[0]["votes"] == 1
        assert all("email" not in row for row in data)
