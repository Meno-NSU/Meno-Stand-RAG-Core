"""The participant (stand, version, LLM) is decided by the server: it labels history rows, ratings and the
leaderboard, and is fixed when a round is stored so that a late vote counts for the version that answered."""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import text

from meno_rag.api.knowledge_bases import V2_KB_ID
from meno_rag.api.stands.participant import Participant, builtin_live_kbs, participant_for, stored_participant
from meno_rag.api.stands.registry import StandRegistry
from meno_rag.config import Settings
from meno_rag.db.orm import ArenaRating
from tests._dbhelpers import with_db
from tests.stands._helpers import LAB, manifest, stand_handler, write_stands
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)


def _settings():
    return Settings(V2_VERSION="kb-v31h", LEGACY_VERSION="2026-07")


def _registry(tmp_path, version="v1"):
    path = write_stands(tmp_path / "stands.json", [LAB])
    http = httpx.AsyncClient(transport=httpx.MockTransport(stand_handler([], manifest=manifest(version))))
    return StandRegistry(str(path), http_client=http)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,hint,expected",
    [
        ("gemma@lab", None, ("lab", "v1", "gemma")),
        ("gemma@menon-1.1", None, ("menon-1.1", "kb-v31h", "gemma")),
        ("gemma@removed", None, ("removed", "unknown", "gemma")),
        ("meno-lite", None, ("menon-1.0", "2026-07", "meno-lite")),
        ("meno-lite", V2_KB_ID, ("menon-1.1", "kb-v31h", "meno-lite")),
    ],
)
async def test_participant_for(tmp_path, model, hint, expected):
    registry = _registry(tmp_path)
    await registry.refresh()
    got = participant_for(model, registry=registry, settings=_settings(), kb_hint=hint)
    assert (got.stand, got.version, got.llm) == expected


def test_participant_for_without_a_registry():
    got = participant_for("gemma@lab", registry=None, settings=_settings())
    assert got == Participant("lab", "unknown", "gemma")


def test_stored_participant_reads_only_the_new_form():
    assert stored_participant("gemma@lab", "lab@v1") == Participant("lab", "v1", "gemma")
    assert stored_participant("gemma@lab", "nsu-stand-faiss-bm25") is None
    assert stored_participant("gemma@lab", None) is None


def test_builtin_live_kbs():
    s = Settings(V2_ENABLED=True, V2_VERSION="kb-v31h", LEGACY_ENABLED=True)
    assert builtin_live_kbs(s) == {"menon-1.0@2026-07", "menon-1.1@kb-v31h"}
    assert builtin_live_kbs(Settings(V2_ENABLED=False, LEGACY_ENABLED=False)) == set()


# ---- through the API


def _arena_client(make_client, tmp_path, version="v1"):  # noqa: F811
    path = write_stands(tmp_path / "stands.json", [LAB])
    c, db = make_client()
    http = httpx.AsyncClient(transport=httpx.MockTransport(stand_handler([], manifest=manifest(version))))
    registry = StandRegistry(str(path), http_client=http)
    c.app.state.stand_registry = registry
    c.portal.call(registry.refresh)
    return c, db, registry


def _swap_registry(c, tmp_path, stands, version="v2"):
    path = write_stands(tmp_path / "stands2.json", stands)
    http = httpx.AsyncClient(transport=httpx.MockTransport(stand_handler([], manifest=manifest(version))))
    registry = StandRegistry(str(path), http_client=http)
    c.app.state.stand_registry = registry
    c.portal.call(registry.refresh)


def _record_turn(c, turn_index=0, kb="nsu-stand-faiss-bm25", model_b="gemma@lab"):
    sides = [
        {"key": "a", "model": "gemma@lab", "knowledge_base_id": kb, "content": "A", "sources": []},
        {"key": "b", "model": model_b, "knowledge_base_id": kb, "content": "B", "sources": []},
    ]
    body = {"session_id": "c1", "question": "Q?", "turn_index": turn_index, "sides": sides}
    assert c.post("/v1/arena/turn", json=body).status_code == 200


def _vote(c, turn_index=0, kb_a="x", kb_b="y", model_a="gemma@lab", model_b="gemma@lab", session="c1"):
    body = {"model_a": model_a, "kb_a": kb_a, "model_b": model_b, "kb_b": kb_b, "winner": "a", "turn_index": turn_index}
    if session:
        body["session_id"] = session
    return c.post("/v1/arena/vote", json=body)


def _ratings(db):
    async def _read(session):
        return {
            tuple(r) for r in (await session.execute(text("SELECT model, knowledge_base FROM arena_ratings"))).all()
        }

    return with_db(db, _read)


def _stored_sides(db):
    import json

    async def _read(session):
        raw = (await session.execute(text("SELECT arena FROM messages WHERE turn_kind='arena'"))).scalar_one()
        return (json.loads(raw) if isinstance(raw, str) else raw)["sides"]

    return with_db(db, _read)


def test_a_late_vote_counts_for_the_version_that_answered(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    _record_turn(c)
    _swap_registry(c, tmp_path, [LAB], version="v2")
    assert _vote(c).status_code == 200
    assert ("gemma", "lab@v1") in _ratings(db) and ("gemma", "lab@v2") not in _ratings(db)


def test_a_late_vote_survives_the_stand_being_removed(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    _record_turn(c)
    _swap_registry(c, tmp_path, [])
    assert _vote(c).status_code == 200
    assert _ratings(db) == {("gemma", "lab@v1")}


def test_a_vote_without_a_stored_turn_uses_the_registry(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    assert _vote(c).status_code == 200
    assert _ratings(db) == {("gemma", "lab@v1")}


def test_a_vote_with_no_turn_and_no_stand_is_marked_unknown(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    _swap_registry(c, tmp_path, [])
    assert _vote(c).status_code == 200
    assert _ratings(db) == {("gemma", "lab@unknown")}


def test_a_stored_turn_for_another_model_is_not_trusted(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    _record_turn(c)
    _swap_registry(c, tmp_path, [LAB], version="v2")
    # side b of the vote names a model other than the stored one: it is resolved now, not read from the turn
    assert _vote(c, model_b="gemma@gone").status_code == 200
    assert _ratings(db) == {("gemma", "lab@v1"), ("gemma", "gone@unknown")}


def test_the_client_kb_is_ignored_and_may_be_empty(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    assert _vote(c, kb_a="bogus", kb_b="").status_code == 200
    assert not any(kb == "bogus" for _, kb in _ratings(db))
    assert _vote(c, turn_index=1, kb_a="x" * 129).status_code == 422


def test_the_stored_turn_records_the_participant(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    _record_turn(c, kb="nsu-stand-faiss-bm25")
    sides = _stored_sides(db)
    assert sides[0]["knowledge_base_id"] == "lab@v1"
    assert sides[0]["model"] == "gemma@lab"  # the public id stays: the frontend shows it and sends it back


def test_in_process_answers_are_labelled_by_participant(make_client, monkeypatch):  # noqa: F811
    monkeypatch.setenv("V2_VERSION", "kb-v31h")
    c, db = make_client()
    r = c.post(
        "/v1/chat/completions",
        json={"model": "gemma@menon-1.1", "user": "conv-1", "messages": [{"role": "user", "content": "Когда сессия?"}]},
    )
    assert r.status_code == 200

    async def _read(session):
        q = "SELECT knowledge_base_id FROM messages WHERE role='assistant'"
        return [r[0] for r in (await session.execute(text(q))).all()]

    assert with_db(db, _read) == ["menon-1.1@kb-v31h"]


def test_the_leaderboard_marks_live_rows(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    assert _vote(c, session=None).status_code == 200

    async def _seed(session):
        session.add(ArenaRating(model="meno-lite", knowledge_base="nsu-stand-faiss-bm25"))

    with_db(db, _seed)
    rows = {(r["model"], r["knowledge_base"]): r["live"] for r in c.get("/v1/arena/leaderboard").json()["data"]}
    assert rows[("gemma", "lab@v1")] is True
    assert rows[("meno-lite", "nsu-stand-faiss-bm25")] is False
