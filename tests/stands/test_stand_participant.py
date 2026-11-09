"""The participant (stand, version, LLM) is decided by the server: it labels history rows, ratings and the
leaderboard, and is fixed when a round is stored so that a late vote counts for the version that answered."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import text

from meno_rag.api.knowledge_bases import LEGACY_KB_ID, V2_KB_ID
from meno_rag.api.stands import participant as participant_mod
from meno_rag.api.stands.participant import Participant, builtin_live_kbs, participant_for, stored_participant
from meno_rag.api.stands.registry import StandRegistry
from meno_rag.config import Settings
from meno_rag.db.orm import ArenaRating
from tests._dbhelpers import with_db
from tests.stands._helpers import LAB, manifest, stand_handler, write_stands
from tests.stands.test_stand_trace import _consenting_guest
from tests.v2.test_api_v2 import FakeLegacy, FakeV2, make_client  # noqa: F401  (fixture)


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
    assert builtin_live_kbs(s) == {LEGACY_KB_ID, V2_KB_ID}
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

    assert with_db(db, _read) == [V2_KB_ID]


def test_the_leaderboard_marks_live_rows(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    assert _vote(c, session=None).status_code == 200

    async def _seed(session):
        session.add(ArenaRating(model="meno-lite", knowledge_base="nsu-stand-faiss-bm25"))
        session.add(ArenaRating(model="meno-lite", knowledge_base="an-old-base"))

    with_db(db, _seed)
    rows = {(r["model"], r["knowledge_base"]): r["live"] for r in c.get("/v1/arena/leaderboard").json()["data"]}
    assert rows[("gemma", "lab@v1")] is True
    assert rows[("meno-lite", "nsu-stand-faiss-bm25")] is True
    assert rows[("meno-lite", "an-old-base")] is False


# ---- labels always fit their columns (String(128))


def test_an_overlong_stand_part_never_makes_a_long_label():
    got = participant_for("x@" + "y" * 200, registry=None, settings=_settings())
    assert got == Participant("menon-1.0", "2026-07", "x@" + "y" * 200, LEGACY_KB_ID)
    assert len(got.kb) <= 128


def test_stored_participant_rejects_an_overlong_stand_or_version():
    assert stored_participant("gemma@lab", "y" * 33 + "@v1") is None
    assert stored_participant("gemma@lab", "lab@" + "v" * 65) is None
    assert stored_participant("gemma@lab", "lab@v 1") is None


def test_a_vote_with_an_overlong_model_id_gets_a_short_rating_label(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    model = "x@" + "y" * 200
    assert _vote(c, model_a=model, model_b="gemma@lab", session=None).status_code == 200
    assert all(len(kb) <= 128 for _, kb in _ratings(db))


def test_a_vote_with_a_model_over_256_characters_is_refused(make_client, tmp_path):  # noqa: F811
    c, _, _ = _arena_client(make_client, tmp_path, version="v1")
    assert _vote(c, model_a="m" * 257, session=None).status_code == 422


# ---- our own pipelines keep their stored names and rating rows, as they did before stands


def _rows(db):
    async def _read(session):
        q = "SELECT model, knowledge_base, matches FROM arena_ratings"
        return {tuple(r) for r in (await session.execute(text(q))).all()}

    return with_db(db, _read)


def test_a_vote_on_our_model_lands_on_its_existing_row(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path)

    async def _seed(session):
        session.add(ArenaRating(model="meno-lite", knowledge_base=LEGACY_KB_ID, matches=7))

    with_db(db, _seed)
    assert _vote(c, model_a="meno-lite", kb_a=LEGACY_KB_ID, model_b="qwen", kb_b=LEGACY_KB_ID).status_code == 200
    assert _rows(db) == {("meno-lite", LEGACY_KB_ID, 8), ("qwen", LEGACY_KB_ID, 1)}


def test_a_vote_on_a_menon_1_1_entry_keeps_the_id_the_client_sent(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path)
    r = _vote(c, model_a="meno-lite@menon-1.1", kb_a=V2_KB_ID, model_b="meno-lite", kb_b=V2_KB_ID)
    assert r.status_code == 200
    assert _ratings(db) == {("meno-lite@menon-1.1", V2_KB_ID), ("meno-lite", V2_KB_ID)}


def test_a_client_cannot_write_into_a_stands_row_through_our_model(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path)

    async def _seed(session):
        session.add(ArenaRating(model="meno-lite", knowledge_base="ragu@1", matches=3))

    with_db(db, _seed)
    assert _vote(c, model_a="meno-lite", kb_a="ragu@1", model_b="qwen", kb_b="lab@v1").status_code == 200
    assert _rows(db) == {
        ("meno-lite", "ragu@1", 3),
        ("meno-lite", LEGACY_KB_ID, 1),
        ("qwen", LEGACY_KB_ID, 1),
    }


@pytest.mark.parametrize("model,fallback", [("meno-lite", LEGACY_KB_ID), ("meno-lite@menon-1.1", V2_KB_ID)])
def test_an_empty_kb_falls_back_to_the_pipeline_of_the_model(make_client, tmp_path, model, fallback):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path)
    assert _vote(c, model_a=model, kb_a="", model_b="qwen", kb_b="").status_code == 200
    assert _ratings(db) == {(model, fallback), ("qwen", LEGACY_KB_ID)}


def test_rating_kb_trusts_only_our_own_names():
    ours, v2 = (
        participant_for("meno-lite", registry=None, settings=_settings()),
        Participant("menon-1.1", "kb-v31h", "meno-lite", V2_KB_ID),
    )
    assert participant_mod.rating_kb(ours, V2_KB_ID) == V2_KB_ID  # today's frontend names the base itself
    assert participant_mod.rating_kb(ours, LEGACY_KB_ID) == LEGACY_KB_ID
    for unknown in ("ragu@1", "", None):
        assert participant_mod.rating_kb(ours, unknown) == LEGACY_KB_ID
        assert participant_mod.rating_kb(v2, unknown) == V2_KB_ID
    stand = Participant("lab", "v1", "gemma")
    assert participant_mod.rating_kb(stand, LEGACY_KB_ID) == "lab@v1"


@pytest.mark.parametrize(
    "client_kb,stored",
    [(LEGACY_KB_ID, LEGACY_KB_ID), (V2_KB_ID, V2_KB_ID), ("ragu@1", LEGACY_KB_ID), (None, LEGACY_KB_ID)],
)
def test_a_stored_side_of_our_model_keeps_the_known_kb(make_client, tmp_path, client_kb, stored):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path)
    sides = [
        {"key": "a", "model": "meno-lite", "knowledge_base_id": client_kb, "content": "A", "sources": []},
        {"key": "b", "model": "gemma@lab", "knowledge_base_id": client_kb, "content": "B", "sources": []},
    ]
    body = {"session_id": "c1", "question": "Q?", "turn_index": 0, "sides": sides}
    assert c.post("/v1/arena/turn", json=body).status_code == 200
    got = _stored_sides(db)
    assert got[0]["knowledge_base_id"] == stored and got[0]["model"] == "meno-lite"
    assert got[1]["knowledge_base_id"] == "lab@v1"  # a stand's label stays the server's


def test_a_late_vote_on_a_stored_side_of_our_model_keeps_its_row(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path)
    sides = [
        {"key": "a", "model": "meno-lite", "knowledge_base_id": V2_KB_ID, "content": "A", "sources": []},
        {"key": "b", "model": "gemma@lab", "knowledge_base_id": "x", "content": "B", "sources": []},
    ]
    body = {"session_id": "c1", "question": "Q?", "turn_index": 0, "sides": sides}
    assert c.post("/v1/arena/turn", json=body).status_code == 200
    assert _vote(c, model_a="meno-lite", kb_a="", kb_b="").status_code == 200
    assert _ratings(db) == {("meno-lite", V2_KB_ID), ("gemma", "lab@v1")}


class _ForAnalysis:
    """A fake pipeline whose outcome has everything the analysis tables read."""

    async def prepare(self, **kw):
        outcome = await super().prepare(**kw)
        outcome.prepared_dialogue_history, outcome.retrieved, outcome.fewshots = "", None, None
        return outcome


class _Legacy(_ForAnalysis, FakeLegacy):
    pass


class _V2(_ForAnalysis, FakeV2):
    pass


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "request_fields,kb",
    [
        ({"model": "menon-1"}, LEGACY_KB_ID),
        ({"model": "menon-1", "knowledge_base_id": V2_KB_ID}, V2_KB_ID),
        ({"model": "gemma@menon-1.1"}, V2_KB_ID),
    ],
)
def test_history_and_runs_of_our_pipelines_carry_the_kb_id(make_client, stream, request_fields, kb):  # noqa: F811
    c, db = make_client()
    c.app.state.pipeline, c.app.state.v2_pipeline = _Legacy(), _V2()
    headers = _consenting_guest(c, db)
    body = {**request_fields, "stream": stream, "user": "conv-1", "messages": [{"role": "user", "content": "Сессия?"}]}
    assert c.post("/v1/chat/completions", json=body, headers=headers).status_code == 200

    async def _read(session):
        messages = (await session.execute(text("SELECT knowledge_base_id FROM messages"))).scalars().all()
        runs = (await session.execute(text("SELECT knowledge_base_id FROM pipeline_runs"))).scalars().all()
        return messages, runs

    assert with_db(db, _read) == ([kb, kb], [kb])


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_run_of_our_pipeline_carries_the_kb_id(make_client, stream):  # noqa: F811
    class Broken(FakeLegacy):
        async def prepare(self, **kw):
            raise RuntimeError("boom")

    c, db = make_client()
    c.app.state.pipeline = Broken()
    body = {"model": "menon-1", "stream": stream, "user": "conv-1", "messages": [{"role": "user", "content": "Q?"}]}
    c.post("/v1/chat/completions", json=body)

    async def _read(session):
        return (await session.execute(text("SELECT knowledge_base_id FROM pipeline_runs"))).scalars().all()

    assert with_db(db, _read) == [LEGACY_KB_ID]


def test_the_metric_labels_of_our_pipelines_still_name_the_stand():
    got = participant_for("meno-lite", registry=None, settings=_settings(), kb_hint=V2_KB_ID)
    assert (got.stand, got.version, got.llm, got.kb) == ("menon-1.1", "kb-v31h", "meno-lite", V2_KB_ID)
    got = participant_for("gemma@menon-1.1", registry=None, settings=_settings())
    assert (got.stand, got.version, got.llm, got.kb) == ("menon-1.1", "kb-v31h", "gemma", V2_KB_ID)
    got = participant_for("meno-lite", registry=None, settings=_settings())
    assert (got.stand, got.version, got.llm, got.kb) == ("menon-1.0", "2026-07", "meno-lite", LEGACY_KB_ID)
    got = participant_for("gemma@gone", registry=None, settings=_settings())
    assert got.builtin_kb is None and got.kb == "gone@unknown"


# ---- an id with "@" that a provider lists is ours, not a stand's


def _provider_lists(c, model):
    c.app.state.vllm_registry.list_models = AsyncMock(return_value=[{"id": model, "endpoint": "http://v"}])


def test_a_vote_on_a_listed_id_keeps_its_row(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path)
    _provider_lists(c, "org/model@v2")
    r = _vote(c, model_a="org/model@v2", kb_a=V2_KB_ID, model_b="qwen", kb_b=LEGACY_KB_ID, session=None)
    assert r.status_code == 200
    assert _ratings(db) == {("org/model@v2", V2_KB_ID), ("qwen", LEGACY_KB_ID)}


def test_a_stored_side_of_a_listed_id_keeps_its_kb(make_client, tmp_path):  # noqa: F811
    c, db, _ = _arena_client(make_client, tmp_path)
    _provider_lists(c, "org/model@v2")
    sides = [
        {"key": "a", "model": "org/model@v2", "knowledge_base_id": V2_KB_ID, "content": "A", "sources": []},
        {"key": "b", "model": "qwen", "knowledge_base_id": LEGACY_KB_ID, "content": "B", "sources": []},
    ]
    body = {"session_id": "c1", "question": "Q?", "turn_index": 0, "sides": sides}
    assert c.post("/v1/arena/turn", json=body).status_code == 200
    assert _stored_sides(db)[0]["knowledge_base_id"] == V2_KB_ID
