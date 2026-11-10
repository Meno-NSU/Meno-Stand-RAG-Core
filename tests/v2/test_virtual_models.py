"""The v2 pipeline offered as «Менон 1.1» entries of the model list."""

from __future__ import annotations

from sqlalchemy import text

from meno_rag.api.knowledge_bases import LEGACY_KB_ID, V2_KB_ID, split_virtual_model, virtual_model_id
from tests._dbhelpers import with_db
from tests.v2.test_api_v2 import _make, make_client  # noqa: F401  (fixture)

LITE = "Менон 1.1 (meno-lite)"
GEMMA = "Менон 1.1 (Gemma 4)"


class FakeRegistry:
    def __init__(self, ids):
        self.ids = ids

    async def list_models(self):
        return [{"id": i, "endpoint": "http://fake/v1"} for i in self.ids]


def _client(make_client, ids=("meno-lite", "gemma-4-12B-it", "other"), **kw):  # noqa: F811
    c, db = make_client(**kw)
    c.app.state.vllm_registry = FakeRegistry(ids)
    return c, db


def _chat(c, model, kb=None):
    body = {"model": model, "messages": [{"role": "user", "content": "Когда сессия?"}]}
    if kb:
        body["knowledge_base_id"] = kb
    return c.post("/v1/chat/completions", json=body)


def test_split_helpers():
    assert virtual_model_id("a") == "a@menon-1.1"
    assert split_virtual_model("a@menon-1.1") == ("a", True)
    assert split_virtual_model("a") == ("a", False)
    assert split_virtual_model(None) == (None, False)


def test_list_has_virtual_entries(make_client):
    c, _ = _client(make_client)
    data = {m["id"]: m for m in c.get("/v1/models").json()["data"]}
    assert data["meno-lite@menon-1.1"]["display_name"] == LITE
    assert data["gemma-4-12B-it@menon-1.1"]["display_name"] == GEMMA
    v = data["meno-lite@menon-1.1"]
    assert v["pipeline"] == "v2" and v["knowledge_base_id"] == V2_KB_ID
    assert v["endpoint"] == data["meno-lite"]["endpoint"]
    assert data["meno-lite"]["display_name"] == "meno-lite" and "pipeline" not in data["meno-lite"]
    assert "other@menon-1.1" not in data


def test_absent_base_no_entry(make_client):
    c, _ = _client(make_client, ids=("meno-lite",))
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "meno-lite@menon-1.1" in ids and "gemma-4-12B-it@menon-1.1" not in ids


def test_v2_unavailable_no_entries(make_client):
    c, _ = _client(make_client, v2_up=False)
    assert not [m for m in c.get("/v1/models").json()["data"] if m["id"].endswith("@menon-1.1")]


def test_chat_virtual_uses_v2(make_client, monkeypatch):
    c, db = _client(make_client)
    from meno_rag.api import main as main_mod

    spy = main_mod._resolve_runtime
    r = _chat(c, "meno-lite@menon-1.1", kb=LEGACY_KB_ID)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Новый"
    assert r.json()["model"] == "meno-lite@menon-1.1"
    assert spy.await_args.args[1] == "meno-lite"

    async def _read(session):
        return (await session.execute(text("SELECT model, knowledge_base_id FROM messages"))).all()

    rows = with_db(db, _read)
    assert rows and all(tuple(r) == ("meno-lite@menon-1.1", V2_KB_ID) for r in rows if r[0])


def test_chat_virtual_503_when_v2_down(make_client):
    c, _ = _client(make_client, v2_up=False)
    assert _chat(c, "meno-lite@menon-1.1").status_code == 503


def test_plain_model_unchanged(make_client):
    c, _ = _client(make_client)
    r = _chat(c, "meno-lite")
    assert r.json()["choices"][0]["message"]["content"] == "Старый"
    assert r.json()["model"] == "menon-1"
    assert _chat(c, "meno-lite", kb=V2_KB_ID).json()["choices"][0]["message"]["content"] == "Новый"


def test_kb_list_hides_v2_by_default(make_client, monkeypatch):
    c, _ = _client(make_client)
    assert [k["id"] for k in c.get("/v1/knowledge-bases").json()["data"]] == [LEGACY_KB_ID]
    monkeypatch.setattr(c.app.state.settings, "v2_expose_as", "both")
    assert len(c.get("/v1/knowledge-bases").json()["data"]) == 2


def test_arena_accepts_virtual_ids(make_client):
    c, _ = _client(make_client)
    vote = {
        "model_a": "meno-lite",
        "kb_a": LEGACY_KB_ID,
        "model_b": "meno-lite@menon-1.1",
        "kb_b": V2_KB_ID,
        "winner": "b",
        "question": "q",
        "session_id": "s1",
    }
    r = c.post("/v1/arena/vote", json=vote)
    assert r.status_code in (200, 201), r.text
