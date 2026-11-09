"""The v2 pipeline served as a second knowledge base."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from meno_rag.api.knowledge_bases import LEGACY_KB_ID, V2_KB_ID
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.api.events import StageEvent, StageName, StageStatus
from tests._dbhelpers import with_db

BENCH_TOKEN = "test-bench-token-0123456789"
SRC = [{"document_title": "Устав", "source_url": "https://nsu.ru/x"}]


def _outcome(sources, trace=None):
    return SimpleNamespace(
        question="Когда сессия?",
        sources=sources,
        stage_durations_ms={},
        stage_details={},
        qa_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        search_queries=["q"],
        trace=trace,
    )


class FakeLegacy:
    async def prepare(self, *, messages, runtime, stage_sink=None, capture_trace=False):
        if stage_sink:
            await stage_sink(StageEvent(stage=StageName.RETRIEVAL, status=StageStatus.COMPLETED, duration_ms=1.0))
        return _outcome(list(SRC))

    async def stream_text(self, *, outcome, runtime, max_tokens, temperature):
        for t in ("Ста", "рый"):
            yield t

    async def generate_text(self, **kw):
        return "Старый"


class FakeV2(FakeLegacy):
    async def prepare(self, *, messages, runtime, stage_sink=None, capture_trace=False):
        if stage_sink:
            await stage_sink(StageEvent(stage=StageName.RETRIEVAL, status=StageStatus.COMPLETED, duration_ms=1.0))
        return _outcome([], trace={"v2": {"route": "nsu"}, "search_queries": ["q"]})

    async def stream_text(self, *, outcome, runtime, max_tokens, temperature):
        for t in ("Но", "вый"):
            yield t

    async def generate_text(self, **kw):
        return "Новый"

    async def finalize_sources(self, outcome, answer, runtime=None):
        assert answer in ("Новый",) or answer == "Новый"
        return list(SRC)


def _make(monkeypatch, tmp_path, *, legacy=True, v2=True, legacy_up=True, v2_up=True):
    db_path = tmp_path / "v2.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("BENCH_API_TOKEN", BENCH_TOKEN)
    monkeypatch.setenv("LEGACY_ENABLED", "true" if legacy else "false")
    monkeypatch.setenv("V2_ENABLED", "true" if v2 else "false")
    monkeypatch.setenv("V2_KB_DIR", "")
    from meno_rag.config import get_settings

    get_settings.cache_clear()
    from meno_rag.api import main as main_mod

    main_mod.get_settings.cache_clear()  # main's own reference, in case config was reloaded elsewhere
    runtime = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://fake/v1"))
    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(return_value=runtime))
    return main_mod, db_path


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    from meno_rag.config import get_settings

    opened = []

    def factory(**kw):
        legacy_up = kw.pop("legacy_up", True)
        v2_up = kw.pop("v2_up", True)
        main_mod, db_path = _make(monkeypatch, tmp_path, **kw)
        cm = TestClient(main_mod.app)
        c = cm.__enter__()
        opened.append(cm)
        c.app.state.pipeline = FakeLegacy() if legacy_up else None
        c.app.state.v2_pipeline = FakeV2() if v2_up else None
        return c, db_path

    yield factory
    for cm in opened:
        cm.__exit__(None, None, None)
    get_settings.cache_clear()
    from meno_rag.api import main as main_mod

    main_mod.get_settings.cache_clear()


def _post(c, kb=None, stream=False, token=None):
    body = {"model": "menon-1", "stream": stream, "messages": [{"role": "user", "content": "Когда сессия?"}]}
    if kb:
        body["knowledge_base_id"] = kb
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return c.post("/v1/chat/completions", json=body, headers=headers)


def _events(r):
    out = []
    for block in r.text.split("\n\n"):
        if not block.strip():
            continue
        ev = "data"
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if ev == "data":
            if data == "[DONE]":
                out.append("[DONE]")
            else:
                d = json.loads(data)["choices"][0]
                if d["delta"].get("role"):
                    out.append("role")
                elif "content" in d["delta"]:
                    out.append("token")
                else:
                    out.append("finish")
        else:
            out.append(ev)
    return out


def test_kb_list_per_flags(make_client, monkeypatch):
    c, _ = make_client()
    monkeypatch.setattr(c.app.state.settings, "v2_expose_as", "both")
    body = c.get("/v1/knowledge-bases").json()
    assert [(k["id"], k["name"], k["available"]) for k in body["data"]] == [
        (LEGACY_KB_ID, "База НГУ (июль 2026) · прежний поиск", True),
        (V2_KB_ID, "База НГУ (ноябрь 2026) · новый поиск", True),
    ]
    assert body["default_selection"]["knowledge_base_id"] == LEGACY_KB_ID


def test_kb_list_v2_only_and_legacy_only(make_client):
    c, _ = make_client(legacy=False)
    body = c.get("/v1/knowledge-bases").json()
    assert [k["id"] for k in body["data"]] == [V2_KB_ID]
    assert body["default_selection"]["knowledge_base_id"] == V2_KB_ID
    c2, _ = make_client(v2=False)
    assert [k["id"] for k in c2.get("/v1/knowledge-bases").json()["data"]] == [LEGACY_KB_ID]


def test_default_is_legacy_and_v2_by_id(make_client):
    c, _ = make_client()
    assert _post(c).json()["choices"][0]["message"]["content"] == "Старый"
    r = _post(c, kb=V2_KB_ID).json()
    assert r["choices"][0]["message"]["content"] == "Новый"
    assert r["sources"] == SRC


def test_v2_by_default_when_legacy_disabled(make_client):
    c, _ = make_client(legacy=False)
    assert _post(c).json()["choices"][0]["message"]["content"] == "Новый"


def test_unavailable_selected_kb_is_503_while_other_is_up(make_client):
    c, _ = make_client(v2_up=False)
    assert _post(c, kb=V2_KB_ID).status_code == 503
    assert _post(c).status_code == 200
    c2, _ = make_client(legacy_up=False)
    assert _post(c2).status_code == 503
    assert _post(c2, kb=V2_KB_ID).status_code == 200


def test_unknown_kb_is_400(make_client):
    c, _ = make_client(v2=False)
    r = _post(c, kb=V2_KB_ID)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "knowledge_base_id"


def test_v2_sse_order(make_client):
    c, _ = make_client()
    ev = _events(_post(c, kb=V2_KB_ID, stream=True))
    assert ev == [
        "stage", "role", "stage", "token", "token", "stage", "sources", "finish", "summary", "[DONE]",
    ]


def test_legacy_sse_order_unchanged(make_client):
    c, _ = make_client()
    ev = _events(_post(c, stream=True))
    assert ev == [
        "stage", "sources", "role", "stage", "token", "token", "stage", "finish", "summary", "[DONE]",
    ]


def test_bench_returns_v2_trace(make_client, monkeypatch):
    c, _ = make_client()
    r = _post(c, kb=V2_KB_ID, token=BENCH_TOKEN)
    assert r.json()["pipeline_trace"]["v2"]["route"] == "nsu"
    ev = _events(_post(c, kb=V2_KB_ID, stream=True, token=BENCH_TOKEN))
    assert ev[-2:] == ["pipeline_trace", "[DONE]"]


def test_db_row_has_v2_participant_label(make_client):
    c, db_path = make_client()
    assert _post(c, kb=V2_KB_ID).status_code == 200

    async def _read(session):
        return (await session.execute(text("SELECT knowledge_base_id FROM messages"))).scalars().all()

    # History is stored unconditionally (no improvement consent needed): user + assistant rows.
    assert with_db(db_path, _read) == ["menon-1.1@unversioned"] * 2
