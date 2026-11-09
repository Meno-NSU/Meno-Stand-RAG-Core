import pytest
from meno_contract.models import PROTOCOL, Context, Manifest, StageEvent, clean_sources
from pydantic import ValidationError

GOOD = {
    "protocol": PROTOCOL,
    "name": "RAGU",
    "version": "2026-11-09-kb36",
    "llms": [{"id": "gemma", "name": "Gemma 4"}],
    "default_llm": "gemma",
    "stages": {"graph_walk": {"ru": "Обход графа", "en": "Graph walk"}},
}


def test_manifest_accepts_the_minimum_and_ignores_unknown_fields():
    m = Manifest.model_validate({**GOOD, "future": 1})
    assert m.version == "2026-11-09-kb36" and m.llms[0].requires_auth is False


@pytest.mark.parametrize(
    "patch",
    [
        {"protocol": "meno-backend/2"},
        {"version": "with space"},
        {"version": "x" * 65},
        {"llms": []},
        {"default_llm": "absent"},
        {"llms": [{"id": "a", "name": "A"}, {"id": "a", "name": "B"}], "default_llm": "a"},
        {"stages": {"Bad-Name": {"ru": "x", "en": "x"}}},
    ],
)
def test_manifest_rejects(patch):
    with pytest.raises(ValidationError):
        Manifest.model_validate({**GOOD, **patch})


def test_stage_detail_is_trimmed_to_the_limits():
    e = StageEvent.model_validate({"stage": "retrieval", "status": "completed", "detail": {"lines": ["я" * 600] * 60}})
    assert len(e.detail.lines) == 50 and len(e.detail.lines[0]) == 500


def test_stage_detail_without_lines_is_empty():
    e = StageEvent.model_validate({"stage": "rerank", "status": "started", "detail": {"round": 2}})
    assert e.detail.lines == []


@pytest.mark.parametrize("patch", [{"status": "done"}, {"stage": "Bad Name"}, {"duration_ms": -1}])
def test_stage_event_rejects(patch):
    with pytest.raises(ValidationError):
        StageEvent.model_validate({"stage": "rerank", "status": "started", **patch})


def test_clean_sources_keeps_only_what_can_be_shown():
    rows, dropped = clean_sources(
        [
            {"document_title": "График", "source_url": "https://nsu.ru/g", "date": None, "score": 0.9},
            {"document_title": 7, "source_url": "", "date": "2026-01-31"},
            {"document_title": "x", "source_url": "javascript:alert(1)"},
            "мусор",
        ]
    )
    assert rows == [
        {"document_title": "График", "source_url": "https://nsu.ru/g"},
        {"document_title": "7", "source_url": "", "date": "2026-01-31"},
    ]
    assert dropped == 2


def test_clean_sources_caps_the_count_and_survives_a_non_list():
    rows, dropped = clean_sources([{"document_title": "t", "source_url": ""}] * 25)
    assert len(rows) == 20 and dropped == 5
    assert clean_sources(None) == ([], 0)


def test_context_id_becomes_a_string():
    assert Context.model_validate({"id": 12, "text": "т"}).id == "12"
