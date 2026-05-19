"""Schema-level guards on /v1/arena/vote payloads.

A frontend bug shouldn't be able to poison the Elo leaderboard with empty
or null model identifiers. Pydantic rejects missing fields with 422
out of the box, but empty strings would slip through plain `str` typing —
explicit `min_length=1` closes that hole."""

import pytest
from pydantic import ValidationError

from meno_rag.schemas import VoteRequest


def _valid_payload(**overrides):
    base = {
        "model_a": "vllm/menon-1",
        "kb_a": "nsu-stand-faiss-bm25",
        "model_b": "openrouter/qwen-2.5-72b-instruct:free",
        "kb_b": "nsu-stand-faiss-bm25",
        "winner": "a",
    }
    base.update(overrides)
    return base


def test_valid_payload_parses():
    vote = VoteRequest(**_valid_payload())
    assert vote.winner == "a"
    assert vote.model_a == "vllm/menon-1"


@pytest.mark.parametrize("field", ["model_a", "model_b", "kb_a", "kb_b"])
def test_empty_string_rejected(field):
    with pytest.raises(ValidationError):
        VoteRequest(**_valid_payload(**{field: ""}))


@pytest.mark.parametrize("field", ["model_a", "model_b", "kb_a", "kb_b"])
def test_null_rejected(field):
    with pytest.raises(ValidationError):
        VoteRequest(**_valid_payload(**{field: None}))


def test_invalid_winner_value_rejected():
    with pytest.raises(ValidationError):
        VoteRequest(**_valid_payload(winner="maybe"))


def test_optional_metadata_defaults_to_none():
    vote = VoteRequest(**_valid_payload())
    assert vote.turn_index is None
    assert vote.history_len_a is None
    assert vote.history_len_b is None


def test_optional_metadata_accepted_when_present():
    vote = VoteRequest(
        **_valid_payload(turn_index=2, history_len_a=4, history_len_b=2)
    )
    assert vote.turn_index == 2
    assert vote.history_len_a == 4
    assert vote.history_len_b == 2


def test_optional_metadata_rejects_non_int():
    with pytest.raises(ValidationError):
        VoteRequest(**_valid_payload(turn_index="two"))
