"""Pydantic models for the meno-backend/1 wire contract."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

PROTOCOL = "meno-backend/1"
STANDARD_STAGES = frozenset(
    {
        "abbreviation_expansion",
        "anaphora_resolution",
        "query_rewrite",
        "retrieval",
        "fusion",
        "rerank",
        "context_assembly",
        "generation",
    }
)
MAX_DETAIL_LINES = 50
MAX_DETAIL_LINE_CHARS = 500
MAX_SOURCES = 20
# Gateway limits shared with the checker so that both judge a stand the same way.
MAX_BLOCK_BYTES = 1_000_000  # one SSE block; contexts and pipeline_trace may be larger
MAX_TRACE_BLOCK_BYTES = 16_000_000
MAX_ANSWER_CHARS = 40_000
MAX_CHUNK_PAUSE_S = 300  # the longest silence between two chunks of a stream
MAX_ANSWER_S = 900  # the longest whole answer
# Budgets of one stream: what passes them is ignored or ends the answer, so a stand cannot make one answer costly.
MAX_STREAM_BYTES = 64_000_000  # read from one stream, comments and thinking included; then the answer ends
MAX_STAGE_NAMES = 64  # distinct stage names in one answer; the others are ignored
MAX_LOGGED_VIOLATIONS = 5  # departures from the contract that the gateway logs per answer (all are counted)
MAX_LLMS = 32
MAX_TITLE_CHARS = 500
MAX_URL_CHARS = 2000

VERSION_PATTERN = r"^[A-Za-z0-9._+-]{1,64}$"
STAGE_NAME_PATTERN = r"^[a-z0-9_]{1,64}$"
_STAGE_NAME_RE = re.compile(STAGE_NAME_PATTERN)
LLM_ID_PATTERN = r"^[A-Za-z0-9._:/+@-]{1,128}$"


def _to_str(value: Any) -> Any:
    return str(value) if isinstance(value, int | float) and not isinstance(value, bool) else value


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StageLabel(_Model):
    ru: str = Field(min_length=1, max_length=128)
    en: str = Field(min_length=1, max_length=128)


class LlmStatus(_Model):
    state: Literal["available", "rate_limited", "unreachable"]
    until: str | None = None


class Llm(_Model):
    id: str = Field(pattern=LLM_ID_PATTERN)
    name: str = Field(min_length=1, max_length=128)
    context_length: int | None = None
    requires_auth: bool = False
    status: LlmStatus | None = None


class Manifest(_Model):
    protocol: str
    name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=256)
    version: str = Field(pattern=VERSION_PATTERN)
    build: str | None = Field(default=None, max_length=128)
    kb: str | None = Field(default=None, max_length=256)
    llms: list[Llm] = Field(max_length=MAX_LLMS)
    default_llm: str
    stages: dict[str, StageLabel] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Manifest:
        if self.protocol != PROTOCOL:
            raise ValueError(f"protocol must be {PROTOCOL}")
        if not self.llms:
            raise ValueError("llms must not be empty")
        ids = [m.id for m in self.llms]
        if len(set(ids)) != len(ids):
            raise ValueError("llm ids must be unique")
        if self.default_llm not in ids:
            raise ValueError("default_llm must be one of llms")
        for stage in self.stages:
            if not _STAGE_NAME_RE.fullmatch(stage):
                raise ValueError(f"bad stage name: {stage!r}")
        return self


class StageDetail(_Model):
    lines: list[str] = Field(default_factory=list)

    @field_validator("lines", mode="after")
    @classmethod
    def _trim(cls, lines: list[str]) -> list[str]:
        return [line[:MAX_DETAIL_LINE_CHARS] for line in lines[:MAX_DETAIL_LINES]]


class StageEvent(_Model):
    stage: str = Field(pattern=STAGE_NAME_PATTERN)
    status: Literal["started", "completed", "failed", "skipped"]
    duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    detail: StageDetail | None = None


class Context(_Model):
    id: Annotated[str, BeforeValidator(_to_str)]
    url: str = ""
    title: str = ""
    text: str
    rank: int | None = None


class ErrorEvent(_Model):
    code: str = "stand_error"
    message: str = ""
    retryable: bool = False
    retry_after_sec: int | None = None


def clean_sources(raw: Any) -> tuple[list[dict[str, str]], int]:
    """Reduce stand-provided sources to displayable rows.

    Returns the rows (at most MAX_SOURCES) and how many entries were dropped.
    """
    if not isinstance(raw, list):
        return [], 0
    rows: list[dict[str, str]] = []
    dropped = 0
    for item in raw:
        if not isinstance(item, dict) or len(rows) >= MAX_SOURCES:
            dropped += 1
            continue
        title = item.get("document_title")
        url = item.get("source_url")
        url = "" if url is None else str(url)
        if (url and not url.lower().startswith(("http://", "https://"))) or len(url) > MAX_URL_CHARS:
            dropped += 1
            continue
        row = {"document_title": ("" if title is None else str(title))[:MAX_TITLE_CHARS], "source_url": url}
        date = item.get("date")
        if date not in (None, ""):
            row["date"] = str(date)
        rows.append(row)
    return rows, dropped
