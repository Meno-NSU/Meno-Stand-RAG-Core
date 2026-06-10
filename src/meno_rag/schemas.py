from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    user: str | None = None
    knowledge_base_id: str | None = None
    knowledge_base: str | None = None

    model_config = {"extra": "allow"}


class ClearHistoryRequest(BaseModel):
    chat_id: str


class ClearHistoryResponse(BaseModel):
    chat_id: str
    status: str


class VoteRequest(BaseModel):
    # Empty model/kb strings are nonsensical for arena (they make leaderboard
    # rows that can never be matched again) — reject at the schema layer so
    # a frontend bug can't silently poison the Elo store.
    model_a: str = Field(..., min_length=1)
    kb_a: str = Field(..., min_length=1)
    model_b: str = Field(..., min_length=1)
    kb_b: str = Field(..., min_length=1)
    winner: Literal["a", "b", "tie", "both_bad"]
    response_a: str | None = None
    response_b: str | None = None
    question: str | None = None
    session_id: str | None = None
    turn_index: int | None = Field(default=None, ge=0)
    history_len_a: int | None = Field(default=None, ge=0)
    history_len_b: int | None = Field(default=None, ge=0)


class FeedbackRequest(BaseModel):
    completion_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    value: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackClearRequest(BaseModel):
    completion_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class SurveyRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    answer: Literal["yes", "maybe", "no", "skipped"]


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    nickname: str | None = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class NicknameRequest(BaseModel):
    nickname: str = Field(..., min_length=1, max_length=64)


class PipelineOutcome(BaseModel):
    question: str
    prepared_dialogue_history: str
    search_queries: list[str]
    context: str
    sources: list[dict[str, str]]
    qa_messages: list[dict[str, str]]
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    stage_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    fewshots: list[dict[str, Any]] = Field(default_factory=list)
