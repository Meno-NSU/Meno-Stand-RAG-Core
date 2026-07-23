from typing import Annotated, Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


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

    # Set by the arena UI. Both sides share one session_id, so letting each side persist
    # itself would write the question twice and two assistant rows in a racing order.
    # The completed comparison is posted once to /v1/arena/turn instead.
    arena: bool = False

    model_config = {"extra": "allow"}


class ClearHistoryRequest(BaseModel):
    chat_id: str


class ClearHistoryResponse(BaseModel):
    chat_id: str
    status: str


class SourceRef(BaseModel):
    document_title: str
    source_url: str


class TurnFeedback(BaseModel):
    rating: Literal["up", "down"]
    comment: str | None = None


class UserTurn(BaseModel):
    kind: Literal["user"] = "user"
    content: str
    created_at: str


class AnswerTurn(BaseModel):
    kind: Literal["answer"] = "answer"
    content: str
    created_at: str
    model: str | None = None
    request_id: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    feedback: TurnFeedback | None = None


class ArenaTurnSide(BaseModel):
    key: str
    model: str | None = None
    knowledge_base_id: str | None = None
    content: str
    sources: list[SourceRef] = Field(default_factory=list)


class ArenaTurn(BaseModel):
    kind: Literal["arena"] = "arena"
    # Side A's answer, mirrored from the stored row's NOT NULL `content` column (see
    # append_arena_turn) — a generic consumer that doesn't special-case "arena" turns still
    # gets a sensible string here. The rendered comparison itself comes from `sides`; a
    # client that understands arena turns renders from there, not from this field.
    content: str
    created_at: str
    winner: Literal["a", "b", "tie", "both_bad"] | None = None
    sides: list[ArenaTurnSide] = Field(default_factory=list)


ConversationTurn = Annotated[UserTurn | AnswerTurn | ArenaTurn, Field(discriminator="kind")]


class SurveyAnswer(BaseModel):
    answer: Literal["yes", "maybe", "no", "skipped"]


class ConversationResponse(BaseModel):
    id: str
    survey: SurveyAnswer | None = None
    turns: list[ConversationTurn]


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


class ArenaSide(BaseModel):
    key: Literal["a", "b"]
    model: str | None = None
    knowledge_base_id: str | None = None
    content: str
    request_id: str | None = None
    sources: list[dict[str, str]] = Field(default_factory=list)


class ArenaTurnRequest(BaseModel):
    """A finished side-by-side comparison, posted once after both sides answer."""

    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    turn_index: int | None = Field(default=None, ge=0)
    sides: list[ArenaSide] = Field(..., min_length=2, max_length=2)


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
    password: str = Field(..., min_length=8)
    nickname: str | None = Field(default=None, max_length=64)

    @field_validator("password")
    @classmethod
    def _password_within_bcrypt_limit(cls, value: str) -> str:
        # bcrypt only uses the first 72 bytes; reject longer input with a clear
        # error instead of silently truncating it (ТЗ §5.9). Byte length, not
        # characters — a 36-char Cyrillic password is already 72 bytes.
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 bytes.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class NicknameRequest(BaseModel):
    nickname: str = Field(..., min_length=1, max_length=64)


class PrivacySettingsPatch(BaseModel):
    document_version: str
    service_and_history: bool
    meno_improvement: bool
    source: str | None = None


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
    trace: dict | None = None
