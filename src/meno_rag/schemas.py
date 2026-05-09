from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[str | list[str]] = None
    user: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    knowledge_base: Optional[str] = None

    model_config = {"extra": "allow"}


class ClearHistoryRequest(BaseModel):
    chat_id: str


class ClearHistoryResponse(BaseModel):
    chat_id: str
    status: str


class VoteRequest(BaseModel):
    model_a: str
    kb_a: str
    model_b: str
    kb_b: str
    winner: Literal["a", "b", "tie", "both_bad"]
    response_a: Optional[str] = None
    response_b: Optional[str] = None
    question: Optional[str] = None
    session_id: Optional[str] = None


class PipelineOutcome(BaseModel):
    question: str
    prepared_dialogue_history: str
    search_queries: list[str]
    context: str
    sources: list[dict[str, str]]
    qa_messages: list[dict[str, str]]
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    stage_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
