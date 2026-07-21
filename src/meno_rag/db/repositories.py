from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from meno_rag.db.orm import (
    ArenaRating,
    ArenaVote,
    ConsentEvent,
    Conversation,
    GenerationRecord,
    GuestSession,
    Message,
    MessageFeedback,
    PipelineRun,
    PipelineStageRun,
    SessionSurvey,
    SourceRecord,
    User,
)


async def ensure_conversation(
    session: AsyncSession,
    conversation_id: str,
    *,
    user_id: str | None = None,
    guest_session_id: str | None = None,
) -> Conversation:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        conversation = Conversation(id=conversation_id, user_id=user_id, guest_session_id=guest_session_id)
        session.add(conversation)
        await session.flush()
    if user_id is not None:
        conversation.user_id = user_id
    if guest_session_id is not None:
        conversation.guest_session_id = guest_session_id
    conversation.updated_at = datetime.now(UTC)
    return conversation


async def append_message(
    session: AsyncSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    model: str | None = None,
    knowledge_base_id: str | None = None,
    request_id: str | None = None,
) -> None:
    await ensure_conversation(session, conversation_id)
    session.add(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            knowledge_base_id=knowledge_base_id,
            request_id=request_id,
        )
    )


async def delete_conversation_cascade(session: AsyncSession, conversation_id: str) -> None:
    """Delete a conversation and every record linked to it (one transaction; caller commits).

    Only ``messages`` has a real FK to ``conversations`` (ON DELETE CASCADE). The
    pipeline_runs subtree, feedback, surveys, and arena votes are joined by the plain
    ``session_id`` string, so we delete them explicitly. Deleting a pipeline_run cascades
    to its stage runs / sources / generation record via their ``run_id`` FK.
    """
    await session.execute(delete(ArenaVote).where(ArenaVote.session_id == conversation_id))
    await session.execute(delete(MessageFeedback).where(MessageFeedback.session_id == conversation_id))
    await session.execute(delete(SessionSurvey).where(SessionSurvey.session_id == conversation_id))
    await session.execute(delete(PipelineRun).where(PipelineRun.session_id == conversation_id))
    await session.execute(delete(Conversation).where(Conversation.id == conversation_id))


async def clear_conversation(session: AsyncSession, conversation_id: str) -> None:
    """Deprecated alias — delegates to the full cascade deletion service."""
    await delete_conversation_cascade(session, conversation_id)


def conversation_owner_matches(
    conversation: Conversation, *, user_id: str | None, guest_session_id: str | None
) -> bool:
    """True if the caller may act on this conversation.

    User-owned → requires matching ``user_id``. Guest-owned → requires matching
    ``guest_session_id``. Untagged (legacy / pre-frontend-token) → allowed (transition
    policy; see the Stage 1b plan scope note).
    """
    if conversation.user_id is not None:
        return user_id is not None and conversation.user_id == user_id
    if conversation.guest_session_id is not None:
        return guest_session_id is not None and conversation.guest_session_id == guest_session_id
    return True


async def create_pipeline_run(
    session: AsyncSession,
    *,
    run_id: str,
    session_id: str,
    model: str,
    generation_model: str | None = None,
    core_model: str | None = None,
    endpoint: str | None,
    knowledge_base_id: str,
    user_question: str,
    search_queries: list[str] | None,
    total_ms: float | None,
    response_len: int | None,
    stream: bool,
    error: str | None = None,
    error_code: str | None = None,
    error_retryable: bool | None = None,
    error_stage: str | None = None,
) -> None:
    session.add(
        PipelineRun(
            id=run_id,
            session_id=session_id,
            model=model,
            generation_model=generation_model or model,
            core_model=core_model or model,
            endpoint=endpoint,
            knowledge_base_id=knowledge_base_id,
            user_question=user_question,
            search_queries=search_queries,
            total_ms=total_ms,
            response_len=response_len,
            stream=stream,
            error=error,
            error_code=error_code,
            error_retryable=error_retryable,
            error_stage=error_stage,
        )
    )


async def add_pipeline_stage(
    session: AsyncSession,
    *,
    run_id: str,
    stage: str,
    status: str,
    duration_ms: float | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    session.add(
        PipelineStageRun(
            run_id=run_id,
            stage=stage,
            status=status,
            duration_ms=duration_ms,
            detail=detail,
        )
    )


async def add_sources(session: AsyncSession, *, run_id: str, sources: list[dict[str, str]]) -> None:
    for idx, source in enumerate(sources):
        session.add(
            SourceRecord(
                run_id=run_id,
                document_title=source.get("document_title") or "",
                source_url=source.get("source_url") or "",
                ordinal=idx,
            )
        )


async def upsert_message_feedback(
    session: AsyncSession,
    *,
    run_id: str,
    session_id: str,
    value: str,
    comment: str | None = None,
    user_id: str | None = None,
) -> None:
    result = await session.execute(
        select(MessageFeedback).where(
            MessageFeedback.run_id == run_id,
            MessageFeedback.session_id == session_id,
        )
    )
    feedback = result.scalar_one_or_none()
    if feedback is None:
        session.add(
            MessageFeedback(
                run_id=run_id,
                session_id=session_id,
                value=value,
                comment=comment,
                user_id=user_id,
            )
        )
    else:
        feedback.value = value
        feedback.comment = comment
        if user_id is not None:
            feedback.user_id = user_id
        feedback.updated_at = datetime.now(UTC)


async def clear_message_feedback(session: AsyncSession, *, run_id: str, session_id: str) -> int:
    result = await session.execute(
        delete(MessageFeedback).where(
            MessageFeedback.run_id == run_id,
            MessageFeedback.session_id == session_id,
        )
    )
    # DELETE yields a CursorResult (has rowcount) at runtime; the async execute()
    # return type is the broader Result, so mypy needs the hint.
    return result.rowcount or 0  # type: ignore[attr-defined]


async def upsert_session_survey(
    session: AsyncSession,
    *,
    session_id: str,
    answer: str,
    user_id: str | None = None,
) -> None:
    result = await session.execute(select(SessionSurvey).where(SessionSurvey.session_id == session_id))
    survey = result.scalar_one_or_none()
    if survey is None:
        session.add(SessionSurvey(session_id=session_id, answer=answer, user_id=user_id))
    else:
        survey.answer = answer
        if user_id is not None:
            survey.user_id = user_id
        survey.updated_at = datetime.now(UTC)


async def create_generation_record(
    session: AsyncSession,
    *,
    run_id: str,
    system_prompt: str,
    user_prompt: str,
    raw_completion: str,
    dialogue_history: str | None = None,
    retrieved: list | dict | None = None,
    fewshots: list | dict | None = None,
    generation_params: list | dict | None = None,
) -> None:
    session.add(
        GenerationRecord(
            run_id=run_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            dialogue_history=dialogue_history,
            raw_completion=raw_completion,
            retrieved=retrieved,
            fewshots=fewshots,
            generation_params=generation_params,
        )
    )


INITIAL_ELO = 1200.0
K_FACTOR = 32.0


async def submit_arena_vote(session: AsyncSession, payload: dict[str, Any]) -> bool:
    """Record an arena vote, idempotent on (session_id, turn_index).

    Returns True if the vote was newly recorded, False if it was a duplicate.

    Multi-turn arena clients send (session_id, turn_index) — these together
    uniquely identify a vote round. A buggy or malicious client could
    re-POST the same vote multiple times (e.g. via stale closures, retries,
    or deliberate replay) and inflate the Elo store. We refuse second-and-
    later submissions for the same pair instead. Legacy single-turn payloads
    that omit turn_index keep their old behaviour (every POST counted).
    """
    sid = payload.get("session_id")
    tidx = payload.get("turn_index")
    if sid and tidx is not None:
        existing = await session.execute(
            select(ArenaVote.id).where(
                ArenaVote.session_id == sid,
                ArenaVote.turn_index == tidx,
            )
        )
        if existing.scalar_one_or_none() is not None:
            return False
    vote = ArenaVote(**payload)
    session.add(vote)
    await _apply_vote_to_ratings(session, payload)
    return True


async def list_arena_leaderboard(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.execute(select(ArenaRating).order_by(ArenaRating.elo.desc()))
    rows = []
    for rating in result.scalars().all():
        win_rate = (rating.wins / rating.matches * 100.0) if rating.matches else 0.0
        rows.append(
            {
                "model": rating.model,
                "knowledge_base": rating.knowledge_base,
                "elo": round(rating.elo),
                "matches": rating.matches,
                "wins": rating.wins,
                "losses": rating.losses,
                "ties": rating.ties,
                "both_bad": rating.both_bad,
                "win_rate": round(win_rate, 1),
            }
        )
    return rows


async def _get_rating(session: AsyncSession, model: str, knowledge_base: str) -> ArenaRating:
    result = await session.execute(
        select(ArenaRating).where(
            ArenaRating.model == model,
            ArenaRating.knowledge_base == knowledge_base,
        )
    )
    rating = result.scalar_one_or_none()
    if rating is None:
        rating = ArenaRating(model=model, knowledge_base=knowledge_base, elo=INITIAL_ELO)
        session.add(rating)
        await session.flush()
    return rating


async def _apply_vote_to_ratings(session: AsyncSession, vote: dict[str, Any]) -> None:
    rating_a = await _get_rating(session, vote["model_a"], vote["kb_a"])
    rating_b = await _get_rating(session, vote["model_b"], vote["kb_b"])

    elo_a = rating_a.elo
    elo_b = rating_b.elo
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    expected_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

    winner = vote["winner"]
    if winner == "a":
        score_a, score_b = 1.0, 0.0
        rating_a.wins += 1
        rating_b.losses += 1
    elif winner == "b":
        score_a, score_b = 0.0, 1.0
        rating_a.losses += 1
        rating_b.wins += 1
    elif winner == "tie":
        score_a, score_b = 0.5, 0.5
        rating_a.ties += 1
        rating_b.ties += 1
    elif winner == "both_bad":
        score_a, score_b = 0.0, 0.0
        rating_a.both_bad += 1
        rating_b.both_bad += 1
        rating_a.losses += 1
        rating_b.losses += 1
    else:
        return

    rating_a.matches += 1
    rating_b.matches += 1
    rating_a.elo = elo_a + K_FACTOR * (score_a - expected_a)
    rating_b.elo = elo_b + K_FACTOR * (score_b - expected_b)
    now = datetime.now(UTC)
    rating_a.updated_at = now
    rating_b.updated_at = now


async def create_user(session: AsyncSession, *, email: str, password_hash: str, nickname: str | None = None) -> User:
    user = User(email=email, password_hash=password_hash, nickname=nickname)
    session.add(user)
    await session.flush()
    return user


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: str) -> User | None:
    return await session.get(User, user_id)


async def update_user_nickname(session: AsyncSession, *, user_id: str, nickname: str) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")
    user.nickname = nickname
    user.updated_at = datetime.now(UTC)
    return user


async def create_guest_session(
    session: AsyncSession, *, secret_hash: str, ttl_days: int, now: datetime | None = None
) -> GuestSession:
    moment = now if now is not None else datetime.now(UTC)
    guest = GuestSession(
        secret_hash=secret_hash,
        created_at=moment,
        last_seen_at=moment,
        expires_at=moment + timedelta(days=ttl_days),
    )
    session.add(guest)
    await session.flush()
    return guest


async def get_guest_session_by_secret_hash(session: AsyncSession, secret_hash: str) -> GuestSession | None:
    result = await session.execute(select(GuestSession).where(GuestSession.secret_hash == secret_hash))
    return result.scalar_one_or_none()


async def touch_guest_session(
    session: AsyncSession, guest: GuestSession, *, ttl_days: int, now: datetime | None = None
) -> GuestSession:
    moment = now if now is not None else datetime.now(UTC)
    guest.last_seen_at = moment
    guest.expires_at = moment + timedelta(days=ttl_days)
    return guest


CONSENT_PURPOSES = ("SERVICE_AND_HISTORY", "ACCOUNT_REGISTRATION", "MENO_IMPROVEMENT")
CONSENT_ACTIONS = ("granted", "revoked")


async def record_consent_event(
    session: AsyncSession,
    *,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    purpose: str,
    action: str,
    document_kind: str,
    document_version: str,
    document_sha256: str,
    source: str,
) -> ConsentEvent:
    """Append a consent event. Exactly one of user_id/guest_session_id at creation.

    Owner columns are plain (no FK) and app-validated here — consistent with the
    initiative's deletion-service approach; the deletion service handles consent
    events on subject deletion.
    """
    if (user_id is None) == (guest_session_id is None):
        raise ValueError("Exactly one of user_id / guest_session_id must be set.")
    if purpose not in CONSENT_PURPOSES:
        raise ValueError(f"Unknown consent purpose: {purpose!r}")
    if action not in CONSENT_ACTIONS:
        raise ValueError(f"Unknown consent action: {action!r}")
    event = ConsentEvent(
        user_id=user_id,
        guest_session_id=guest_session_id,
        purpose=purpose,
        action=action,
        document_kind=document_kind,
        document_version=document_version,
        document_sha256=document_sha256,
        source=source,
    )
    session.add(event)
    await session.flush()
    return event


async def current_consent_state(
    session: AsyncSession, *, user_id: str | None = None, guest_session_id: str | None = None
) -> dict[str, bool]:
    """Resolve the latest granted/revoked action per purpose (append-only log)."""
    state = {purpose: False for purpose in CONSENT_PURPOSES}
    if user_id is not None:
        clause = ConsentEvent.user_id == user_id
    elif guest_session_id is not None:
        clause = ConsentEvent.guest_session_id == guest_session_id
    else:
        return state
    rows = (
        await session.execute(
            select(ConsentEvent.purpose, ConsentEvent.action)
            .where(clause)
            .order_by(ConsentEvent.created_at, ConsentEvent.id)
        )
    ).all()
    for purpose, action in rows:
        if purpose in state:
            state[purpose] = action == "granted"
    return state


async def list_contributor_leaderboard(session: AsyncSession) -> list[dict[str, Any]]:
    """Registered users ranked by arena votes + feedback given + questions asked.

    Exposes nickname only (never email); a null/empty nickname falls back to
    ``anon-<first 8 of id>``. Anonymous activity (user_id NULL) is excluded.
    """
    vote_counts = dict(
        (
            await session.execute(
                select(ArenaVote.user_id, func.count())
                .where(ArenaVote.user_id.is_not(None))
                .group_by(ArenaVote.user_id)
            )
        )
        .tuples()
        .all()
    )
    feedback_counts = dict(
        (
            await session.execute(
                select(MessageFeedback.user_id, func.count())
                .where(MessageFeedback.user_id.is_not(None))
                .group_by(MessageFeedback.user_id)
            )
        )
        .tuples()
        .all()
    )
    # The join holds by construction: _persist_success calls ensure_conversation
    # (conversations.id == session_id) before create_pipeline_run, so every run has
    # a matching conversation. There is no FK enforcing it — runs inserted outside
    # that path would be undercounted here.
    question_counts = dict(
        (
            await session.execute(
                select(Conversation.user_id, func.count(PipelineRun.id))
                .join(PipelineRun, PipelineRun.session_id == Conversation.id)
                .where(Conversation.user_id.is_not(None))
                .group_by(Conversation.user_id)
            )
        )
        .tuples()
        .all()
    )
    user_ids = set(vote_counts) | set(feedback_counts) | set(question_counts)
    if not user_ids:
        return []
    users = (await session.execute(select(User).where(User.id.in_(user_ids)))).scalars().all()
    rows: list[dict[str, Any]] = []
    for user in users:
        votes = int(vote_counts.get(user.id, 0))
        feedback = int(feedback_counts.get(user.id, 0))
        questions = int(question_counts.get(user.id, 0))
        rows.append(
            {
                "nickname": user.nickname or f"anon-{user.id[:8]}",
                "votes": votes,
                "feedback": feedback,
                "questions": questions,
                "total": votes + feedback + questions,
            }
        )
    rows.sort(key=lambda r: (-r["total"], r["nickname"]))
    return rows
