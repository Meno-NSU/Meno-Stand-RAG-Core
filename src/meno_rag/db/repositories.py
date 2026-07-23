from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select, update
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
    analysis_allowed: bool | None = None,
) -> Conversation:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            guest_session_id=guest_session_id,
            analysis_allowed=bool(analysis_allowed),
        )
        session.add(conversation)
        await session.flush()
    if user_id is not None:
        conversation.user_id = user_id
    if guest_session_id is not None:
        conversation.guest_session_id = guest_session_id
    # Only touch analysis_allowed when explicitly provided, so append_message's
    # bare ensure_conversation call doesn't reset the flag the caller just set.
    if analysis_allowed is not None:
        conversation.analysis_allowed = analysis_allowed
    conversation.updated_at = datetime.now(UTC)
    return conversation


def shown_source_refs(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    """Only the title and the link — the fields Цель 1 (сервисная обработка) covers.

    Durable conversation records are written under the service consent alone, so they must
    never accumulate retrieval content such as chunk text or relevance scores, which Цель 3
    gates. Applied at the storage boundary so no caller can bypass it.
    """
    return [
        {
            "document_title": source.get("document_title") or "",
            "source_url": source.get("source_url") or "",
        }
        for source in sources
    ]


async def append_message(
    session: AsyncSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    model: str | None = None,
    knowledge_base_id: str | None = None,
    request_id: str | None = None,
    sources: list[dict[str, str]] | None = None,
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
            sources=None if sources is None else shown_source_refs(sources),
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


async def delete_subject_data(
    session: AsyncSession, *, user_id: str | None = None, guest_session_id: str | None = None
) -> None:
    """Erase everything tied to a subject (152-ФЗ right to erasure). Exactly one id.

    Deletes every conversation the subject owns (each via delete_conversation_cascade),
    their consent events, and the subject row itself — the account for a registered user
    (their JWT then resolves to no user) or the guest_session for a guest.

    Both branches also sweep records tagged with the subject's own id that the per-
    conversation cascade above would not reach — most commonly feedback left on a
    conversation the subject does not own (an untagged/legacy one, which the write-side
    ownership policy in api/feedback.py still lets anyone rate; see
    conversation_owner_matches). The registered-user branch reaches further only because
    ArenaVote and SessionSurvey carry a user_id column with no guest_session_id
    counterpart: for a user we sweep ArenaVote, MessageFeedback, and SessionSurvey by
    user_id; for a guest we can only sweep MessageFeedback by guest_session_id — an arena
    vote or survey answer a guest left on a conversation they don't own has no guest column
    to match here and is not reachable by this function. Aggregate Elo (arena_ratings) is
    anonymous and stays regardless.
    """
    if (user_id is None) == (guest_session_id is None):
        raise ValueError("Exactly one of user_id / guest_session_id must be set.")

    if user_id is not None:
        conv_clause = Conversation.user_id == user_id
    else:
        conv_clause = Conversation.guest_session_id == guest_session_id
    conversation_ids = (await session.execute(select(Conversation.id).where(conv_clause))).scalars().all()
    for conversation_id in conversation_ids:
        await delete_conversation_cascade(session, conversation_id)

    if user_id is not None:
        await session.execute(delete(ArenaVote).where(ArenaVote.user_id == user_id))
        await session.execute(delete(MessageFeedback).where(MessageFeedback.user_id == user_id))
        await session.execute(delete(SessionSurvey).where(SessionSurvey.user_id == user_id))
        await session.execute(delete(ConsentEvent).where(ConsentEvent.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))
    else:
        await session.execute(delete(MessageFeedback).where(MessageFeedback.guest_session_id == guest_session_id))
        await session.execute(delete(ConsentEvent).where(ConsentEvent.guest_session_id == guest_session_id))
        await session.execute(delete(GuestSession).where(GuestSession.id == guest_session_id))


async def list_subject_conversations(
    session: AsyncSession, *, user_id: str | None = None, guest_session_id: str | None = None
) -> list[dict]:
    """The subject's conversations, newest first, each with a short preview (its first user
    message) so a client can render the chat list without loading every message."""
    if user_id is not None:
        clause = Conversation.user_id == user_id
    elif guest_session_id is not None:
        clause = Conversation.guest_session_id == guest_session_id
    else:
        return []
    conversations = (
        (await session.execute(select(Conversation).where(clause).order_by(Conversation.updated_at.desc())))
        .scalars()
        .all()
    )
    items: list[dict] = []
    for conversation in conversations:
        preview = (
            await session.execute(
                select(Message.content)
                .where(Message.conversation_id == conversation.id, Message.role == "user")
                .order_by(Message.created_at)
                .limit(1)
            )
        ).scalar_one_or_none()
        items.append({"id": conversation.id, "updated_at": conversation.updated_at, "preview": (preview or "")[:80]})
    return items


async def set_subject_conversations_analysis_allowed(
    session: AsyncSession, *, user_id: str | None = None, guest_session_id: str | None = None, allowed: bool
) -> int:
    """Flip ``analysis_allowed`` on all of a subject's existing conversations; returns the
    count updated. Backs retroactive consent: granting MENO_IMPROVEMENT makes already-stored
    dialogues analysis-eligible, revoking it takes them back out (symmetric). Exactly one id."""
    if (user_id is None) == (guest_session_id is None):
        raise ValueError("Exactly one of user_id / guest_session_id must be set.")
    clause = (
        Conversation.user_id == user_id if user_id is not None else Conversation.guest_session_id == guest_session_id
    )
    ids = list((await session.execute(select(Conversation.id).where(clause))).scalars().all())
    if ids:
        await session.execute(update(Conversation).where(clause).values(analysis_allowed=allowed))
    return len(ids)


async def get_conversation_messages(session: AsyncSession, conversation_id: str) -> list[Message]:
    result = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    )
    return list(result.scalars().all())


async def delete_conversations_older_than(session: AsyncSession, *, cutoff: datetime) -> int:
    """Delete conversations (and their cascade) not updated since ``cutoff``; returns the count.
    Backs the retention CLI (152-ФЗ storage limitation)."""
    ids = (await session.execute(select(Conversation.id).where(Conversation.updated_at < cutoff))).scalars().all()
    for conversation_id in ids:
        await delete_conversation_cascade(session, conversation_id)
    return len(ids)


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
    for idx, source in enumerate(shown_source_refs(sources)):
        session.add(
            SourceRecord(
                run_id=run_id,
                document_title=source["document_title"],
                source_url=source["source_url"],
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
    guest_session_id: str | None = None,
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
                guest_session_id=guest_session_id,
            )
        )
    else:
        feedback.value = value
        feedback.comment = comment
        if user_id is not None:
            feedback.user_id = user_id
        if guest_session_id is not None:
            feedback.guest_session_id = guest_session_id
        feedback.updated_at = datetime.now(UTC)


async def clear_message_feedback(
    session: AsyncSession,
    *,
    run_id: str,
    session_id: str,
    user_id: str | None = None,
    guest_session_id: str | None = None,
) -> int:
    """Delete the caller's own rating for (run_id, session_id).

    Scoped by caller identity with the same precedence get_conversation_feedback uses for
    reads — user_id when authenticated, else guest_session_id — so one guest cannot delete
    another guest's rating merely because conversation_owner_matches lets both of them write
    into the same untagged (legacy) conversation: they could not read it, but before this
    scoping they could still destroy it. A fully anonymous caller (no user_id, no
    guest_session_id — no X-Guest-Token was ever presented) matches on (run_id, session_id)
    alone, same as always: there is no narrower identity to scope to, and untagged feedback
    predates guest_session_id entirely.
    """
    clause = [MessageFeedback.run_id == run_id, MessageFeedback.session_id == session_id]
    if user_id is not None:
        clause.append(MessageFeedback.user_id == user_id)
    elif guest_session_id is not None:
        clause.append(MessageFeedback.guest_session_id == guest_session_id)
    result = await session.execute(delete(MessageFeedback).where(*clause))
    # DELETE yields a CursorResult (has rowcount) at runtime; the async execute()
    # return type is the broader Result, so mypy needs the hint.
    return result.rowcount or 0  # type: ignore[attr-defined]


async def get_conversation_feedback(
    session: AsyncSession, *, conversation_id: str, user_id: str | None = None, guest_session_id: str | None = None
) -> dict[str, dict[str, str | None]]:
    """The caller's ratings in this conversation, keyed by run_id.

    The write path (`/v1/feedback`) now checks conversation ownership before every upsert
    (see `conversation_owner_matches` and its callers in `api/feedback.py`), so a row tagged
    with a given `user_id`/`guest_session_id` was in fact written by that subject — this is
    no longer a partial mitigation standing in for a missing boundary, it is a real per-
    subject scope. A signed-in caller sees only rows tagged with their own `user_id`; a
    guest sees only rows tagged with their own `guest_session_id` — two different guests are
    now distinguishable, unlike the old fallback to `user_id IS NULL`, which was every
    guest's row at once. With neither id (no authenticated user, no guest session) there is
    no subject to scope to, so nothing is returned.
    """
    if user_id is not None:
        clause = MessageFeedback.user_id == user_id
    elif guest_session_id is not None:
        clause = MessageFeedback.guest_session_id == guest_session_id
    else:
        return {}
    rows = (
        (await session.execute(select(MessageFeedback).where(MessageFeedback.session_id == conversation_id, clause)))
        .scalars()
        .all()
    )
    return {row.run_id: {"rating": row.value, "comment": row.comment} for row in rows}


async def get_session_survey(session: AsyncSession, *, conversation_id: str) -> dict[str, str] | None:
    """The end-of-session survey answer for this conversation, or None if unanswered.

    Unlike get_conversation_feedback, this takes no user_id and applies no per-subject
    scoping — deliberately, not an oversight. `SessionSurvey` carries
    `UniqueConstraint("session_id")`, so the answer is a property of the conversation
    itself, not of a subject, and the only caller (`get_conversation`) has already passed
    the conversation ownership check before this runs.
    """
    survey = (
        await session.execute(select(SessionSurvey).where(SessionSurvey.session_id == conversation_id))
    ).scalar_one_or_none()
    return None if survey is None else {"answer": survey.answer}


async def append_arena_turn(
    session: AsyncSession,
    *,
    conversation_id: str,
    question: str,
    sides: list[dict],
    turn_index: int | None = None,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    analysis_allowed: bool = False,
) -> None:
    """Store a comparison as one user row plus one assistant row, idempotent on
    (conversation_id, turn_index).

    A retry, a double-fired client effect, or two tabs replaying the same comparison must not
    create a second (user, assistant) pair — that would reintroduce the duplicated-question,
    broken-alternation bug this feature exists to fix. When `turn_index` is not None and a
    stored `turn_kind="arena"` row already carries that `turn_index` for this conversation, it
    is updated in place instead of appending a second pair (mirrors upsert_message_feedback's
    insert-or-update shape). The existing user row is left untouched — a repost of the same
    turn carries the same question, and the invariant this closes is "no duplicate rows", not
    "keep the question text in sync".

    `turn_index=None` never matches an existing turn and always appends a fresh pair. Matching
    None against another None would make every turn_index-less post collide with every other
    one on the conversation — worse than the duplicate this guards against for the clients
    that never send it.

    If an ArenaVote for (conversation_id, turn_index) already exists — a vote that raced ahead
    of its own turn — its winner is carried onto the stored turn here: the same value
    set_arena_turn_winner would have written had the turn existed first. Skipped when
    turn_index is None for the same collision reason as above (and because
    set_arena_turn_winner/submit_arena_vote's own dedupe already treat a missing turn_index as
    unmatchable).

    `content` on the assistant row is side A's answer: the column is NOT NULL and previews and
    exports need text, while `winner` may be "tie" or "both_bad", so "the winning answer" is not
    always defined. Clients render from `arena["sides"]`.
    """
    await ensure_conversation(
        session,
        conversation_id,
        user_id=user_id,
        guest_session_id=guest_session_id,
        analysis_allowed=analysis_allowed,
    )
    # Same allow-list as an ordinary answer: this row is written under the service consent,
    # so a side's sources may carry only the title and the link (Цель 1).
    stored_sides = [{**side, "sources": shown_source_refs(side.get("sources") or [])} for side in sides]

    existing: Message | None = None
    winner: str | None = None
    if turn_index is not None:
        # Matched in Python rather than queried inside the JSON column, same reasoning as
        # set_arena_turn_winner: SQLite and PostgreSQL spell JSON queries differently, and on
        # PostgreSQL the live column is `json`, not `jsonb`, so it has no equality operator to
        # query with. A conversation holds few arena turns.
        rows = (
            (
                await session.execute(
                    select(Message).where(
                        Message.conversation_id == conversation_id, Message.turn_kind == "arena"
                    )
                )
            )
            .scalars()
            .all()
        )
        existing = next((row for row in rows if (row.arena or {}).get("turn_index") == turn_index), None)
        winner = (
            await session.execute(
                select(ArenaVote.winner).where(
                    ArenaVote.session_id == conversation_id,
                    ArenaVote.turn_index == turn_index,
                )
            )
        ).scalar_one_or_none()

    if existing is not None:
        # Reassign rather than mutate: a plain JSON column is not change-tracked, so an
        # in-place `existing.arena["sides"] = ...` would never reach the database.
        existing.content = stored_sides[0]["content"]
        existing.arena = {"turn_index": turn_index, "winner": winner, "sides": stored_sides}
        return

    await append_message(session, conversation_id=conversation_id, role="user", content=question)
    session.add(
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content=stored_sides[0]["content"],
            turn_kind="arena",
            arena={"turn_index": turn_index, "winner": winner, "sides": stored_sides},
        )
    )


async def set_arena_turn_winner(
    session: AsyncSession, *, conversation_id: str, turn_index: int | None, winner: str
) -> bool:
    """Mark which side won a stored comparison. Returns False if no such turn exists.

    The turn is matched in Python rather than by querying inside the JSON column, which SQLite
    and PostgreSQL spell differently — and on PostgreSQL the live column is `json`, not `jsonb`,
    so it has no equality operator to query with. A conversation holds few arena turns.
    """
    if turn_index is None:
        return False
    rows = (
        (
            await session.execute(
                select(Message).where(Message.conversation_id == conversation_id, Message.turn_kind == "arena")
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        stored = row.arena or {}
        if stored.get("turn_index") == turn_index:
            # Reassign rather than mutate: a plain JSON column is not change-tracked, so an
            # in-place `stored["winner"] = winner` would never reach the database.
            row.arena = {**stored, "winner": winner}
            return True
    return False


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
    state = dict.fromkeys(CONSENT_PURPOSES, False)
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
