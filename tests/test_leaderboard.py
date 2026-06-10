# tests/test_leaderboard.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_contributor_leaderboard_counts_and_sort(tmp_path):
    from meno_rag.db import repositories
    from meno_rag.db.orm import ArenaVote, Conversation, MessageFeedback, PipelineRun, User

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'lb.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(User(id="u1", email="a@b.c", password_hash="h", nickname="Alice"))
            s.add(User(id="u2", email="d@e.f", password_hash="h", nickname=None))  # nickname fallback
            # u1: 2 votes, 1 feedback, 1 question
            s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id="u1"))
            s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="b", user_id="u1"))
            s.add(MessageFeedback(run_id="r1", session_id="c1", value="up", user_id="u1"))
            s.add(Conversation(id="c1", user_id="u1"))
            s.add(
                PipelineRun(id="r1", session_id="c1", model="m", knowledge_base_id="k", user_question="q", stream=False)
            )
            # u2: 1 vote only
            s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id="u2"))
            # an anonymous vote (no user) must not appear
            s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id=None))
            await s.commit()
        async with db.sessionmaker() as s:
            rows = await repositories.list_contributor_leaderboard(s)
    finally:
        await db.close()

    assert [r["nickname"] for r in rows] == ["Alice", "anon-u2"]  # sorted by total desc
    assert rows[0] == {"nickname": "Alice", "votes": 2, "feedback": 1, "questions": 1, "total": 4}
    assert rows[1]["votes"] == 1 and rows[1]["total"] == 1
    assert all("email" not in r for r in rows)  # privacy: nickname only
