# tests/test_repositories_users.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_user_crud(tmp_path):
    from meno_rag.db import repositories

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'u.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            user = await repositories.create_user(s, email="a@b.c", password_hash="h", nickname="Al")
            await s.commit()
            uid = user.id
        async with db.sessionmaker() as s:
            by_email = await repositories.get_user_by_email(s, "a@b.c")
            by_id = await repositories.get_user_by_id(s, uid)
            assert by_email is not None and by_email.id == uid
            assert by_id is not None and by_id.email == "a@b.c"
        async with db.sessionmaker() as s:
            updated = await repositories.update_user_nickname(s, user_id=uid, nickname="Alice")
            await s.commit()
            assert updated.nickname == "Alice"
        async with db.sessionmaker() as s:
            assert await repositories.get_user_by_email(s, "missing@x.y") is None
    finally:
        await db.close()
