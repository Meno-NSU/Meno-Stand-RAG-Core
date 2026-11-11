"""0020 replaces the click log (feedback_events) with the comment history (feedback_comments): every distinct comment
text found in the log is kept once, dated by its first appearance, and nothing else carries over (the current
ratings and survey answers stay in message_feedback / session_surveys). The downgrade rebuilds the log from the
current values and the kept comments."""

from __future__ import annotations

from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect, text

from alembic import command as alembic_command
from tests.test_migrate import REPO_ALEMBIC_INI

EVENTS = [
    # kind, action, run_id, session_id, user_id, guest_session_id, value, comment, created_at
    ("rating", "snapshot", "r1", "s1", None, "g1", "down", "wrong", "2026-11-05"),
    ("survey", "snapshot", None, "s1", None, "g1", "yes", None, "2026-11-06"),
    ("rating", "set", "r1", "s1", None, "g1", "up", "wrong", "2026-11-11 08:00"),  # same text again
    ("rating", "clear", "r1", "s1", None, "g1", None, None, "2026-11-11 08:01"),
    ("rating", "set", "r1", "s1", None, "g1", "down", "wrong year", "2026-11-11 08:02"),  # rewritten
    ("rating", "set", "r2", "s1", None, "g1", "up", None, "2026-11-11 08:03"),
    ("rating", "set", "r2", "s1", None, "g1", "up", "  ", "2026-11-11 08:04"),  # blank
    ("rating", "set", "r3", "s2", "u1", None, "down", "wrong", "2026-11-11 08:05"),  # same text, another answer
    ("survey", "set", None, "s1", None, "g1", "no", None, "2026-11-11 08:06"),
    ("rating", "set", "r3", "s2", None, "g9", "up", "wrong", "2026-11-11 08:07"),  # same text, someone else
]


def _cfg(url: str) -> AlembicConfig:
    cfg = AlembicConfig(str(REPO_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def test_0020_keeps_every_comment_from_the_log_and_round_trips(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    alembic_command.upgrade(_cfg(url), "0019_keep_user_data")
    engine = create_engine(url)
    try:
        with engine.begin() as c:
            c.execute(text("DELETE FROM feedback_events"))
            for row in EVENTS:
                c.execute(
                    text(
                        "INSERT INTO feedback_events "
                        "(kind, action, run_id, session_id, user_id, guest_session_id, value, comment, created_at) "
                        "VALUES (:k, :a, :r, :s, :u, :g, :v, :c, :t)"
                    ),
                    dict(zip("karsugvct", row, strict=True)),
                )
            c.execute(
                text(
                    "INSERT INTO message_feedback (id, run_id, session_id, user_id, guest_session_id, value, comment, created_at, updated_at) "
                    "VALUES ('f1', 'r1', 's1', NULL, 'g1', 'down', 'wrong year', '2026-11-01', '2026-11-11 08:02:30')"
                )
            )
            # Written by code older than 0019 while the database was already at 0019 (a rollback): not in the log.
            c.execute(
                text(
                    "INSERT INTO message_feedback (id, run_id, session_id, user_id, guest_session_id, value, comment, created_at, updated_at) "
                    "VALUES ('f2', 'r4', 's3', 'u2', NULL, 'up', 'older code', '2026-11-01', '2026-11-10')"
                )
            )
            c.execute(
                text(
                    "INSERT INTO session_surveys (id, session_id, user_id, guest_session_id, answer, created_at, updated_at) "
                    "VALUES ('v1', 's1', NULL, 'g1', 'no', '2026-11-01', '2026-11-11 08:06')"
                )
            )

        alembic_command.upgrade(_cfg(url), "head")

        names = inspect(engine)
        assert "feedback_events" not in names.get_table_names()
        with engine.connect() as c:
            kept = c.execute(
                text(
                    "SELECT run_id, session_id, user_id, guest_session_id, value, comment, created_at "
                    "FROM feedback_comments ORDER BY id"
                )
            ).all()
        assert [tuple(k)[:6] for k in kept] == [
            ("r1", "s1", None, "g1", "down", "wrong"),  # first seen in the snapshot, with the rating it had then
            ("r1", "s1", None, "g1", "down", "wrong year"),
            ("r3", "s2", "u1", None, "down", "wrong"),
            ("r3", "s2", None, "g9", "up", "wrong"),
            ("r4", "s3", "u2", None, "up", "older code"),  # from the current value; f1's text is in the log already
        ]
        assert [str(k[6])[:16] for k in kept] == [
            "2026-11-05",
            "2026-11-11 08:02",
            "2026-11-11 08:05",
            "2026-11-11 08:07",
            "2026-11-10",
        ]

        alembic_command.downgrade(_cfg(url), "0019_keep_user_data")
        names = inspect(engine)
        assert "feedback_comments" not in names.get_table_names()
        with engine.connect() as c:
            log = c.execute(text("SELECT kind, action, run_id, value, comment FROM feedback_events")).all()
        assert sorted((tuple(e) for e in log), key=repr) == sorted(
            [
                ("rating", "set", "r1", "down", "wrong"),
                ("rating", "set", "r1", "down", "wrong year"),
                ("rating", "snapshot", "r1", "down", "wrong year"),  # the current ratings
                ("rating", "set", "r3", "down", "wrong"),
                ("rating", "set", "r3", "up", "wrong"),
                ("rating", "set", "r4", "up", "older code"),
                ("rating", "snapshot", "r4", "up", "older code"),
                ("survey", "snapshot", None, "no", None),
            ],
            key=repr,
        )

        alembic_command.upgrade(_cfg(url), "head")  # and up again: nothing is doubled
        with engine.connect() as c:
            assert c.execute(text("SELECT count(*) FROM feedback_comments")).scalar_one() == 5
    finally:
        engine.dispose()
