"""0019 starts the feedback history with the current values (oldest first) and marks every existing answer as not
interrupted; the downgrade takes the schema back to 0018."""

from __future__ import annotations

from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect, text

from alembic import command as alembic_command
from tests.test_migrate import REPO_ALEMBIC_INI


def _cfg(url: str) -> AlembicConfig:
    cfg = AlembicConfig(str(REPO_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def test_0019_snapshots_current_feedback_and_round_trips(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    alembic_command.upgrade(_cfg(url), "0018_pipeline_run_stand_build")
    engine = create_engine(url)
    try:
        with engine.begin() as c:
            c.execute(
                text(
                    "INSERT INTO conversations (id, analysis_allowed, created_at, updated_at) VALUES ('s1', 0, '2026-11-01', '2026-11-01')"
                )
            )
            c.execute(
                text(
                    "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES ('m1', 's1', 'assistant', 'A', '2026-11-01')"
                )
            )
            c.execute(
                text(
                    "INSERT INTO message_feedback (id, run_id, session_id, user_id, guest_session_id, value, comment, created_at, updated_at) "
                    "VALUES ('f1', 'r1', 's1', NULL, 'g1', 'down', 'wrong', '2026-11-01', '2026-11-05')"
                )
            )
            c.execute(
                text(
                    "INSERT INTO session_surveys (id, session_id, user_id, guest_session_id, answer, created_at, updated_at) "
                    "VALUES ('v1', 's1', 'u1', NULL, 'yes', '2026-11-01', '2026-11-03')"
                )
            )

        alembic_command.upgrade(_cfg(url), "0019_keep_user_data")

        with engine.connect() as c:
            events = c.execute(
                text(
                    "SELECT kind, action, run_id, session_id, user_id, guest_session_id, value, comment, created_at "
                    "FROM feedback_events ORDER BY id"
                )
            ).all()
            interrupted = c.execute(text("SELECT interrupted FROM messages WHERE id = 'm1'")).scalar_one()
        assert [tuple(e)[:8] for e in events] == [
            ("survey", "snapshot", None, "s1", "u1", None, "yes", None),  # changed on the 3rd: first
            ("rating", "snapshot", "r1", "s1", None, "g1", "down", "wrong"),
        ]
        assert [str(e[8])[:10] for e in events] == ["2026-11-03", "2026-11-05"]
        assert not interrupted
        assert "interrupted_stage" in {col["name"] for col in inspect(engine).get_columns("pipeline_runs")}

        alembic_command.downgrade(_cfg(url), "0018_pipeline_run_stand_build")
        names = inspect(engine)
        assert "feedback_events" not in names.get_table_names()
        assert "interrupted" not in {col["name"] for col in names.get_columns("messages")}
        assert "interrupted_stage" not in {col["name"] for col in names.get_columns("pipeline_runs")}
        with engine.connect() as c:  # the data the downgrade did not own is still there
            assert c.execute(text("SELECT content FROM messages WHERE id = 'm1'")).scalar_one() == "A"
            assert c.execute(text("SELECT value FROM message_feedback")).scalar_one() == "down"

        alembic_command.upgrade(_cfg(url), "0019_keep_user_data")  # and up again
    finally:
        engine.dispose()
