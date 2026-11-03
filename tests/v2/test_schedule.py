import json
from datetime import date

from meno_rag.v2.schedule import ScheduleIndex, load_schedule


def rec(**kw):
    base = {"group": "26201", "faculty": "fit", "weekday": 1, "start": "09:00", "end": "10:35",
            "subject": "Математический анализ", "kind": "лекция", "room": "209 КПА",
            "teacher": "Терсенов Арис Саввич", "interval_weeks": 1, "first_date": "2026-09-07",
            "until": "2026-12-31", "url": "https://table.nsu.ru/group/26201", "collected_at": "2026-11-02"}
    base.update(kw)
    return base


TODAY = date(2026, 11, 2)  # Monday
IDX = ScheduleIndex([
    rec(),
    rec(weekday=3, start="11:10", end="12:45", subject="Физика", teacher="Иванов Иван Иванович", kind="семинар"),
    rec(weekday=2, subject="Биweekly", interval_weeks=2, first_date="2026-09-08"),   # parity: 8 weeks -> shown
    rec(weekday=4, subject="Нечётная", interval_weeks=2, first_date="2026-09-14"),   # 7 weeks -> hidden
    rec(weekday=5, subject="Старое", until="2026-10-01"),
    rec(group="26202", faculty="mmf", subject="Другая группа", url="https://table.nsu.ru/group/26202"),
])


def test_group_lookup():
    [b] = IDX.lookup("расписание группы 26201", TODAY)
    assert b.text.startswith("Расписание группы 26201 (ФИТ) на неделю с 02.11 по 08.11, по данным table.nsu.ru на 02.11.2026")
    assert "09:00–10:35 · Математический анализ (лекция) · ауд. 209 КПА · Терсенов Арис Саввич" in b.text
    assert "Биweekly" in b.text and "Нечётная" not in b.text and "Старое" not in b.text
    assert b.url.endswith("26201") and b.date == "2026-11-02"


def test_teacher_lookup_inflected():
    [b] = IDX.lookup("что за пара у Терсенова?", TODAY)
    assert "группа 26201" in b.text and "Физика" not in b.text


def test_weekday_filter():
    [b] = IDX.lookup("26201 в среду", TODAY)
    assert "Физика" in b.text and "Математический" not in b.text
    [b] = IDX.lookup("26201 завтра", TODAY)  # Tuesday
    assert "Биweekly" in b.text and "Физика" not in b.text


def test_nothing_found():
    assert IDX.lookup("когда сессия?", TODAY) == []
    assert IDX.lookup("группа 99999", TODAY) == []


def test_max_blocks_and_load(tmp_path):
    assert len(IDX.lookup("26201 26202 Иванов", TODAY, max_blocks=2)) == 2
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(rec(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_schedule(str(p)) is not None
    assert load_schedule("") is None and load_schedule(str(tmp_path / "no")) is None


def test_teacher_requires_schedule_intent():
    assert IDX.lookup("где работает Иванов", TODAY) == []
    [b] = IDX.lookup("когда пара у Иванова", TODAY)
    assert "Физика" in b.text
    assert len(IDX.lookup("когда у Иванова?", TODAY)) == 1
    assert len(IDX.lookup("группа 26201", TODAY)) == 1  # group numbers stay ungated
