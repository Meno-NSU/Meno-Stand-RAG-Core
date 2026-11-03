"""Structural lookup of group / teacher timetables (table.nsu.ru export)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer

log = logging.getLogger(__name__)

_STEM = SnowballStemmer("russian")
GROUP_RE = re.compile(r"\b\d{5}(?:\.\d)?\b")
FACULTIES = {
    "ef": "ЭФ", "fen": "ФЕН", "ff": "ФФ", "fit": "ФИТ", "ggf": "ГГФ", "gi": "ГИ",
    "iir": "ИИР", "immt": "ИМиМТ", "iphp": "ИФП", "mmf": "ММФ", "pish": "ПИШ",
}
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
_WEEKDAY_RES = [
    r"понедельник\w*", r"вторник\w*", r"сред(?:а|ы|е|у|ой)", r"четверг\w*",
    r"пятниц\w*", r"суббот\w*", r"воскресен\w*",
]


@dataclass
class ScheduleBlock:
    title: str
    url: str
    date: str
    text: str


def _stem(word: str) -> str:
    return _STEM.stem(word.lower().replace("ё", "е"))


def _same_stem(a: str, b: str) -> bool:
    # Snowball stems inflections of one surname inconsistently (терсен / терсенов).
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and long_.startswith(short) and len(long_) - len(short) <= 2


class ScheduleIndex:
    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.groups: dict[str, list[dict]] = {}
        self.teachers: dict[str, list[dict]] = {}  # full name -> records
        self.surname_stems: dict[str, set[str]] = {}
        for r in records:
            self.groups.setdefault(str(r["group"]), []).append(r)
            for name in re.split(r"[;,]", r.get("teacher") or ""):
                name = name.strip()
                if not name:
                    continue
                self.teachers.setdefault(name, []).append(r)
                self.surname_stems.setdefault(_stem(name.split()[0]), set()).add(name)

    def lookup(self, text: str, today: date, max_blocks: int = 2) -> list[ScheduleBlock]:
        """Group numbers always match; teacher surnames only when the question is about a timetable."""
        target, day = _target(text, today)
        blocks: list[ScheduleBlock] = []
        seen: set[str] = set()
        for g in GROUP_RE.findall(text):
            if g in seen:
                continue
            seen.add(g)
            recs = [r for k, rs in self.groups.items() if k == g or ("." not in g and k.split(".")[0] == g) for r in rs]
            if recs:
                blocks.append(_group_block(g, recs, today, target, day))
        names: list[str] = []
        for w in re.findall(r"[А-Яа-яЁё-]{5,}", text) if has_schedule_intent(text) else []:
            q = _stem(w)
            for st, ns in self.surname_stems.items():
                if _same_stem(q, st):
                    names.extend(n for n in sorted(ns) if n not in names)
        for name in names:
            blocks.append(_teacher_block(name, self.teachers[name], today, target, day))
        return blocks[:max_blocks]


INTENT_PREFIXES = ("расписан", "пар", "занят", "лекци", "семинар", "практик", "ауд")
INTENT_PHRASES = ("когда у", "во сколько", "где сейчас")


def has_schedule_intent(text: str) -> bool:
    low = text.lower().replace("ё", "е")
    if any(p in low for p in INTENT_PHRASES):
        return True
    return any(w.startswith(INTENT_PREFIXES) for w in re.findall(r"[а-я]+", low))


def load_schedule(path: str | Path | None) -> ScheduleIndex | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        log.warning("schedule file not found: %s", p)
        return None
    records = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return ScheduleIndex(records) if records else None


def _target(text: str, today: date) -> tuple[date, int | None]:
    """Returns (a date inside the wanted week, weekday 0-6 to show or None)."""
    t = text.lower().replace("ё", "е")
    if "послезавтра" in t:
        d = today + timedelta(days=2)
        return d, d.weekday()
    if "завтра" in t:
        d = today + timedelta(days=1)
        return d, d.weekday()
    if "сегодня" in t:
        return today, today.weekday()
    for i, pat in enumerate(_WEEKDAY_RES):
        if re.search(rf"\b{pat}\b", t):
            return today, i
    return today, None


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _week_lessons(recs: list[dict], today: date, target: date, day: int | None) -> tuple[date, dict[int, list[dict]]]:
    monday = _monday(target)
    by_day: dict[int, list[dict]] = {}
    for r in recs:
        wd = int(r["weekday"]) - 1
        if day is not None and wd != day:
            continue
        lesson = monday + timedelta(days=wd)
        if date.fromisoformat(r["until"]) < today or lesson > date.fromisoformat(r["until"]):
            continue
        first = date.fromisoformat(r["first_date"])
        if lesson < first:
            continue
        if int(r.get("interval_weeks", 1)) == 2 and (monday - _monday(first)).days % 14 != 0:
            continue
        by_day.setdefault(wd, []).append(r)
    for rs in by_day.values():
        rs.sort(key=lambda r: r["start"])
    return monday, by_day


def _render(header: str, monday: date, by_day: dict[int, list[dict]], line) -> str:
    out = [header]
    if not by_day:
        out.append("Занятий на этот период в расписании нет.")
    for wd in sorted(by_day):
        d = monday + timedelta(days=wd)
        out.append(f"{WEEKDAYS[wd].capitalize()}, {d:%d.%m}:")
        out.extend(line(r) for r in by_day[wd])
    return "\n".join(out)


def _period(monday: date) -> str:
    return f"с {monday:%d.%m} по {monday + timedelta(days=6):%d.%m}"


def _lesson(r: dict, tail: str) -> str:
    kind = f" ({r['kind']})" if r.get("kind") else ""
    room = f" · ауд. {r['room']}" if r.get("room") else ""
    return f"{r['start']}–{r['end']} · {r['subject']}{kind}{room} · {tail}"


def _group_block(g: str, recs: list[dict], today: date, target: date, day: int | None) -> ScheduleBlock:
    monday, by_day = _week_lessons(recs, today, target, day)
    fac = FACULTIES.get(recs[0].get("faculty", ""), recs[0].get("faculty", "").upper())
    title = f"Расписание группы {g}" + (f" ({fac})" if fac else "")
    header = f"{title} на неделю {_period(monday)}, по данным table.nsu.ru на {today:%d.%m.%Y}"
    text = _render(header, monday, by_day, lambda r: _lesson(r, r.get("teacher", "")))
    return ScheduleBlock(title, recs[0].get("url", ""), today.isoformat(), text)


def _teacher_block(name: str, recs: list[dict], today: date, target: date, day: int | None) -> ScheduleBlock:
    monday, by_day = _week_lessons(recs, today, target, day)
    title = f"Расписание преподавателя {name}"
    header = f"{title} на неделю {_period(monday)}, по данным table.nsu.ru на {today:%d.%m.%Y}"
    text = _render(header, monday, by_day, lambda r: _lesson(r, f"группа {r['group']}"))
    return ScheduleBlock(title, recs[0].get("url", ""), today.isoformat(), text)
