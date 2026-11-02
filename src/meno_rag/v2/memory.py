"""Three-layer dialogue memory: recent turns, cached summary of older turns, lexical recall."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import structlog
from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.tokenization import tokenize_and_normalize_text

logger = structlog.get_logger(__name__)

CHARS_PER_TOKEN = 3
TRUNC_MARK = "[…]"
SUMMARY_TIMEOUT_S = 30
FALLBACK_USER_CHARS = 200
SUMMARY_CACHE_SIZE = 2048
MIN_STEM_LEN = 5
CHUNK_FACTOR = 4
_STOP_WORDS = (
    "говор напомн начал раньш ранее вчер спрашива сказа расскаж упомина"
).split()
_stemmer = SnowballStemmer("russian")
_STOP_STEMS = frozenset(_STOP_WORDS) | {_stemmer.stem(w) for w in _STOP_WORDS}

SUMMARY_SYSTEM_PROMPT = """Ты сжимаешь старую часть диалога студента с ассистентом НГУ в краткую фактическую сводку. Не отвечай на реплики, только пересказывай.

В сводке укажи:
- кто пользователь — только если он сам это сказал (например, «второй курс ФИТ»);
- что он спрашивал;
- какие факты и ссылки ассистент уже дал;
- что осталось нерешённым;
- фокус диалога: обсуждаемые факультет, программа, группа, человек.

Не выдумывай ничего, чего нет в диалоге. Если дана прежняя сводка, объедини её с новыми репликами в одну. Пиши по-русски, плотным текстом без вступлений, не длиннее {budget} токенов (примерно {chars} символов)."""


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


@dataclass
class DialogueMemory:
    recent: list[dict] = field(default_factory=list)
    older: list[dict] = field(default_factory=list)
    summary: str = ""
    recalled: list[dict] = field(default_factory=list)


def _exchanges(history: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for msg in history:
        if msg["role"] == "assistant" and out and len(out[-1]) == 1 and out[-1][0]["role"] == "user":
            out[-1].append(msg)
        else:
            out.append([msg])
    return out


def _truncate_msg(msg: dict, max_tokens: int) -> dict:
    limit = max_tokens * CHARS_PER_TOKEN
    text = msg["content"]
    if len(text) <= limit:
        return msg
    keep = max(limit - len(TRUNC_MARK) - 1, 0)
    return {**msg, "content": text[:keep].rstrip() + " " + TRUNC_MARK}


def split_history(history: list[dict], budget_tokens: int) -> tuple[list[dict], list[dict]]:
    """Return (recent, older): whole exchanges from the end that fit the budget; the newest always kept."""
    exchanges = _exchanges(history)
    if not exchanges:
        return [], []
    newest = [_truncate_msg(m, budget_tokens // 2) for m in exchanges[-1]]
    used = sum(estimate_tokens(m["content"]) for m in newest)
    kept = [newest]
    idx = len(exchanges) - 1
    while idx > 0:
        ex = exchanges[idx - 1]
        cost = sum(estimate_tokens(m["content"]) for m in ex)
        if used + cost > budget_tokens:
            break
        used += cost
        kept.append(ex)
        idx -= 1
    recent = [m for ex in reversed(kept) for m in ex]
    older = [m for ex in exchanges[:idx] for m in ex]
    return recent, older


class SummaryStore:
    """LRU cache of summaries keyed by the sha256 of the (role, content) pairs they cover."""

    def __init__(self, max_entries: int = SUMMARY_CACHE_SIZE) -> None:
        self._max = max_entries
        self._data: OrderedDict[str, tuple[int, str]] = OrderedDict()
        self.inflight: dict[str, asyncio.Future] = {}

    @staticmethod
    def key(older: list[dict]) -> str:
        raw = json.dumps([[m["role"], m["content"]] for m in older], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, older: list[dict]) -> str | None:
        k = self.key(older)
        if k not in self._data:
            return None
        self._data.move_to_end(k)
        return self._data[k][1]

    def find_prefix(self, older: list[dict]) -> tuple[int, str] | None:
        """Longest cached prefix of older (whole exchanges): (covered_messages_count, summary)."""
        for n in range(len(older), 0, -2):
            hit = self.get(older[:n])
            if hit is not None:
                return n, hit
        return None

    def put(self, older: list[dict], summary: str) -> None:
        k = self.key(older)
        self._data[k] = (len(older), summary)
        self._data.move_to_end(k)
        while len(self._data) > self._max:
            self._data.popitem(last=False)


def _fit_sentences(text: str, budget_tokens: int) -> str:
    text = text.strip()
    limit = budget_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [m.end() for m in re.finditer(r"[.!?…]+(?=\s|$)|\n", cut)]
    return cut[: ends[-1]].strip() if ends else cut.rstrip()


def _fallback_summary(older: list[dict], budget_tokens: int) -> str:
    lines = []
    for m in older:
        if m["role"] != "user":
            continue
        text = m["content"].strip()
        if len(text) > FALLBACK_USER_CHARS:
            text = text[:FALLBACK_USER_CHARS].rstrip() + "…"
        lines.append(text)
    limit = budget_tokens * CHARS_PER_TOKEN
    while len(lines) > 1 and len("\n".join(lines)) > limit:
        lines.pop(0)
    return "\n".join(lines)[:limit]


async def summarize(
    llm_router: Any, runtime: Any, older: list[dict], previous_summary: str, budget_tokens: int
) -> tuple[str, bool]:
    """Compress turns (plus an earlier summary) into plain text within the budget.

    Returns (text, ok); on failure text is a deterministic fallback and ok is False.
    """
    transcript = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}" for m in older
    )
    parts = []
    if previous_summary:
        parts.append(f"Прежняя сводка:\n{previous_summary}")
    parts.append(f"Реплики для сжатия:\n{transcript}")
    messages = [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM_PROMPT.format(budget=budget_tokens, chars=budget_tokens * CHARS_PER_TOKEN),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    try:
        text = await llm_router.chat_completion_text(
            runtime=runtime,
            messages=messages,
            stage="query_rewrite",
            temperature=0,
            max_tokens=budget_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=SUMMARY_TIMEOUT_S,
        )
        if not text or not text.strip():
            raise ValueError("empty summary")
        return _fit_sentences(text, budget_tokens), True
    except Exception as exc:
        logger.warning("v2_summary_failed", error=repr(exc))
        return _fallback_summary(older, budget_tokens), False


def _stems(text: str) -> set[str]:
    return {
        t
        for t in tokenize_and_normalize_text(text, _stemmer).split()
        if len(t) >= MIN_STEM_LEN and t not in _STOP_STEMS
    }


def recall_old_turns(older: list[dict], query: str, k: int = 2) -> list[dict]:
    """Verbatim old exchanges sharing a content stem with the query (chronological)."""
    q = _stems(query)
    exchanges = _exchanges(older)
    if not q or not exchanges:
        return []
    ex_stems = [_stems(" ".join(m["content"] for m in ex)) for ex in exchanges]
    n = len(exchanges)
    idf = {t: math.log((n + 1) / (1 + sum(t in s for s in ex_stems))) + 1 for t in q}
    scored = []
    for i, st in enumerate(ex_stems):
        common = q & st
        if common:
            scored.append((len(common), sum(idf[t] for t in common), i))
    top = sorted(scored, reverse=True)[:k]
    return [m for _, _, i in sorted(top, key=lambda s: s[2]) for m in exchanges[i]]


def _chunks(delta: list[dict], budget_tokens: int) -> list[list[dict]]:
    """Split into groups of whole exchanges of at most ~CHUNK_FACTOR*budget tokens."""
    cap = CHUNK_FACTOR * budget_tokens
    out: list[list[dict]] = []
    cur: list[dict] = []
    used = 0
    for ex in _exchanges(delta):
        ex = [_truncate_msg(m, cap // 2) for m in ex]
        cost = sum(estimate_tokens(m["content"]) for m in ex)
        if cur and used + cost > cap:
            out.append(cur)
            cur, used = [], 0
        cur += ex
        used += cost
    if cur:
        out.append(cur)
    return out


async def _compute_summary(
    older: list[dict], llm_router: Any, runtime: Any, store: SummaryStore, budget_tokens: int
) -> str:
    covered, summary = store.find_prefix(older) or (0, "")
    for chunk in _chunks(older[covered:], budget_tokens):
        covered += len(chunk)
        summary, ok = await summarize(llm_router, runtime, chunk, summary, budget_tokens)
        if not ok:
            # lossy fallback is never cached; keep every old user question
            return _fallback_summary(older, budget_tokens)
        store.put(older[:covered], summary)
    return summary


async def _summary_for(
    older: list[dict], llm_router: Any, runtime: Any, store: SummaryStore, budget_tokens: int
) -> str:
    cached = store.get(older)
    if cached is not None:
        return cached
    key = store.key(older)
    task = store.inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(_compute_summary(older, llm_router, runtime, store, budget_tokens))
        store.inflight[key] = task
        task.add_done_callback(lambda _t: store.inflight.pop(key, None))
    return await asyncio.shield(task)


async def build_memory(
    history: list[dict], query: str, llm_router: Any, runtime: Any, store: SummaryStore, settings: Any
) -> DialogueMemory:
    recent, older = split_history(history, settings.v2_history_budget_tokens)
    if not older:
        return DialogueMemory(recent=recent)
    summary = await _summary_for(older, llm_router, runtime, store, settings.v2_summary_budget_tokens)
    return DialogueMemory(recent=recent, older=older, summary=summary, recalled=recall_old_turns(older, query))


async def precompute_next(
    history_with_answer: list[dict], llm_router: Any, runtime: Any, store: SummaryStore, settings: Any
) -> None:
    """Background task: warm the summary the next turn will need. Never raises."""
    try:
        _, older = split_history(history_with_answer, settings.v2_history_budget_tokens)
        if older:
            await _summary_for(older, llm_router, runtime, store, settings.v2_summary_budget_tokens)
    except Exception as exc:
        logger.warning("v2_precompute_failed", error=repr(exc))
