"""Evidence judge: one short guided-JSON LLM call that decides which retrieved documents answer the question,
whether they are enough, whether searching again can help and how the answer should be given."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlsplit

from meno_rag.v2.analysis import _dedup
from meno_rag.v2.memory import CHARS_PER_TOKEN
from meno_rag.v2.prompting import _doc_type_label, _format_date
from meno_rag.v2.search import Candidate
from meno_rag.v2.snippets import snippet
from meno_rag.v2.sources import clean_title

logger = logging.getLogger(__name__)

VERDICTS = ("sufficient", "partial", "insufficient")
ANSWER_MODES = ("documents", "partial", "clarify", "general_knowledge", "no_data")
MAX_INDICES = 16
MAX_QUERIES = 2
MISSING_WORDS = 20
FALLBACK_DOCS = 6
JUDGE_MAX_TOKENS = 220
PROMPT_TOKENS = 1800  # target size of the whole judge prompt (len/3 estimate)
MAX_EXCERPT = 300
MIN_EXCERPT = 100
TITLE_CHARS = 70
MAX_REPAIR_TRIES = 200

_INDEX_LIST = {"type": "array", "items": {"type": "integer"}, "maxItems": MAX_INDICES}
JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "relevant": _INDEX_LIST,
        "stale": _INDEX_LIST,
        "other_entity": _INDEX_LIST,
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "answer_mode": {"type": "string", "enum": list(ANSWER_MODES)},
        "missing": {"type": "string"},
        "search_more": {"type": "boolean"},
        "queries": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_QUERIES},
        "clarify_question": {"type": "string"},
    },
    "required": ["relevant", "stale", "other_entity", "verdict", "answer_mode", "missing", "search_more",
                 "queries", "clarify_question"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Ты отбираешь документы сайтов НГУ для ответа на вопрос. Сам не отвечай. Сегодня {today}. Верни JSON:
relevant — номера документов, которые помогают ответить и говорят о том же человеке, подразделении, программе или событии, что имеет в виду пользователь.
other_entity — номера документов об однофамильце, другом факультете, программе или уровне обучения.
stale — номера из relevant, которым противоречит более свежий документ, если вопрос о текущем положении.
verdict: sufficient — сведений хватает; partial — на часть; insufficient — нет.
answer_mode: documents; partial — ответить тем, что есть, и сказать, чего нет; clarify — вопрос понятен по-разному, а документы о разных вариантах; general_knowledge — только вопрос об общих понятиях или навыках, не о фактах про НГУ, людей, подразделения, события, правила; no_data — подходящих документов нет.
missing — чего не хватает, до 20 слов.
search_more — true, только если недостающее вероятно есть на сайтах НГУ в других словах. Отзывов, мнений и оценок людей там обычно нет; если документы явно этого не касаются — partial и false.
queries — до 2 запросов, только при search_more; имена и названия из вопроса сохраняй дословно.
clarify_question — вопрос пользователю для clarify, иначе ""."""


@dataclass
class Judgement:
    relevant: list[int]  # 0-based indices into the judged candidates, about the entity the user means
    verdict: str
    answer_mode: str = "documents"
    stale: set[int] = field(default_factory=set)  # subset of relevant
    other_entity: set[int] = field(default_factory=set)
    missing: str = ""
    search_more: bool = False
    queries: list[str] = field(default_factory=list)
    clarify_question: str = ""
    fallback: bool = False


# --- prompt -------------------------------------------------------------------------------


def _host(url: str | None) -> str:
    host = urlsplit((url or "").strip()).netloc.lower().removeprefix("www.")
    return host or "без адреса"


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: max(limit - 1, 1)]
    return (cut[: cut.rindex(" ")] if " " in cut else cut).rstrip(" ,.:;—-") + "…"


def _header(n: int, cand: Candidate) -> str:
    c = cand.chunk
    title = _cut(clean_title(c.title) or "без названия", TITLE_CHARS)
    return f"[{n}] {title} · {_doc_type_label(c.doc_type)} · {_format_date(c.date)} · {_host(c.url)}"


def _excerpt(cand: Candidate, limit: int, terms: dict[str, float] | None = None) -> str:
    """The window of the chunk that best matches the question terms; the head when nothing matches."""
    if terms:
        focused = snippet(cand.chunk.text or "", terms, limit)
        if focused:
            return focused
    text = " ".join((cand.chunk.index_text or cand.chunk.text or "").split())
    for head in (" ".join((cand.chunk.title or "").split()), clean_title(cand.chunk.title)):
        if head and text.startswith(head):
            text = text[len(head):].lstrip(" .:—-|")
            break
    return _cut(text, limit)


def build_judge_messages(
    *, question: str, standalone: str, needs: list[str], dialogue_hint: str, candidates: list[Candidate],
    today: date, tried: Sequence[str] = (), terms: dict[str, float] | None = None,
) -> list[dict]:
    """Excerpts share what is left of PROMPT_TOKENS after the fixed parts (MIN_EXCERPT..MAX_EXCERPT chars);
    with terms (snippets.query_terms of the question and needs) each excerpt is the best-matching window."""
    anchor = standalone.strip() or question.strip()
    parts = [f"Вопрос: {anchor}"]
    if question.strip() and question.strip() != anchor:
        parts.append(f"Реплика: {question.strip()}")
    if dialogue_hint.strip():
        parts.append(dialogue_hint.strip())
    items = [q for q in needs if q.strip() and q.strip() != anchor]
    if items:
        parts.append("Нужно узнать: " + "; ".join(f"{n}) {q}" for n, q in enumerate(items, 1)))
    if tried:
        parts.append("Уже искали: " + "; ".join(tried))
    system = SYSTEM_PROMPT.format(today=f"{today:%d.%m.%Y}")
    headers = [_header(n, c) for n, c in enumerate(candidates, 1)]
    fixed = len(system) + sum(len(p) + 1 for p in parts) + sum(len(h) + 2 for h in headers) + 16
    per = (PROMPT_TOKENS * CHARS_PER_TOKEN - fixed) // max(len(candidates), 1)
    per = max(MIN_EXCERPT, min(MAX_EXCERPT, per))
    docs = "\n".join(f"{h}\n{_excerpt(c, per, terms)}" for h, c in zip(headers, candidates, strict=True))
    parts.append("Документы:\n" + (docs or "(ничего не найдено)"))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


# --- parsing ------------------------------------------------------------------------------


def _close_prefix(text: str) -> dict | None:
    """The longest prefix of a (possibly truncated) JSON object that parses once its brackets are closed."""
    stack: list[str] = []
    cuts: list[tuple[int, str]] = []  # (end of prefix, closing brackets at that point)
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                cuts.append((i + 1, "".join(reversed(stack))))
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
            if not stack:
                break  # the first complete object ends here
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))
        elif ch.isdigit() or ch in "el":  # the end of a number, true, false or null
            cuts.append((i + 1, "".join(reversed(stack))))
    for end, closers in reversed(cuts[-MAX_REPAIR_TRIES:]):
        try:
            obj = json.loads(text[:end] + closers)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_object(raw: str) -> dict | None:
    """The first complete JSON object in raw (fences and chatter around it ignored), else the longest
    prefix of a truncated one that can be closed, else None."""
    start = raw.find("{")
    if start < 0:
        return None
    text = raw[start:]
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    return _close_prefix(text)


def _indices(value, n: int) -> list[int]:
    """1-based numbers from the prompt -> unique 0-based indices; unknown ones are ignored."""
    out: list[int] = []
    for v in value if isinstance(value, list) else []:
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= n and v - 1 not in out:
            out.append(v - 1)
    return out


def fallback_judgement(candidates: list[Candidate], threshold: float, limit: int = FALLBACK_DOCS) -> Judgement:
    """Only when the judge gave no usable verdict at all: the best cross-encoder candidates >= threshold."""
    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i].score, candidates[i].pos))
    picked = [i for i in order if candidates[i].score >= threshold][:limit]
    return Judgement(picked, "partial", answer_mode="documents" if picked else "no_data", fallback=True)


def parse_judgement(raw: str, candidates: list[Candidate], threshold: float) -> Judgement:
    """Never raises: an unusable reply gives the fallback judgement."""
    data = extract_object(raw or "")
    verdict = data.get("verdict") if data else None
    mode = data.get("answer_mode") if data else None
    if verdict not in VERDICTS or mode not in ANSWER_MODES:
        logger.warning("evidence judge reply unusable, falling back to the top reranker documents: %r",
                       (raw or "")[:200])
        return fallback_judgement(candidates, threshold)
    n = len(candidates)
    other = set(_indices(data.get("other_entity"), n))
    relevant = [i for i in _indices(data.get("relevant"), n) if i not in other]
    queries = data.get("queries")
    missing, clarify = data.get("missing"), data.get("clarify_question")
    j = Judgement(
        relevant=relevant,
        verdict=verdict,
        answer_mode=mode,
        stale=set(_indices(data.get("stale"), n)) & set(relevant),
        other_entity=other,
        missing=" ".join(missing.split()[:MISSING_WORDS]) if isinstance(missing, str) else "",
        search_more=data.get("search_more") is True,
        queries=_dedup(queries if isinstance(queries, list) else [])[:MAX_QUERIES],
        clarify_question=clarify.strip() if isinstance(clarify, str) else "",
    )
    return _reconcile(j)


def _reconcile(j: Judgement) -> Judgement:
    """Make the answer mode and the search decision agree with what the judge approved.

    Only approved documents are evidence: with none, the answer is general knowledge when the judge chose
    that, otherwise no data (never "the best documents it did not reject")."""
    if not j.relevant and j.answer_mode != "general_knowledge":
        j.answer_mode = "no_data"
    elif j.answer_mode == "no_data" and j.relevant:
        j.answer_mode = "partial"  # some approved evidence: answer what it covers, say what is missing
    if j.verdict == "sufficient" and j.answer_mode in ("no_data", "partial"):
        j.verdict = "partial" if j.relevant else "insufficient"
    if j.verdict == "sufficient" or not j.queries:
        j.search_more = False
    if not j.search_more:
        j.queries = []
    return j


async def judge(
    llm_router,
    runtime,
    *,
    question: str,
    standalone: str,
    needs: list[str],
    dialogue_hint: str,
    candidates: list[Candidate],
    today: date,
    timeout_s: float,
    fallback_threshold: float = 0.5,
    tried: Sequence[str] = (),
    terms: dict[str, float] | None = None,
) -> Judgement:
    """Never raises: timeouts, transport errors and broken JSON give the fallback judgement."""
    messages = build_judge_messages(
        question=question, standalone=standalone, needs=needs, dialogue_hint=dialogue_hint,
        candidates=candidates, today=today, tried=list(tried), terms=terms,
    )
    try:
        raw = await asyncio.wait_for(
            llm_router.chat_completion_text(
                runtime=runtime,
                messages=messages,
                stage="rerank",
                temperature=0,
                max_tokens=JUDGE_MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "evidence_judgement", "schema": JUDGE_SCHEMA, "strict": True},
                },
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=timeout_s,
            ),
            timeout_s,
        )
    except Exception as exc:  # any failure degrades to the cross-encoder ranking
        logger.warning("evidence judge failed, falling back to the top reranker documents: %r", exc)
        return fallback_judgement(candidates, fallback_threshold)
    return parse_judgement(str(raw or ""), candidates, fallback_threshold)
