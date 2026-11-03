"""Evidence judge: one guided-JSON LLM call that decides which retrieved documents answer the question,
whether they are enough, what to search next and how the answer should be given."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlsplit

from meno_rag.v2.analysis import _dedup, _extract_json
from meno_rag.v2.prompting import _doc_type_label, _format_date
from meno_rag.v2.search import Candidate
from meno_rag.v2.sources import clean_title

logger = logging.getLogger(__name__)

VERDICTS = ("sufficient", "partial", "insufficient")
ANSWER_MODES = ("documents", "partial", "clarify", "general_knowledge", "no_data")
DOC_MODES = ("documents", "partial")
EXCERPT_CHARS = 400
MAX_QUERIES = 3
MAX_DOC_ITEMS = 24
FALLBACK_DOCS = 6
JUDGE_MAX_TOKENS = 700

JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "docs": {
            "type": "array",
            "maxItems": MAX_DOC_ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "same_entity": {"type": "boolean"},
                    "relevant": {"type": "boolean"},
                    "stale": {"type": "boolean"},
                    "need": {"type": ["integer", "null"]},
                },
                "required": ["i", "same_entity", "relevant", "stale", "need"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "missing": {"type": "string"},
        "queries": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_QUERIES},
        "answer_mode": {"type": "string", "enum": list(ANSWER_MODES)},
        "clarify_question": {"type": "string"},
    },
    "required": ["docs", "verdict", "missing", "queries", "answer_mode", "clarify_question"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Ты проверяешь, хватает ли найденных документов с сайтов НГУ (Новосибирский государственный университет), чтобы ответить на вопрос пользователя. Сам на вопрос не отвечай. Сегодня {today}.

Тебе даны вопрос, пронумерованный список того, что нужно узнать, и документы, найденные поиском по вопросу, этим пунктам и прежним запросам: номер, название, тип, дата, сайт и начало текста.

docs — записи только о документах, которые относятся к вопросу, и о похожих, но неподходящих; остальные пропусти.
- i — номер документа.
- same_entity — true, только если документ о том же человеке, подразделении, программе или событии, что имеет в виду пользователь. Однофамилец, другой факультет, другая программа или уровень обучения — false; такой документ не relevant.
- relevant — true, если документ помогает ответить.
- stale — true, если вопрос о текущем положении дел, а более свежий документ ему противоречит (сменились руководитель, сроки, правила). Для вопросов о текущем положении предпочитай самые свежие документы с датой.
- need — номер пункта из «Нужно узнать», который закрывает документ, или null.

verdict — sufficient: документы закрывают все пункты; partial: часть; insufficient: ничего.
missing — коротко, чего не хватает; "" если всего хватает.
queries — до трёх новых поисковых запросов, только если verdict не sufficient. Сохраняй в них дословно имена, фамилии, названия и сокращения из вопроса; не обобщай вопрос и не меняй его тему. Не повторяй вопрос, пункты и прежние запросы.
answer_mode:
- documents — ответить по документам;
- partial — ответить по документам на то, что есть, и сказать, чего нет;
- clarify — вопрос можно понять по-разному (уровень обучения, факультет, год и т. п.), а документы относятся к разным вариантам;
- general_knowledge — только если вопрос об общих понятиях или навыках (объяснить тему, помочь с текстом или задачей) и не касается конкретных фактов об НГУ, его людях, подразделениях, событиях или правилах;
- no_data — вопрос о конкретных людях, подразделениях, событиях или правилах, а подходящих документов нет.
clarify_question — для clarify короткий уточняющий вопрос пользователю, иначе "".
Ответ — только JSON."""


@dataclass
class DocJudgement:
    i: int  # 0-based index into the judged candidates
    relevant: bool
    need: int | None  # 0-based index into needs
    stale: bool
    same_entity: bool

    @property
    def counts(self) -> bool:
        """Evidence only when relevant AND about the entity the user means."""
        return self.relevant and self.same_entity


@dataclass
class Judgement:
    docs: list[DocJudgement]
    verdict: str
    missing: str = ""
    queries: list[str] = field(default_factory=list)
    answer_mode: str = "documents"
    clarify_question: str = ""
    fallback: bool = False

    @property
    def relevant(self) -> list[int]:
        return [d.i for d in self.docs if d.counts]

    @property
    def stale(self) -> set[int]:
        return {d.i for d in self.docs if d.counts and d.stale}

    @property
    def rejected(self) -> set[int]:
        return {d.i for d in self.docs if not d.counts}


def _host(url: str | None) -> str:
    host = urlsplit((url or "").strip()).netloc.lower().removeprefix("www.")
    return host or "без адреса"


def _excerpt(cand: Candidate, title: str) -> str:
    text = " ".join((cand.chunk.index_text or cand.chunk.text or "").split())
    raw_title = " ".join((cand.chunk.title or "").split())
    for head in (raw_title, title):
        if head and text.startswith(head):
            text = text[len(head):].lstrip(" .:—-|")
            break
    if len(text) > EXCERPT_CHARS:
        cut = text[:EXCERPT_CHARS]
        text = (cut[: cut.rindex(" ")] if " " in cut else cut) + "…"
    return text


def render_candidates(candidates: list[Candidate]) -> str:
    blocks = []
    for n, cand in enumerate(candidates, 1):
        c = cand.chunk
        title = clean_title(c.title) or "без названия"
        header = f"[{n}] {title} · {_doc_type_label(c.doc_type)} · {_format_date(c.date)} · {_host(c.url)}"
        blocks.append(f"{header}\n{_excerpt(cand, title)}")
    return "\n\n".join(blocks)


def build_judge_messages(
    *, question: str, standalone: str, needs: list[str], dialogue_hint: str, candidates: list[Candidate],
    today: date, tried: Sequence[str] = (),
) -> list[dict]:
    parts = [f"Вопрос: {standalone.strip() or question.strip()}"]
    if question.strip() and question.strip() != standalone.strip():
        parts.append(f"Реплика пользователя дословно: {question.strip()}")
    if dialogue_hint.strip():
        parts.append(dialogue_hint.strip())
    items = needs or [standalone]
    parts.append("Нужно узнать:\n" + "\n".join(f"{n}. {q}" for n, q in enumerate(items, 1)))
    if tried:
        parts.append("Прежние запросы:\n" + "\n".join(f"- {q}" for q in tried))
    parts.append("Документы:\n" + (render_candidates(candidates) if candidates else "(ничего не найдено)"))
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=f"{today:%d.%m.%Y}")},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def fallback_judgement(
    candidates: list[Candidate], threshold: float, *, exclude: set[int] = frozenset(), limit: int = FALLBACK_DOCS
) -> Judgement:
    """Without a usable judgement: the best cross-encoder candidates above the threshold."""
    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i].score, candidates[i].pos))
    picked = [i for i in order if i not in exclude and candidates[i].score >= threshold][:limit]
    docs = [DocJudgement(i, True, None, False, True) for i in picked]
    return Judgement(docs, "partial", answer_mode="documents" if docs else "no_data", fallback=True)


def _as_bool(value, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def parse_judgement(raw: str, candidates: list[Candidate], needs: list[str], threshold: float) -> Judgement:
    data = _extract_json(raw)  # also skips a ```json fence
    verdict, mode = data.get("verdict"), data.get("answer_mode")
    if verdict not in VERDICTS or mode not in ANSWER_MODES:
        raise ValueError(f"bad verdict/answer_mode {verdict!r}/{mode!r}")
    docs: list[DocJudgement] = []
    seen: set[int] = set()
    items = data.get("docs")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        n = _as_int(item.get("i"))
        if n is None or not 1 <= n <= len(candidates) or n - 1 in seen:
            continue  # numbers in the prompt start at 1; unknown ones are ignored
        need = _as_int(item.get("need"))
        need = need - 1 if need is not None and 1 <= need <= max(len(needs), 1) else None
        seen.add(n - 1)
        docs.append(DocJudgement(
            n - 1,
            _as_bool(item.get("relevant"), False),
            need,
            _as_bool(item.get("stale"), False),
            _as_bool(item.get("same_entity"), True),
        ))
    queries = data.get("queries")
    missing, clarify = data.get("missing"), data.get("clarify_question")
    j = Judgement(
        docs=docs,
        verdict=verdict,
        missing=missing.strip() if isinstance(missing, str) else "",
        queries=_dedup(queries if isinstance(queries, list) else [])[:MAX_QUERIES],
        answer_mode=mode,
        clarify_question=clarify.strip() if isinstance(clarify, str) else "",
    )
    return _reconcile(j, candidates, threshold)


def _reconcile(j: Judgement, candidates: list[Candidate], threshold: float) -> Judgement:
    """Make the answer mode agree with the documents the judge accepted."""
    if j.answer_mode in DOC_MODES and not j.relevant:
        # "answer from documents" without naming any: take the best ones it did not reject
        fb = fallback_judgement(candidates, threshold, exclude=j.rejected)
        if fb.docs:
            j.docs = [*j.docs, *fb.docs]
            j.answer_mode = "partial"
        else:
            j.answer_mode = "no_data"
    elif j.answer_mode == "no_data" and j.relevant:
        j.answer_mode = "partial"  # some accepted evidence: answer what it covers, say what is missing
    if j.verdict == "sufficient" and j.answer_mode in ("no_data", "partial"):
        j.verdict = "partial" if j.relevant else "insufficient"
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
) -> Judgement:
    messages = build_judge_messages(
        question=question, standalone=standalone, needs=needs, dialogue_hint=dialogue_hint,
        candidates=candidates, today=today, tried=list(tried),
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
        return parse_judgement(raw, candidates, needs, fallback_threshold)
    except Exception as exc:  # any failure degrades to the cross-encoder ranking
        logger.warning("evidence judge failed, falling back to reranker scores: %r", exc)
        return fallback_judgement(candidates, fallback_threshold)
