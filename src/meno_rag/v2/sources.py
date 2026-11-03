"""Post-hoc source attribution: which retrieved chunks actually back the answer."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from razdel import sentenize

from meno_rag.v2.search import Candidate

log = logging.getLogger(__name__)

MIN_WORDS = 4
SOCIAL_HOSTS = ("vk.com", "t.me", "telegram.me")
# one call scores every (query, texts) group and returns one score list per group
RerankerCall = Callable[[list[tuple[str, list[str]]]], Awaitable[list[list[float]]]]

KEEP_SCHEMA = {
    "type": "object",
    "properties": {"keep": {"type": "array", "items": {"type": "integer"}}},
    "required": ["keep"],
    "additionalProperties": False,
}


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        for s in sentenize(line):
            t = s.text.strip()
            if t.startswith("_") or t.lower().startswith("источник"):
                continue  # markers appended by code, not claims
            if len(re.findall(r"\w+", t)) >= MIN_WORDS:
                out.append(t)
    return out


async def attribute(
    answer: str,
    context_chunks: list[Candidate],
    reranker_call: RerankerCall,
    threshold: float,
) -> list[tuple[Candidate, float]]:
    sentences = split_sentences(answer)
    if not sentences or not context_chunks:
        return []
    texts = [c.chunk.text for c in context_chunks]
    best = [0.0] * len(context_chunks)
    all_scores = await reranker_call([(sentence, texts) for sentence in sentences])
    for scores in all_scores:
        for i, sc in enumerate(scores):
            best[i] = max(best[i], float(sc))
    used = [(c, b) for c, b in zip(context_chunks, best) if b >= threshold]
    used.sort(key=lambda p: -p[1])
    return used


def _norm_url(url: str) -> str:
    p = urlsplit((url or "").strip().lower())
    host = p.netloc.removeprefix("www.")
    return f"{host}{p.path.rstrip('/')}" + (f"?{p.query}" if p.query else "")


def _is_social(url: str) -> bool:
    host = urlsplit((url or "").lower()).netloc.removeprefix("www.")
    return any(host == h or host.endswith("." + h) for h in SOCIAL_HOSTS)


def _iso_date(raw: str | None) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(raw or "").strip())
    return m[1] if m else ""


def to_sources(used: list[tuple[Candidate, float]], docs: dict, max_sources: int) -> list[dict]:
    has_site = {c.chunk.doc_id for c, _ in used if not _is_social(c.chunk.url)}
    out: list[dict] = []
    seen_urls: set[str] = set()
    for cand, _ in sorted(used, key=lambda p: -p[1]):
        ch = cand.chunk
        key = _norm_url(ch.url)
        if key in seen_urls:
            continue
        if ch.doc_id and ch.doc_id in has_site and _is_social(ch.url):
            continue  # prefer the nsu.ru page over its repost
        seen_urls.add(key)
        title = ch.title or (docs.get(ch.doc_id) or {}).get("title") or ""
        out.append({"document_title": title, "source_url": ch.url, "date": _iso_date(ch.date)})
        if len(out) >= max_sources:
            break
    return out


async def llm_check(llm_router, runtime, answer: str, sources: list[dict]) -> list[dict]:
    if not sources:
        return sources
    listing = "\n".join(
        f"{i}. {s.get('document_title', '')} ({s.get('date') or 'без даты'})" for i, s in enumerate(sources)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Тебе дан ответ ассистента и список источников. Верни номера источников, "
                'которые подтверждают сказанное в ответе. Ответ — JSON {"keep": [номера]}.'
            ),
        },
        {"role": "user", "content": f"Ответ:\n{answer}\n\nИсточники:\n{listing}"},
    ]
    try:
        raw = await llm_router.chat_completion_text(
            runtime=runtime,
            messages=messages,
            stage="rerank",
            temperature=0,
            max_tokens=64,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "source_check", "schema": KEEP_SCHEMA, "strict": True},
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=15,
        )
        keep = json.loads(raw)["keep"]
        idx = {int(i) for i in keep}
        return [s for i, s in enumerate(sources) if i in idx]
    except Exception:  # noqa: BLE001 - any failure leaves sources unchanged
        log.warning("source llm_check failed", exc_info=True)
        return sources
