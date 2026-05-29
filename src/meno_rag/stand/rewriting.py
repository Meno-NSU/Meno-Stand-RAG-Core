import json
import re
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer

from meno_rag.llm.think_detector import extract_thinking
from meno_rag.stand.prompts import FEW_SHOTS, REWRITING_SYSTEM_PROMPT
from meno_rag.stand.tokenization import tokenize_and_normalize_text

__all__ = [
    "FEW_SHOTS",
    "REWRITING_SYSTEM_PROMPT",
    "find_candidates_to_abbreviations",
    "load_abbreviations",
    "parse_rewritten_queries",
    "prepare_prompt_for_rewriting",
]


def load_abbreviations(src_fname: str | Path) -> dict[str, dict[str, str | list[str]]]:
    src_abbr = json.loads(Path(src_fname).read_text(encoding="utf-8", errors="ignore"))
    prep_abbr: dict[str, dict[str, str | list[str]]] = {}
    for k in src_abbr:
        prep_k = " ".join(k.strip().split()).strip().lower()
        if isinstance(src_abbr[k], str):
            explanations = [" ".join(src_abbr[k].strip().split()).strip()]
        elif isinstance(src_abbr[k], list):
            explanations_set = set()
            for val in src_abbr[k]:
                if not isinstance(val, str):
                    raise OSError(f'The file "{src_fname}" contains a wrong record: {k}: {src_abbr[k]}.')
                norm_val = " ".join(val.strip().split()).strip()
                if len(norm_val) == 0:
                    raise OSError(f'The file "{src_fname}" contains a wrong record: {k}: {src_abbr[k]}.')
                explanations_set.add(norm_val)
            explanations = sorted(explanations_set)
        else:
            raise OSError(f'The file "{src_fname}" contains a wrong record: {k}: {src_abbr[k]}.')
        if (len(prep_k) == 0) or (len(explanations) == 0):
            raise OSError(f'The file "{src_fname}" contains a wrong record: {k}: {src_abbr[k]}.')
        if prep_k in prep_abbr:
            raise OSError(f'The file "{src_fname}" contains a duplicated abbreviation {k}.')
        prep_abbr[prep_k] = {"abbreviation": k, "explanation": sorted(explanations)}
    return prep_abbr


def find_candidates_to_abbreviations(
    source_text: str,
    all_abbreviations: dict[str, dict[str, str | list[str]]],
    stemmer: SnowballStemmer | None = None,
) -> str:
    tokens = set(tokenize_and_normalize_text(source_text, stemmer).split())
    selected_abbreviations = set()
    for cur_token in tokens:
        if cur_token in all_abbreviations:
            selected_abbreviations.add(cur_token)
    if stemmer is not None:
        tokens = set(tokenize_and_normalize_text(source_text).split())
        for cur_token in tokens:
            if cur_token in all_abbreviations:
                selected_abbreviations.add(cur_token)
    if len(selected_abbreviations) == 0:
        return ""
    explanation = ""
    for src_abbr in sorted(selected_abbreviations):
        explanations = all_abbreviations[src_abbr]["explanation"]
        assert isinstance(explanations, list)
        for val in explanations:
            explanation += f"\n{all_abbreviations[src_abbr]['abbreviation']} — {val}"
    return explanation.strip()


def prepare_prompt_for_rewriting(
    cur_question: str,
    dialogue_history: str,
    abbr_dict: dict[str, dict[str, str | list[str]]],
    stemmer: SnowballStemmer | None = None,
    few_shots: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    prep_question = cur_question.strip()
    if len(prep_question) == 0:
        return []
    messages = [{"role": "system", "content": REWRITING_SYSTEM_PROMPT}]
    used_few_shots = FEW_SHOTS if few_shots is None else few_shots
    for cur_shot in used_few_shots:
        messages.append({"role": "user", "content": cur_shot["input"]})
        messages.append({"role": "assistant", "content": cur_shot["target"]})
    selected_abbreviations = find_candidates_to_abbreviations(
        cur_question + "\n" + dialogue_history, abbr_dict, stemmer
    )
    if len(selected_abbreviations) > 0:
        user_prompt = (
            "ИСТОРИЯ ДИАЛОГА:\n"
            + dialogue_history.strip()
            + "\n\nСЛОВАРЬ АББРЕВИАТУР:\n"
            + selected_abbreviations
            + "\n\nТЕКУЩИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n"
            + prep_question
        )
    else:
        user_prompt = (
            "ИСТОРИЯ ДИАЛОГА:\n"
            + dialogue_history.strip()
            + "\n\nСЛОВАРЬ АББРЕВИАТУР:"
            + "\n\nТЕКУЩИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n"
            + prep_question
        )
    messages.append({"role": "user", "content": user_prompt})
    return messages


# Bare structural tokens emitted by chat-template-aware models. Without
# filtering, a model that finishes its rewrite with a sentinel like
# `<|im_end|>` on its own line gives us a junk retrieval query.
_BARE_TAG_RE = re.compile(r"^<[/!|]?[A-Za-z0-9_ \-:|]{0,40}>$")
# Pure ordinal/bullet line ("1.", "2)", "- ") with no actual query text.
_BARE_BULLET_RE = re.compile(r"^[\s\-•*]*[0-9]*[.)]?\s*$")


def _looks_like_query(line: str) -> bool:
    """True if `line` looks like a real search query — not a structural
    artefact from the rewrite LLM.

    Rejected:
    - bare tags: `<think>`, `</think>`, `<|im_end|>`, `<assistant>`
    - bullets/ordinals with no payload: `1.`, `- `, `*`
    - very short lines (< 3 chars after strip) — no real query is that short
    """
    cleaned = line.strip()
    if len(cleaned) < 3:
        return False
    if _BARE_TAG_RE.match(cleaned):
        return False
    return not _BARE_BULLET_RE.match(cleaned)


def parse_rewritten_queries(text: str) -> list[str]:
    """Split a rewrite-stage LLM response into search queries.

    Robust to two failure modes that the reference research code didn't
    handle and that wreck retrieval quality with thinking models:

    1. `<think>...</think>` reasoning blocks (Qwen3, DeepSeek-R1, etc.) are
       stripped before splitting on newlines. Without this, lines like
       `<think>`, `Let me think about...`, `</think>` go straight into
       FAISS/BM25 as separate queries.
    2. Bare structural tokens (`<|im_end|>`, lone bullets) are dropped via
       `_looks_like_query`.

    Truncated reasoning blocks (`<think>` opened but never closed because
    we hit `max_tokens`) leave `visible_text == ""` — that returns [],
    which is the right answer: the model produced no usable queries.
    """
    _, visible = extract_thinking(text)
    return [line.strip() for line in visible.strip().split("\n") if _looks_like_query(line)]
