import json
from pathlib import Path
from typing import Optional

from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.prompts import FEW_SHOTS, REWRITING_SYSTEM_PROMPT
from meno_rag.stand.tokenization import tokenize_and_normalize_text

__all__ = [
    "REWRITING_SYSTEM_PROMPT",
    "FEW_SHOTS",
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
            explanations = sorted(list(explanations_set))
        else:
            raise OSError(f'The file "{src_fname}" contains a wrong record: {k}: {src_abbr[k]}.')
        if (len(prep_k) == 0) or (len(explanations) == 0):
            raise OSError(f'The file "{src_fname}" contains a wrong record: {k}: {src_abbr[k]}.')
        if prep_k in prep_abbr:
            raise OSError(f'The file "{src_fname}" contains a duplicated abbreviation {k}.')
        prep_abbr[prep_k] = {"abbreviation": k, "explanation": sorted(list(explanations))}
    return prep_abbr


def find_candidates_to_abbreviations(
    source_text: str,
    all_abbreviations: dict[str, dict[str, str | list[str]]],
    stemmer: Optional[SnowballStemmer] = None,
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
    for src_abbr in sorted(list(selected_abbreviations)):
        explanations = all_abbreviations[src_abbr]["explanation"]
        assert isinstance(explanations, list)
        for val in explanations:
            explanation += f"\n{all_abbreviations[src_abbr]['abbreviation']} — {val}"
    return explanation.strip()


def prepare_prompt_for_rewriting(
    cur_question: str,
    dialogue_history: str,
    abbr_dict: dict[str, dict[str, str | list[str]]],
    stemmer: Optional[SnowballStemmer] = None,
    few_shots: Optional[list[dict[str, str]]] = None,
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


def parse_rewritten_queries(text: str) -> list[str]:
    return [line.strip() for line in text.strip().split("\n") if line.strip()]
