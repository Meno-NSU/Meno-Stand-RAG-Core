import json
import math
from typing import Any

from meno_rag.stand.prompts import SYSTEM_PROMPT_FOR_RELEVANCE

__all__ = [
    "MAX_RERANKER_TOKENS",
    "POSSIBLE_LABELS",
    "SYSTEM_PROMPT_FOR_RELEVANCE",
    "build_prompt",
    "rerank_merge_score",
    "response_format_schema",
    "score_from_json_response",
    "score_from_logprobs",
]

MAX_RERANKER_TOKENS: int = 8192
POSSIBLE_LABELS = ["0", "1", "2"]


def build_prompt(search_query: str, estimated_document: str, is_json: bool = False) -> list[dict[str, str]]:
    user_prompt = "**Поисковый запрос пользователя:** " + " ".join(search_query.strip().split()) + "\n\n"
    user_prompt += estimated_document.strip() + "\n\n**Ваша оценка релевантности:**\n"
    if is_json:
        modified_chapter_name = "## Формат ответа"
        found_idx = SYSTEM_PROMPT_FOR_RELEVANCE.find(modified_chapter_name)
        if found_idx < 0:
            raise RuntimeError(f"The system prompt does not contain the chapter `{modified_chapter_name}`.")
        modified_system_prompt = SYSTEM_PROMPT_FOR_RELEVANCE[:found_idx] + modified_chapter_name + "\n\n"
        modified_system_prompt += (
            'Ответьте в формате JSON вида `{"label": <your_label>}`, где '
            "`<your_label>` — это цифра 2, 1 или 0. Не добавляйте никаких пояснений, "
            "комментариев или дополнительного текста.\n"
        )
        return [{"role": "system", "content": modified_system_prompt}, {"role": "user", "content": user_prompt}]
    return [{"role": "system", "content": SYSTEM_PROMPT_FOR_RELEVANCE}, {"role": "user", "content": user_prompt}]


def score_from_logprobs(choice: dict[str, Any]) -> float:
    try:
        top_logprobs_list = choice["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Reranker response does not contain top_logprobs.") from exc

    token_to_logprob = {str(item["token"]): float(item["logprob"]) for item in top_logprobs_list if "token" in item}
    raw = {label: token_to_logprob.get(label, -100.0) for label in POSSIBLE_LABELS}
    max_lp = max(raw.values())
    exp_values = {k: math.exp(v - max_lp) for k, v in raw.items()}
    total = sum(exp_values.values())
    probs = {k: v / total for k, v in exp_values.items()}
    predicted_class = int(max(probs, key=lambda label: probs[label]))
    if predicted_class == 2:
        return probs["2"]
    return 0.0


def score_from_json_response(content: str) -> float:
    """Mirrors meno_stand rerank_utils.py:138 — return the raw numeric label
    (0.0, 1.0, or 2.0). Combined with rerank_merge_score (α=0.8), label=1
    chunks survive the >0 filter and label=2 chunks dominate ordering."""

    parsed = json.loads(content.strip())
    return float(parsed["label"])


def rerank_merge_score(retrieval_score: float, rerank_score: float, alpha: float) -> float:
    if rerank_score > 0.0:
        return (1.0 - alpha) * retrieval_score + alpha * rerank_score
    return rerank_score


def response_format_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "relevance",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["0", "1", "2"],
                    }
                },
                "required": ["label"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
