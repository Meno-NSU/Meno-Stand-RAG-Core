"""Registry of the knowledge bases (answer pipelines) the API serves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from meno_rag.api import metrics
from meno_rag.config import Settings

LEGACY_KB_ID = "nsu-stand-faiss-bm25"
V2_KB_ID = "nsu-kb-2026-11"


@dataclass(frozen=True)
class KnowledgeBase:
    id: str
    name: str
    description: str
    rag_engine_id: str


LEGACY_KB = KnowledgeBase(
    id=LEGACY_KB_ID,
    name="База НГУ (июль 2026) · прежний поиск",
    description="Корпус документов НГУ на июль 2026 года и прежний поиск по нему.",
    rag_engine_id="stand_rag",
)
V2_KB = KnowledgeBase(
    id=V2_KB_ID,
    name="База НГУ (ноябрь 2026) · новый поиск",
    description=("Выпуск базы за ноябрь 2026 (kb/chunks/v1, f374a7ed) с гибридным поиском и кросс-энкодером."),
    rag_engine_id="v2_rag",
)


def knowledge_bases(settings: Settings) -> list[KnowledgeBase]:
    result = []
    if settings.legacy_enabled:
        result.append(LEGACY_KB)
    if settings.v2_enabled:
        result.append(V2_KB)
    return result


def default_kb_id(settings: Settings) -> str:
    """The legacy base if enabled, else the v2 one."""
    return LEGACY_KB_ID if settings.legacy_enabled else V2_KB_ID


V2_MODEL_SUFFIX = "@menon-1.1"


def virtual_model_id(base_id: str) -> str:
    return f"{base_id}{V2_MODEL_SUFFIX}"


def split_virtual_model(model_id: str | None) -> tuple[str | None, bool]:
    """``(base_id, True)`` for a «Менон 1.1» virtual id, else ``(model_id, False)``."""
    if model_id and model_id.endswith(V2_MODEL_SUFFIX) and len(model_id) > len(V2_MODEL_SUFFIX):
        return model_id[: -len(V2_MODEL_SUFFIX)], True
    return model_id, False


def parse_model_aliases(raw: str) -> list[tuple[str, str]]:
    """``key=display name;...`` -> [(key, display name)] (V2_MODEL_ALIASES, LLM_DISPLAY_NAMES)."""
    out = []
    for part in raw.split(";"):
        base, sep, name = part.partition("=")
        if sep and base.strip() and name.strip():
            out.append((base.strip(), name.strip()))
    return out


def virtual_model_entries(settings: Settings, models: list[dict]) -> list[dict]:
    """Copies of the base model entries, offered as the v2 pipeline («Менон 1.1»)."""
    if not settings.v2_enabled or settings.v2_expose_as not in ("models", "both"):
        return []
    by_id = {m["id"]: m for m in models}
    entries = []
    for base_id, display in parse_model_aliases(settings.v2_model_aliases):
        base = by_id.get(base_id)
        if base is None:
            continue
        entries.append(
            {
                **base,
                "id": virtual_model_id(base_id),
                "display_name": display,
                "pipeline": "v2",
                "knowledge_base_id": V2_KB_ID,
                "stand_id": metrics.MENON_11,
                "stand_name": metrics.BUILTIN_STAND_NAMES[metrics.MENON_11],
                "llm_id": base_id,
                "llm_name": base.get("llm_name", base_id),
            }
        )
    return entries


def apply_llm_display_names(entries: list[dict], raw: str) -> None:
    """Sets each entry's ``llm_name`` from the ``LLM_DISPLAY_NAMES`` map, by its ``llm_id``."""
    names = dict(parse_model_aliases(raw))
    for entry in entries:
        entry["llm_name"] = names.get(entry.get("llm_id", ""), entry["llm_name"])


def label_rating_row(row: dict, registry: Any, llm_display_names: str) -> None:
    """Adds ``stand_id``, ``stand_name``, ``stand_version`` and ``llm_name`` to a leaderboard row, by its knowledge
    base and model. A stand that is no longer configured is named by its id."""
    kb = row["knowledge_base"]
    llm, virtual = split_virtual_model(row["model"])
    llm = llm or row["model"]
    version = None
    if virtual:
        stand_id = metrics.MENON_11
        name = metrics.BUILTIN_STAND_NAMES[stand_id]
    elif kb in (LEGACY_KB_ID, V2_KB_ID):
        stand_id = metrics.MENON_10 if kb == LEGACY_KB_ID else metrics.MENON_11
        name = metrics.BUILTIN_STAND_NAMES[stand_id]
    else:
        stand_id, _, version = kb.partition("@")
        version = version or None
        name = (registry.display_name(stand_id) if registry is not None else None) or stand_id
    row["stand_id"], row["stand_name"], row["stand_version"] = stand_id, name, version
    row["llm_name"] = dict(parse_model_aliases(llm_display_names)).get(llm, llm)
