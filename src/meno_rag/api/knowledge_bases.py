"""Registry of the knowledge bases (answer pipelines) the API serves."""

from __future__ import annotations

from dataclasses import dataclass

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
    description=(
        "Выпуск базы за ноябрь 2026 (kb/chunks/v1, f374a7ed) с гибридным поиском "
        "и кросс-энкодером."
    ),
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
