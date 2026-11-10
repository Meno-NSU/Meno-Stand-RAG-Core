"""Who took part: a stand, its version and an LLM. The server decides it, never the client.

A stand's history rows and ratings carry the label `<stand>@<version>` as the knowledge base. v1 and v2 run inside
this process and are described by the same kind of record, with one difference: what is stored and rated for them is
their knowledge base id, so that their existing rating rows go on. Their stand and version reach only the metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import structlog

from meno_rag.api.knowledge_bases import LEGACY_KB_ID, V2_KB_ID, split_virtual_model
from meno_rag.api.stands.config import STAND_ID
from meno_rag.api.stands.registry import StandRegistry
from meno_rag.config import Settings

LEGACY_STAND_ID = "menon-1.0"
V2_STAND_ID = "menon-1.1"
UNKNOWN_VERSION = "unknown"
BUILTIN_KBS = frozenset({LEGACY_KB_ID, V2_KB_ID})
VERSION = re.compile(r"[A-Za-z0-9._+-]{1,64}")  # the contract's version pattern; keeps a label within 128 characters

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Participant:
    stand: str
    version: str
    llm: str
    builtin_kb: str | None = None  # set for our own pipelines only

    @property
    def kb(self) -> str:
        return self.builtin_kb if self.builtin_kb is not None else f"{self.stand}@{self.version}"


def participant_for(
    public_model: str,
    *,
    registry: StandRegistry | None,
    settings: Settings,
    kb_hint: str | None = None,
    listed: bool = False,
) -> Participant:
    """The participant behind a public model id, resolved now. `listed`: vLLM or OpenRouter lists the id, so it is
    one of our own models even if it has the form `<llm>@<stand>`."""
    target = registry.resolve(public_model) if registry is not None else None
    if target is not None:
        return Participant(target.backend.id, target.version, target.model)
    base, virtual = split_virtual_model(public_model)
    if virtual:
        return Participant(V2_STAND_ID, settings.v2_version, base or public_model, V2_KB_ID)
    llm, _, stand = public_model.rpartition("@")
    if not listed and llm and STAND_ID.fullmatch(stand):  # anything else is not a stand id and must not grow a label
        return Participant(stand, UNKNOWN_VERSION, llm)
    if kb_hint == V2_KB_ID:
        return Participant(V2_STAND_ID, settings.v2_version, public_model, V2_KB_ID)
    return Participant(LEGACY_STAND_ID, settings.legacy_version, public_model, LEGACY_KB_ID)


async def listed_by_provider(state: Any, public_model: str) -> bool:
    """Whether vLLM or OpenRouter lists the id now. Their cached lists are read, so a list is fetched only when its
    cache has run out, as the model resolution of every chat does anyway. A list that cannot be read lists nothing."""
    for name in ("vllm_registry", "openrouter_registry"):
        provider = getattr(state, name, None)
        if provider is None:
            continue
        try:
            models = await provider.list_models()
        except Exception as exc:
            logger.warning("provider_model_list_failed", registry=name, error=str(exc))
            continue
        if any(isinstance(m, dict) and m.get("id") == public_model for m in models):
            return True
    return False


async def current_participant(
    public_model: str, *, state: Any, registry: StandRegistry | None, settings: Settings, kb_hint: str | None = None
) -> Participant:
    """participant_for, with the provider lists asked only for an id of the stand form that no stand serves now."""
    participant = participant_for(public_model, registry=registry, settings=settings, kb_hint=kb_hint)
    if participant.builtin_kb is not None or (registry is not None and registry.resolve(public_model) is not None):
        return participant
    if await listed_by_provider(state, public_model):
        return participant_for(public_model, registry=registry, settings=settings, kb_hint=kb_hint, listed=True)
    return participant


def stored_participant(public_model: str, stored_kb: str | None) -> Participant | None:
    """The participant a round was stored with; None if the stored label is not of the form `<stand>@<version>`."""
    if not stored_kb or "@" not in stored_kb:
        return None
    stand, _, version = stored_kb.rpartition("@")
    if not STAND_ID.fullmatch(stand) or not VERSION.fullmatch(version):
        return None
    llm = public_model.rpartition("@")[0] if "@" in public_model else public_model
    return Participant(stand, version, llm)


def rating_kb(participant: Participant, client_kb: str | None) -> str:
    """The knowledge base stored and rated for a participant. Today's frontend names our bases itself and their rows
    go on under those names; no other client value is taken, so a vote cannot reach a stand's row through our model."""
    if participant.builtin_kb is not None and client_kb in (LEGACY_KB_ID, V2_KB_ID):
        return client_kb
    return participant.kb


def builtin_live_kbs(settings: Settings) -> set[str]:
    """The knowledge base ids of our own pipelines that are served now."""
    live = set()
    if settings.legacy_enabled:
        live.add(LEGACY_KB_ID)
    if settings.v2_enabled:
        live.add(V2_KB_ID)
    return live
