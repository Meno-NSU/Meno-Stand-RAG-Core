"""Who took part: a stand, its version and an LLM. The server decides it, never the client.

The label `<stand>@<version>` is what history rows, ratings and metrics carry as the knowledge base. v1 and v2
run inside this process and are described by the same kind of record."""

from __future__ import annotations

import re
from dataclasses import dataclass

from meno_rag.api.knowledge_bases import V2_KB_ID, split_virtual_model
from meno_rag.api.stands.config import STAND_ID
from meno_rag.api.stands.registry import StandRegistry
from meno_rag.config import Settings

LEGACY_STAND_ID = "menon-1.0"
V2_STAND_ID = "menon-1.1"
UNKNOWN_VERSION = "unknown"
VERSION = re.compile(r"[A-Za-z0-9._+-]{1,64}")  # the contract's version pattern; keeps a label within 128 characters


@dataclass(frozen=True)
class Participant:
    stand: str
    version: str
    llm: str

    @property
    def kb(self) -> str:
        return f"{self.stand}@{self.version}"


def participant_for(
    public_model: str, *, registry: StandRegistry | None, settings: Settings, kb_hint: str | None = None
) -> Participant:
    """The participant behind a public model id, resolved now."""
    target = registry.resolve(public_model) if registry is not None else None
    if target is not None:
        return Participant(target.backend.id, target.version, target.model)
    base, virtual = split_virtual_model(public_model)
    if virtual:
        return Participant(V2_STAND_ID, settings.v2_version, base or public_model)
    llm, _, stand = public_model.rpartition("@")
    if llm and STAND_ID.fullmatch(stand):  # anything else is not a stand id and must not grow a label
        return Participant(stand, UNKNOWN_VERSION, llm)
    if kb_hint == V2_KB_ID:
        return Participant(V2_STAND_ID, settings.v2_version, public_model)
    return Participant(LEGACY_STAND_ID, settings.legacy_version, public_model)


def stored_participant(public_model: str, stored_kb: str | None) -> Participant | None:
    """The participant a round was stored with; None if the stored label is not of the form `<stand>@<version>`."""
    if not stored_kb or "@" not in stored_kb:
        return None
    stand, _, version = stored_kb.rpartition("@")
    if not STAND_ID.fullmatch(stand) or not VERSION.fullmatch(version):
        return None
    llm = public_model.rpartition("@")[0] if "@" in public_model else public_model
    return Participant(stand, version, llm)


def builtin_live_kbs(settings: Settings) -> set[str]:
    """The labels of our own pipelines that are served now."""
    live = set()
    if settings.legacy_enabled:
        live.add(Participant(LEGACY_STAND_ID, settings.legacy_version, "").kb)
    if settings.v2_enabled:
        live.add(Participant(V2_STAND_ID, settings.v2_version, "").kb)
    return live
