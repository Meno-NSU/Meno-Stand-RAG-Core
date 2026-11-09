"""The stands of the registry file and what they offer: manifests, entries of /v1/models, id resolution.

A stand with a manifest (GET /meno/v1/manifest) is a full stand; one that answers 404 there is a plain
OpenAI endpoint, listed through GET /v1/models, kept out of the arena. The lists are cached; a stand that
stops answering keeps its last list, marked unreachable. The file is re-read on refresh when it changed;
a broken file leaves the previous list in place (at the start: no stands). Nothing a stand sends can stall
or break the model list: discovery has a deadline and a size limit, and a request for the list never waits
for a refresh."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from meno_contract.models import LLM_ID_PATTERN, MAX_BLOCK_BYTES, MAX_LLMS, PROTOCOL, Manifest
from pydantic import ValidationError

from meno_rag.api import metrics
from meno_rag.api.admission import AdmissionController
from meno_rag.api.stands.config import StandEntry, read_stands_file

logger = structlog.get_logger(__name__)

PLAIN_VERSION = "plain"
MANIFEST_PATH = "/meno/v1/manifest"
FAILURE_RETRY_S = 10.0  # how soon a stand that did not answer is asked again
MAX_PLAIN_NAME_CHARS = 128  # as a manifest's LLM name
_LLM_ID = re.compile(LLM_ID_PATTERN)


class _DiscoveryError(Exception):
    """A discovery answer that cannot be used: an HTTP error or a body over the size limit."""


@dataclass(frozen=True)
class StandTarget:
    """One LLM of one stand: the stand, its own model id and our id of the entry."""

    backend: StandEntry
    model: str
    public_id: str
    version: str
    build: str | None
    manifest: Manifest | None
    limiter: AdmissionController
    requires_auth: bool

    @property
    def kb_id(self) -> str:
        """Stored as the knowledge base of the turn: the answer comes from that stand's own search."""
        return f"{self.backend.id}@{self.version}"

    @property
    def arena(self) -> bool:
        return self.backend.arena and self.manifest is not None


class _State:
    """What is known of one stand: its manifest or plain model list, and whether it answered last time."""

    def __init__(self, limiter: AdmissionController) -> None:
        self.limiter = limiter
        self.manifest: Manifest | None = None
        self.plain_models: list[dict[str, Any]] = []
        self.known = False  # a manifest or a model list was received at some point
        self.up = False
        self.problem: str | None = None


class StandRegistry:
    def __init__(
        self,
        path: str,
        *,
        http_client: httpx.AsyncClient,
        timeout: float = 5.0,
        cache_ttl: float = 300.0,
        reserved_ids: frozenset[str] = frozenset(),
        deadline: float | None = None,
    ) -> None:
        if os.environ.get("EXTERNAL_BACKENDS"):
            logger.warning("external_backends_ignored", detail="EXTERNAL_BACKENDS is no longer read; use STANDS_FILE")
        self.http = http_client
        self._path = path
        self._reserved = reserved_ids
        self._timeout = timeout
        # One stand's whole discovery (manifest, then perhaps /v1/models): a per-read timeout alone lets a stand that
        # drips its answer hold the start of the service, or a refresh, for as long as it likes.
        self._deadline = deadline if deadline is not None else 2 * timeout
        self._cache_ttl = cache_ttl
        self._mtime = self._stat()
        self._stands: dict[str, StandEntry] = {}
        self._states: dict[str, _State] = {}
        try:
            entries = read_stands_file(path, reserved_ids=reserved_ids)
        except ValueError as exc:
            # A mistake in one entry must not take the service down at its next restart: it starts with no stands
            # and serves our own pipelines. The file is read again once it changes (see _reload).
            metrics.record_stand_config_error()
            logger.error("stands_file_invalid", path=path, error=str(exc))
            entries = []
        self._apply(entries)
        self._valid_until = 0.0
        self._refreshing: asyncio.Task[None] | None = None
        self._background: asyncio.Task[None] | None = None

    def _stat(self) -> int | None:
        try:
            return os.stat(self._path).st_mtime_ns if self._path else None
        except OSError:
            return None

    def _apply(self, entries: list[StandEntry]) -> None:
        states: dict[str, _State] = {}
        for e in entries:
            old = self._states.get(e.id)
            if old is not None and old.limiter.max_concurrent == e.max_concurrent:
                states[e.id] = old
            else:
                state = _State(AdmissionController(e.max_concurrent))
                if old is not None:  # the limit changed: keep what is known of the stand
                    state.manifest, state.plain_models = old.manifest, old.plain_models
                    state.known, state.up, state.problem = old.known, old.up, old.problem
                states[e.id] = state
        self._stands = {e.id: e for e in entries}
        self._states = states

    def _reload(self) -> None:
        mtime = self._stat()
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            entries = read_stands_file(self._path, reserved_ids=self._reserved)
        except ValueError as exc:
            metrics.record_stand_config_error()
            logger.error("stands_file_reload_failed", path=self._path, error=str(exc))
            return
        self._apply(entries)
        logger.info("stands_file_reloaded", stands=list(self._stands))

    async def _get(self, stand: StandEntry, path: str) -> tuple[int, bytes]:
        """The status and, for a 200, the body, read as a stream and never past MAX_BLOCK_BYTES."""
        url = f"{stand.url}{path}"
        async with self.http.stream("GET", url, headers=stand.headers(), timeout=self._timeout) as r:
            if r.status_code != 200:
                return r.status_code, b""
            body = bytearray()
            async for piece in r.aiter_bytes():
                body.extend(piece)
                if len(body) > MAX_BLOCK_BYTES:
                    raise _DiscoveryError(f"{path} answered more than {MAX_BLOCK_BYTES} bytes")
            return 200, bytes(body)

    async def _discover(self, stand: StandEntry) -> None:
        state = self._states.get(stand.id)
        if state is None:  # the file dropped the stand after this discovery was planned
            return
        try:
            async with asyncio.timeout(self._deadline):
                status, body = await self._get(stand, MANIFEST_PATH)
                if status == 404:
                    await self._discover_plain(stand, state)
                elif status >= 500:
                    raise _DiscoveryError(f"the manifest request answered HTTP {status}")
                elif status != 200:
                    self._reject(stand, state, f"the manifest request answered HTTP {status}", violation=False)
                else:
                    try:
                        payload = json.loads(body)
                        if isinstance(payload, dict) and payload.get("protocol") != PROTOCOL:
                            raise ValueError(f"protocol {payload.get('protocol')!r} is not {PROTOCOL}")
                        manifest = Manifest.model_validate(payload)
                    except (ValueError, ValidationError) as exc:
                        self._reject(stand, state, f"bad manifest: {_first_line(exc)}", violation=True)
                    else:
                        state.manifest, state.plain_models = manifest, []
                        state.known, state.up, state.problem = True, True, None
                        logger.info("stand_manifest_loaded", stand=stand.id, version=manifest.version)
        except Exception as exc:  # a timeout of the deadline included
            state.up = False
            logger.warning("stand_discovery_failed", stand=stand.id, error=str(exc) or type(exc).__name__)
        metrics.set_stand_up(stand.id, state.up)

    async def _discover_plain(self, stand: StandEntry, state: _State) -> None:
        status, body = await self._get(stand, "/v1/models")
        if status != 200:
            raise _DiscoveryError(f"/v1/models answered HTTP {status}")
        payload = json.loads(body)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise _DiscoveryError("/v1/models has no data list")
        rows = [row for row in map(_plain_row, data) if row is not None]
        state.plain_models = rows[:MAX_LLMS]
        state.manifest = None
        state.known, state.up, state.problem = True, True, None
        logger.info("stand_models_discovered", stand=stand.id, count=len(state.plain_models))

    def _reject(self, stand: StandEntry, state: _State, reason: str, *, violation: bool) -> None:
        state.manifest, state.plain_models = None, []
        state.known, state.up, state.problem = False, False, reason
        logger.warning("stand_rejected", stand=stand.id, reason=reason)
        if violation:
            metrics.record_stand_violation(stand.id, "bad_manifest")

    async def refresh(self) -> None:
        """Re-reads the file if it changed and asks every stand again. Refreshes never overlap: a caller that comes
        while one runs (POST /v1/models/refresh during a background refresh) waits for it and starts none."""
        if self._refreshing is None or self._refreshing.done():
            self._refreshing = asyncio.create_task(self._refresh_now())
        # A caller that gives up (a cancelled request) must not cancel the refresh the others are waiting for.
        await asyncio.shield(self._refreshing)

    async def _refresh_now(self) -> None:
        self._reload()
        await asyncio.gather(*(self._discover(s) for s in list(self._stands.values())))
        ttl = self._cache_ttl if all(s.up for s in self._states.values()) else min(self._cache_ttl, FAILURE_RETRY_S)
        self._valid_until = time.monotonic() + ttl

    async def _refresh_in_background(self) -> None:
        try:
            await self.refresh()
        except Exception as exc:
            logger.error("stand_refresh_failed", error=str(exc) or type(exc).__name__)
            self._valid_until = time.monotonic() + FAILURE_RETRY_S
        finally:
            self._background = None

    async def entries(self) -> list[dict[str, Any]]:
        # A due refresh runs in the background, one at a time, and this answers at once with what is known: a stand
        # that is down must not stall the caller (the start of the service awaits the first refresh itself).
        if time.monotonic() > self._valid_until and self._background is None:
            self._background = asyncio.create_task(self._refresh_in_background())  # the attribute keeps it alive
        out: list[dict[str, Any]] = []
        for stand in self._stands.values():
            try:
                out.extend(self._entries_of(stand))
            except Exception as exc:  # one stand never takes the others, or our own models, out of the list
                logger.error("stand_entries_failed", stand=stand.id, error=str(exc) or type(exc).__name__)
                metrics.record_stand_violation(stand.id, "bad_model_list")
        return out

    def _entries_of(self, stand: StandEntry) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        state = self._states[stand.id]
        if state.known:
            m = state.manifest
            version = m.version if m else PLAIN_VERSION
            backend_name = (m.name if m else stand.name) or stand.id
            for llm in _llms(state):
                status = {"state": "available", "until": None, "last_error": None}
                if m is not None and llm.get("status"):  # a status is read only from a checked manifest
                    status["state"], status["until"] = llm["status"].state, llm["status"].until
                if not state.up:
                    status["state"] = "unreachable"
                out.append(
                    {
                        "id": f"{llm['id']}@{stand.id}",
                        "object": "model",
                        "created": llm.get("created") or int(time.time()),
                        "owned_by": stand.id,
                        "provider": "external",
                        "featured": False,
                        "stages": ["generation"],
                        "status": status,
                        "display_name": llm.get("display_name") or llm.get("name") or llm["id"],
                        "context_length": llm.get("context_length"),
                        "backend_id": stand.id,
                        "backend_name": backend_name,
                        "llm_id": llm["id"],
                        "version": version,
                        "arena": stand.arena and m is not None,
                        "auth_required": bool(llm.get("requires_auth")),
                    }
                )
        return out

    def resolve(self, public_id: str | None) -> StandTarget | None:
        if not public_id or "@" not in public_id:
            return None
        model, _, sid = public_id.rpartition("@")
        if not model:
            return None
        stand, state = self._stands.get(sid), self._states.get(sid)
        if stand is None or state is None or not state.known:
            return None
        llm = next((x for x in _llms(state) if x["id"] == model), None)
        if llm is None:
            return None
        m = state.manifest
        return StandTarget(
            backend=stand,
            model=model,
            public_id=public_id,
            version=m.version if m else PLAIN_VERSION,
            build=m.build if m else None,
            manifest=m,
            limiter=state.limiter,
            requires_auth=bool(llm.get("requires_auth")),
        )

    def live_kbs(self) -> set[str]:
        """The knowledge-base labels of the stand versions known now."""
        return {
            f"{sid}@{s.manifest.version if s.manifest else PLAIN_VERSION}" for sid, s in self._states.items() if s.known
        }

    def problems(self) -> dict[str, str]:
        return {sid: s.problem for sid, s in self._states.items() if s.problem}


def _llms(state: _State) -> list[dict[str, Any]]:
    """The LLMs of a stand as dicts, the default one first."""
    m = state.manifest
    if m is None:
        return list(state.plain_models)
    rows = [llm.model_dump() | {"status": llm.status} for llm in m.llms]
    return sorted(rows, key=lambda x: x["id"] != m.default_llm)  # stable: the rest keep the manifest order


def _plain_row(item: Any) -> dict[str, Any] | None:
    """The fields of a plain endpoint's model row that pass the checks; None for a row without a valid id."""
    if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not _LLM_ID.fullmatch(item["id"]):
        return None
    row: dict[str, Any] = {"id": item["id"]}
    for key in ("created", "context_length"):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            row[key] = value
    for key in ("display_name", "name"):
        value = item.get(key)
        if isinstance(value, str) and 0 < len(value) <= MAX_PLAIN_NAME_CHARS:
            row[key] = value
    return row


def _first_line(exc: Exception) -> str:
    lines = str(exc).strip().splitlines()
    return lines[-1] if len(lines) > 1 else (lines[0] if lines else type(exc).__name__)
