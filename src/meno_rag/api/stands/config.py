"""The stand list: a JSON file on the server, its path in STANDS_FILE.

  [{"id": "ragu", "url": "http://host:9006", "api_key": "...", "arena": true, "max_concurrent": 16, "name": "..."}]

id: [a-z0-9][a-z0-9.-]{0,31}, assigned by us; url: http(s) base of the stand; api_key, arena (default false),
max_concurrent (default 32) and name (shown for an endpoint without a manifest) are optional."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

STAND_ID = re.compile(r"[a-z0-9][a-z0-9.-]{0,31}")
URL = re.compile(r"https?://\S+")
DEFAULT_MAX_CONCURRENT = 32


@dataclass(frozen=True)
class StandEntry:
    id: str
    url: str
    api_key: str = ""
    arena: bool = False
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    name: str = ""

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def parse_stands(raw: str, *, reserved_ids: frozenset[str] = frozenset()) -> list[StandEntry]:
    """The content of the stands file; ValueError naming the entry on any mistake."""
    if not raw or not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the stands file is not valid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("the stands file must be a JSON list of stands")
    stands: list[StandEntry] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"stand {item!r} is not an object")
        sid, url = item.get("id"), item.get("url")
        api_key, name = item.get("api_key") or "", item.get("name") or ""
        arena, limit = item.get("arena", False), item.get("max_concurrent", DEFAULT_MAX_CONCURRENT)
        if not isinstance(sid, str) or not STAND_ID.fullmatch(sid):
            raise ValueError(f"stand id {sid!r}: use 1-32 lowercase letters, digits, dots and dashes")
        if sid in reserved_ids:
            raise ValueError(f"stand id {sid!r} is reserved for our own models")
        if any(s.id == sid for s in stands):
            raise ValueError(f"stand id {sid!r} is used twice")
        if not isinstance(url, str) or not URL.fullmatch(url):
            raise ValueError(f"stand {sid!r}: url must start with http:// or https://")
        if not isinstance(api_key, str) or not isinstance(name, str):
            raise ValueError(f"stand {sid!r}: api_key and name must be strings")
        if not isinstance(arena, bool):
            raise ValueError(f"stand {sid!r}: arena must be true or false")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"stand {sid!r}: max_concurrent must be a positive integer")
        stands.append(StandEntry(sid, url.rstrip("/"), api_key, arena, limit, name.strip()))
    return stands


def read_stands_file(path: str, *, reserved_ids: frozenset[str] = frozenset()) -> list[StandEntry]:
    """An empty path means no stands; a path that cannot be read is an error."""
    if not path:
        return []
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read the stands file {path!r}: {exc}") from exc
    return parse_stands(raw, reserved_ids=reserved_ids)
