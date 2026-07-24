from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def sanitize_validation_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Reduce Pydantic validation errors to what names the problem — ``loc``/``msg``/
    ``type`` — and nothing else. Pydantic v2 attaches the offending ``input`` (and
    sometimes ``ctx``) to each error; for a body like /v1/arena/turn that ``input`` is
    the full message content, which must never reach the logs (size + the content-preview
    policy) nor be echoed back verbatim."""
    return [{"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")} for err in errors]
