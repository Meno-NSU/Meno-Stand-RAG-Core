"""Detect and strip <think>...</think> reasoning blocks emitted by Qwen3 and
other thinking models. Used by observability code to log when a response is
dominated by reasoning rather than visible content."""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def extract_thinking(content: str) -> tuple[str, str]:
    """Return (thinking_text, visible_text).

    - Closed `<think>...</think>` blocks are collected into thinking_text.
    - If a `<think>` opens but never closes (truncated by max_tokens), the
      remainder of the response is treated as thinking and visible_text is "".
    """
    if not content or "<think" not in content.lower():
        return "", content or ""

    thinking_parts: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        thinking_parts.append(match.group(0))
        return ""

    visible = _THINK_BLOCK_RE.sub(_capture, content)

    # Handle unclosed <think> tag (response was cut off mid-thinking).
    open_match = _THINK_OPEN_RE.search(visible)
    if open_match is not None:
        thinking_parts.append(visible[open_match.start() :])
        visible = visible[: open_match.start()]

    return "".join(thinking_parts), visible.strip()


def has_thinking(content: str | None) -> bool:
    return content is not None and "<think" in content.lower()
