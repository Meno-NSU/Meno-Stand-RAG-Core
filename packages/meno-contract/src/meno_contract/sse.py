"""Server-Sent Events parsing shared by the gateway and the conformance checker.

No third-party imports on purpose: this module must work without fastapi or uvicorn.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class SseBlock:
    event: str
    data: str
    raw_size: int  # size of the block as received, in UTF-8 bytes


class SseBlockTooLarge(Exception):
    """A single SSE block grew past the allowed size."""


def parse_block(text: str) -> SseBlock | None:
    """Parse one block (lines without the closing blank line).

    Returns None for blocks that are empty or hold only comments (": ping").
    """
    event = "message"
    data_lines: list[str] = []
    seen_field = False
    for line in text.split("\n"):
        if not line or line.startswith(":"):
            continue
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "event":
            event = value or "message"
            seen_field = True
        elif name == "data":
            data_lines.append(value)
            seen_field = True
    if not seen_field:
        return None
    return SseBlock(event=event, data="\n".join(data_lines), raw_size=len(text.encode("utf-8")))


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _check_size(text: str, max_bytes: int) -> None:
    # A character is at most 4 bytes, so only encode when the cheap bounds are inconclusive.
    if len(text) > max_bytes or (len(text) * 4 > max_bytes and len(text.encode("utf-8")) > max_bytes):
        raise SseBlockTooLarge(f"SSE block is larger than {max_bytes} bytes")


async def iter_blocks(pieces: AsyncIterator[str], *, max_bytes: int) -> AsyncGenerator[SseBlock, None]:
    """Turn decoded text pieces from the network into SSE blocks.

    Pieces may split a block (or a CRLF pair) anywhere. A trailing CR is held back until the
    next piece shows whether it is part of CRLF. Raises SseBlockTooLarge when a block, even
    an unfinished one, exceeds max_bytes.
    """
    buffer = ""
    held_cr = ""
    async for piece in pieces:
        text = held_cr + piece
        held_cr = ""
        if text.endswith("\r"):
            text, held_cr = text[:-1], "\r"
        buffer += _normalize(text)
        *complete, buffer = buffer.split("\n\n")
        for raw in complete:
            _check_size(raw, max_bytes)
            block = parse_block(raw)
            if block is not None:
                yield block
        _check_size(buffer, max_bytes)
    buffer += _normalize(held_cr)
    # The stream may end without the final blank line: the rest is still a block.
    for raw in buffer.split("\n\n"):
        _check_size(raw, max_bytes)
        block = parse_block(raw)
        if block is not None:
            yield block
