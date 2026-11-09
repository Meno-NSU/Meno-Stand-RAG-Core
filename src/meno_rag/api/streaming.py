"""A streaming response that frees the chat slots however it ends.

The chat generators free their slots (the shared admission slot and a stand's) in their own `finally`. An async
generator that never ran a step never runs it, not even on aclose or garbage collection, and Starlette awaits
send(http.response.start) before it first iterates the body: a client that leaves at that moment (BaseHTTPMiddleware
then cancels the app) would leave the slots taken for good. One that leaves mid-stream leaves the generator suspended
until something closes it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from starlette.responses import ContentStream, StreamingResponse
from starlette.types import Receive, Scope, Send


def once(release: Callable[[], None]) -> Callable[[], None]:
    """`release` on the first call; later calls do nothing. The generator and the response both call it."""
    done = False

    def release_once() -> None:
        nonlocal done
        if not done:
            done = True
            release()

    return release_once


class ReleasingStreamingResponse(StreamingResponse):
    """A StreamingResponse that calls `release` when it ends, however it ends. Give it the same `once` guard as the
    body generator, which releases first on the normal path (after storing the answer)."""

    def __init__(self, content: ContentStream, *, release: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()
