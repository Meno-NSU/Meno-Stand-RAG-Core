"""A streaming response that frees the chat slots however it ends.

The chat generators free their slots (the shared admission slot and a stand's) in their own `finally`. An async
generator that never ran a step never runs it, not even on aclose or garbage collection, and Starlette awaits
send(http.response.start) before it first iterates the body: a client that leaves at that moment (BaseHTTPMiddleware
then cancels the app) would leave the slots taken for good. One that leaves mid-stream leaves the generator suspended
until something closes it: Starlette never does, so the response closes it when it ends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anyio
import structlog
from starlette.responses import ContentStream, StreamingResponse
from starlette.types import Receive, Scope, Send

logger = structlog.get_logger(__name__)


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
            try:
                # A client that left while the body was paused at a yield (a full socket buffer, a failed send) leaves
                # the generator suspended: its `finally` (store the stopped answer, close the upstream stream, free
                # the slots) would run only when the garbage collector got to it, maybe after shutdown. Shielded:
                # this runs inside the scope the disconnect cancelled. A finished generator closes as a no-op.
                aclose = getattr(self.body_iterator, "aclose", None)
                if aclose is not None:
                    with anyio.CancelScope(shield=True):
                        await aclose()
            except Exception as exc:
                logger.warning("stream_body_close_failed", error=str(exc))
            finally:
                self._release()
