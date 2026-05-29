"""Admission control for the chat endpoint.

Under overload the backend does not fail fast on its own — requests queue on
the per-stage semaphores and just get slow. With a frontend that has no request
timeout, that surfaces as an unbounded spinner. This bounds the number of chat
requests allowed to be in flight; beyond the limit the API returns a fast 503
with Retry-After instead of silently queueing forever.

Single-process / single-event-loop: `try_acquire`/`release` mutate the counter
without awaiting, so they are atomic w.r.t. the event loop and need no lock."""

from __future__ import annotations


class AdmissionController:
    def __init__(self, max_concurrent: int) -> None:
        # max_concurrent <= 0 disables the limit (unbounded, legacy behavior).
        self._max = max_concurrent
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def max_concurrent(self) -> int:
        return self._max

    def try_acquire(self) -> bool:
        if self._max > 0 and self._active >= self._max:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        if self._active > 0:
            self._active -= 1
