"""A chat stream frees its slots (the shared admission slot and, for a stand, the stand's) once, however its response
ends. Starlette awaits send(http.response.start) before it first iterates the generator, so a client that leaves at that
moment cancels the response before the generator, and the `finally` that frees the slots, has ever run.

The endpoint's own response object is driven here with a fake scope, receive and send."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import suppress

import pytest
from fastapi import Request, Response
from starlette.requests import ClientDisconnect

from meno_rag.api import main as main_mod
from meno_rag.schemas import ChatCompletionRequest
from tests.stands._helpers import stand_client, stand_handler
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)

QUESTION = [{"role": "user", "content": "Когда сессия?"}]


def _scope(c, spec_version="2.4"):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
        "app": c.app,
        "state": {},
    }


def _setup(make_client, tmp_path, path):  # noqa: F811
    """A client, the model of a stream request on `path` and the slots that request takes."""
    if path == "stand":
        c, _, reg = stand_client(make_client, tmp_path, stand_handler([]))
        return c, "gemma@lab", [c.app.state.admission, reg.resolve("gemma@lab").limiter]
    c, _ = make_client()
    return c, "menon-1", [c.app.state.admission]


async def _response(c, model, scope):
    """The endpoint's response to a stream request, before anything is sent."""
    payload = ChatCompletionRequest(model=model, stream=True, messages=QUESTION)
    return await main_mod.chat_completions(payload, Request(scope), Response())


async def _never_disconnects():
    await asyncio.Event().wait()


@pytest.mark.parametrize("path", ["stand", "builtin"])
@pytest.mark.parametrize("how,spec_version", [("send_raises", "2.4"), ("cancelled", "2.4"), ("cancelled", "2.0")])
def test_a_stream_cancelled_before_it_starts_frees_its_slots(make_client, tmp_path, path, how, spec_version):  # noqa: F811
    c, model, slots = _setup(make_client, tmp_path, path)
    scope = _scope(c, spec_version)

    async def run():
        response = await _response(c, model, scope)
        taken = [s.active for s in slots]
        at_start = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.start":
                if how == "send_raises":
                    raise asyncio.CancelledError
                at_start.set()
                await asyncio.Event().wait()  # the client is gone: the task is cancelled here

        task = asyncio.create_task(response(scope, _never_disconnects, send))
        if how == "cancelled":
            await asyncio.wait_for(at_start.wait(), 5)
            task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        return taken, inspect.getasyncgenstate(response.body_iterator)

    taken, generator_state = c.portal.call(run)
    # It never ran a step, so only the response can have freed the slots; the response then closed it.
    assert taken == [1] * len(slots) and generator_state == inspect.AGEN_CLOSED
    assert [s.active for s in slots] == [0] * len(slots)


@pytest.mark.parametrize("path", ["stand", "builtin"])
def test_a_stream_whose_client_left_before_it_started_frees_its_slots(make_client, tmp_path, path):  # noqa: F811
    # uvicorn sends spec_version 2.3: Starlette then listens for the disconnect next to the stream in a task group, and
    # a disconnect cancels the group, so the response returns normally and no exception reaches the release.
    c, model, slots = _setup(make_client, tmp_path, path)
    scope = _scope(c, "2.3")

    async def run():
        response = await _response(c, model, scope)
        taken = [s.active for s in slots]

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                await asyncio.Event().wait()  # the client is gone before the headers are out

        await asyncio.wait_for(response(scope, receive, send), 5)  # returns without an exception
        return taken, inspect.getasyncgenstate(response.body_iterator)

    taken, generator_state = c.portal.call(run)
    # It never ran a step, so only the response can have freed the slots; the response then closed it.
    assert taken == [1] * len(slots) and generator_state == inspect.AGEN_CLOSED
    assert [s.active for s in slots] == [0] * len(slots)


@pytest.mark.parametrize("path", ["stand", "builtin"])
def test_a_full_stream_frees_its_slots_once(make_client, tmp_path, path):  # noqa: F811
    c, model, slots = _setup(make_client, tmp_path, path)
    scope = _scope(c)

    async def run():
        for s in slots:  # another request holds one of each: a second release would free it
            assert s.try_acquire()
        response = await _response(c, model, scope)
        sent = []

        async def send(message):
            sent.append(message)

        await asyncio.wait_for(response(scope, _never_disconnects, send), 10)
        return sent

    sent = c.portal.call(run)
    assert b"data: [DONE]" in b"".join(m.get("body", b"") for m in sent)
    assert sent[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
    assert [s.active for s in slots] == [1] * len(slots)


@pytest.mark.parametrize("path", ["stand", "builtin"])
def test_a_client_that_leaves_mid_stream_frees_its_slots_once(make_client, tmp_path, path):  # noqa: F811
    c, model, slots = _setup(make_client, tmp_path, path)
    scope = _scope(c)

    async def run():
        for s in slots:  # another request holds one of each: a second release would free it
            assert s.try_acquire()
        response = await _response(c, model, scope)

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("the client closed the connection")

        with suppress(ClientDisconnect):
            await asyncio.wait_for(response(scope, _never_disconnects, send), 10)
        when_the_response_ended = [s.active for s in slots]
        await response.body_iterator.aclose()  # the generator's own finally, which also frees the slots
        return when_the_response_ended

    when_the_response_ended = c.portal.call(run)
    assert when_the_response_ended == [1] * len(slots)
    assert [s.active for s in slots] == [1] * len(slots)


@pytest.mark.parametrize("path", ["menon-1", "stand"])
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_the_response_closes_its_generator_when_the_client_leaves(make_client, tmp_path, path, spec_version):  # noqa: F811
    # Starlette never closes the body: a client that leaves while the body is paused at a yield (a slow client whose
    # socket buffer is full, then gone; or a send that fails) would leave the generator suspended until the garbage
    # collector got to it, and only then store the stopped answer and close the upstream stream.
    c, model, slots = _setup(make_client, tmp_path, path)
    scope = _scope(c, spec_version)

    async def run():
        response = await _response(c, model, scope)
        stuck = asyncio.Event()
        bodies = 0

        async def send(message):
            nonlocal bodies
            if message["type"] != "http.response.body":
                return
            bodies += 1
            if bodies < 2:
                return
            if spec_version == "2.4":  # Starlette reads a failed send as the disconnect
                raise OSError("the client closed the connection")
            stuck.set()
            await asyncio.Event().wait()

        async def receive():  # with 2.3 Starlette listens for the disconnect beside the stream
            await stuck.wait()
            return {"type": "http.disconnect"}

        with suppress(ClientDisconnect):
            await asyncio.wait_for(response(scope, receive, send), 10)
        closed = inspect.getasyncgenstate(response.body_iterator) == inspect.AGEN_CLOSED
        await main_mod.drain_persist_tasks(5)
        return closed, [s.active for s in slots]

    assert c.portal.call(run) == (True, [0] * len(slots))
