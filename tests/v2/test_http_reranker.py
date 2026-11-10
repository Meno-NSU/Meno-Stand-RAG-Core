import json

import httpx
import pytest

from meno_rag.v2.rerank import HttpReranker, load_reranker


def _make(handler, **kw):
    r = HttpReranker("http://rr:8002/", "BAAI/bge-reranker-v2-m3", **kw)
    r._client = httpx.Client(transport=httpx.MockTransport(handler))
    return r


def _echo(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    docs = body["documents"]
    res = [{"index": i, "relevance_score": float(d)} for i, d in enumerate(docs)]
    res.sort(key=lambda x: -x["relevance_score"])
    return httpx.Response(200, json={"results": res})


def test_scores_in_input_order_when_service_sorts():
    assert _make(_echo).score("q", ["0.1", "0.9", "0.5"]) == [0.1, 0.9, 0.5]


def test_batches_of_32_and_url():
    calls = []

    def handler(request):
        calls.append(request)
        return _echo(request)

    texts = [str(i / 100) for i in range(70)]
    out = _make(handler).score("q", texts)
    assert out == [i / 100 for i in range(70)]
    assert [len(json.loads(c.content)["documents"]) for c in calls] == [32, 32, 6]
    assert str(calls[0].url) == "http://rr:8002/v1/rerank"
    assert json.loads(calls[0].content)["model"] == "BAAI/bge-reranker-v2-m3"


def test_truncate_sent_only_when_set():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _echo(request)

    _make(handler).score("q", ["0.1"])
    _make(handler, truncate_tokens=512).score("q", ["0.1"])
    assert "truncate_prompt_tokens" not in seen[0]
    assert seen[1]["truncate_prompt_tokens"] == 512


def test_http_error_raises():
    r = _make(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(RuntimeError, match="rerank service"):
        r.score("q", ["a"])


def test_connect_error_raises():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(RuntimeError, match="rerank service"):
        _make(handler).score("q", ["a"])


def test_incomplete_results_raise():
    r = _make(lambda req: httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]}))
    with pytest.raises(RuntimeError):
        r.score("q", ["a", "b"])


def test_empty_texts_no_call():
    def handler(req):
        raise AssertionError("called")

    assert _make(handler).score("q", []) == []


def test_load_reranker_http():
    r = load_reranker("http", "", "cpu", url="http://x:1", model="m", truncate_tokens=256)
    assert isinstance(r, HttpReranker)
    assert r.truncate_tokens == 256
    with pytest.raises(ValueError):
        load_reranker("http", "", "cpu")
