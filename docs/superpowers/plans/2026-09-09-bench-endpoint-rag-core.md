# Bench-режим RAG-Core — Implementation Plan (План A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Запрос с валидным `BENCH_API_TOKEN` получает от `/v1/chat/completions` полный трейс RAG-воронки инлайн, не пишет ничего в продовые таблицы и берёт слоты из отдельного бюджета конкурентности.

**Architecture:** Новых роутов нет. nginx (План B) отобразит `/bench/v1/*` → `/v1/*`, а существующий обработчик `chat_completions` ветвится по флагу, который вычисляется из `Authorization: Bearer`. Трейс уже собирается функцией `build_pipeline_trace` и лежит в `outcome.trace` — в bench-режиме мы его возвращаем вместо того, чтобы писать в trace-БД.

**Tech Stack:** Python 3, FastAPI, pydantic-settings, pytest + pytest-asyncio, `fastapi.testclient.TestClient`, `unittest.mock.AsyncMock`.

**Спека:** `docs/superpowers/specs/2026-09-09-bench-endpoint-and-monitoring-design.md`

---

## Структура файлов

**Создаются:**

| Файл | Ответственность |
|---|---|
| `src/meno_rag/api/bench.py` | Единственная ответственность: распознать bench-токен. Без побочных эффектов, без I/O. |
| `tests/test_bench_token.py` | Юнит-тесты распознавания. |
| `tests/test_bench_endpoint.py` | Интеграционные тесты поведения эндпоинта. |

**Модифицируются:**

| Файл | Что меняется |
|---|---|
| `src/meno_rag/config.py` | Две настройки: `bench_api_token`, `bench_max_concurrent`. |
| `src/meno_rag/api/main.py` | Второй `AdmissionController`; ветка bench в `chat_completions`, `_non_stream_response`, `_stream_response`. |
| `example.env` | Документация новых переменных. |

`bench.py` держится отдельным файлом намеренно: `main.py` уже 56 КБ, и распознавание секрета — самостоятельная, чисто тестируемая единица, которую не надо тащить через веб-слой.

---

## Task 1: Настройки bench-режима

**Files:**
- Modify: `src/meno_rag/config.py:152` (после блока Pipeline trace capture)
- Test: `tests/test_bench_token.py`

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_bench_token.py`:

```python
"""Распознавание токена бенчмарк-эндпоинта."""

from __future__ import annotations

from meno_rag.config import Settings


def test_bench_settings_default_to_disabled():
    s = Settings()
    assert s.bench_api_token == ""
    assert s.bench_max_concurrent == 4


def test_bench_settings_read_from_env():
    s = Settings(BENCH_API_TOKEN="secret-token", BENCH_MAX_CONCURRENT=7)
    assert s.bench_api_token == "secret-token"
    assert s.bench_max_concurrent == 7
```

- [ ] **Step 2: Запустить тест, убедиться что падает**

Run: `uv run pytest tests/test_bench_token.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'bench_api_token'`

- [ ] **Step 3: Добавить настройки**

В `src/meno_rag/config.py` сразу после строки `pipeline_trace_queue_max: int = Field(...)` и перед `auth_jwt_secret`:

```python
    # --- Benchmark endpoint (developers only, token-gated at the nginx edge) ---
    # Empty = feature off, and it can only be turned on deliberately. A request
    # carrying this token in `Authorization: Bearer` gets the full pipeline trace
    # inline and writes nothing to production tables.
    bench_api_token: str = Field(default="", validation_alias="BENCH_API_TOKEN")
    # Separate concurrency budget: a benchmark run must never 503 real users.
    # Small on purpose — benchmarks are throughput-insensitive, users are not.
    bench_max_concurrent: int = Field(default=4, validation_alias="BENCH_MAX_CONCURRENT")
```

- [ ] **Step 4: Запустить тест, убедиться что проходит**

Run: `uv run pytest tests/test_bench_token.py -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Коммит**

```bash
git add src/meno_rag/config.py tests/test_bench_token.py
git commit -m "feat(bench): add BENCH_API_TOKEN and BENCH_MAX_CONCURRENT settings"
```

---

## Task 2: Распознавание токена

**Files:**
- Create: `src/meno_rag/api/bench.py`
- Test: `tests/test_bench_token.py` (дополняется)

- [ ] **Step 1: Написать падающие тесты**

Дописать в `tests/test_bench_token.py`:

```python
from types import SimpleNamespace

from starlette.datastructures import Headers

from meno_rag.api.bench import is_bench_request


def _request(headers: dict[str, str], configured_token: str):
    """Минимальный дубль Request: используются только .headers и .app.state.settings."""
    settings = SimpleNamespace(bench_api_token=configured_token)
    app = SimpleNamespace(state=SimpleNamespace(settings=settings))
    return SimpleNamespace(headers=Headers(headers), app=app)


def test_valid_token_is_recognised():
    r = _request({"authorization": "Bearer secret-token"}, "secret-token")
    assert is_bench_request(r) is True


def test_bearer_prefix_is_case_insensitive():
    r = _request({"authorization": "bearer secret-token"}, "secret-token")
    assert is_bench_request(r) is True


def test_wrong_token_is_rejected():
    r = _request({"authorization": "Bearer wrong"}, "secret-token")
    assert is_bench_request(r) is False


def test_empty_setting_disables_the_feature():
    r = _request({"authorization": "Bearer anything"}, "")
    assert is_bench_request(r) is False


def test_missing_header_is_rejected():
    r = _request({}, "secret-token")
    assert is_bench_request(r) is False


def test_empty_bearer_value_is_rejected():
    r = _request({"authorization": "Bearer   "}, "secret-token")
    assert is_bench_request(r) is False


def test_non_bearer_scheme_is_rejected():
    r = _request({"authorization": "Basic secret-token"}, "secret-token")
    assert is_bench_request(r) is False


def test_x_auth_token_header_is_not_accepted():
    """X-Auth-Token принадлежит JWT приложения; bench-токен туда не ходит."""
    r = _request({"x-auth-token": "secret-token"}, "secret-token")
    assert is_bench_request(r) is False


def test_missing_settings_state_is_safe():
    """Никогда не бросает, даже если state ещё не заполнен."""
    r = SimpleNamespace(headers=Headers({"authorization": "Bearer x"}), app=SimpleNamespace(state=SimpleNamespace()))
    assert is_bench_request(r) is False
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_bench_token.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'meno_rag.api.bench'`

- [ ] **Step 3: Реализовать модуль**

Создать `src/meno_rag/api/bench.py`:

```python
"""Recognition of the developer benchmark token.

nginx already 404s the benchmark path without a valid token, so by the time a
request reaches here the edge has checked it. This module checks it a second
time on purpose: nginx decides *reachability*, the application needs the flag to
decide *behavior* — return the full pipeline trace, write nothing to production
tables, and draw from a separate concurrency budget. It also means an SSH tunnel
straight to 127.0.0.1:9006 gets bench mode with no nginx in the path.
"""

from __future__ import annotations

import hmac

from fastapi import Request


def _bearer_token(request: Request) -> str | None:
    """The Bearer credential from ``Authorization``, or None.

    Deliberately does NOT read ``X-Auth-Token``: that header carries the
    application's JWT, and letting two different kinds of secret share one header
    is how they eventually get mistaken for each other.
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip() or None


def is_bench_request(request: Request) -> bool:
    """True when the request carries the configured benchmark token. Never raises."""
    settings = getattr(request.app.state, "settings", None)
    expected = getattr(settings, "bench_api_token", "") or ""
    if not expected:
        return False
    token = _bearer_token(request)
    if token is None:
        return False
    return hmac.compare_digest(token, expected)
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `uv run pytest tests/test_bench_token.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Коммит**

```bash
git add src/meno_rag/api/bench.py tests/test_bench_token.py
git commit -m "feat(bench): recognise the benchmark token from Authorization: Bearer"
```

---

## Task 3: Отдельный бюджет конкурентности

**Files:**
- Modify: `src/meno_rag/api/main.py:281`
- Test: `tests/test_bench_endpoint.py`

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_bench_endpoint.py`:

```python
"""Поведение /v1/chat/completions в bench-режиме."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meno_rag.api.admission import AdmissionController

BENCH_TOKEN = "test-bench-token-0123456789"


@pytest.mark.asyncio
async def test_lifespan_installs_a_separate_bench_admission_controller():
    from meno_rag.api.main import app

    with TestClient(app):
        assert isinstance(app.state.bench_admission, AdmissionController)
        assert app.state.bench_admission is not app.state.admission
        assert app.state.bench_admission.max_concurrent == 4
```

- [ ] **Step 2: Запустить тест, убедиться что падает**

Run: `uv run pytest tests/test_bench_endpoint.py -v`
Expected: FAIL — `AttributeError: 'State' object has no attribute 'bench_admission'`

- [ ] **Step 3: Завести второй контроллер**

В `src/meno_rag/api/main.py` сразу после строки

```python
    app.state.admission = AdmissionController(settings.max_concurrent_chats)
```

добавить:

```python
    # Benchmark traffic draws from its own budget so a dev's benchmark run can
    # never exhaust the pool that serves real users.
    app.state.bench_admission = AdmissionController(settings.bench_max_concurrent)
```

- [ ] **Step 4: Запустить тест, убедиться что проходит**

Run: `uv run pytest tests/test_bench_endpoint.py -v`
Expected: PASS, 1 passed

- [ ] **Step 5: Коммит**

```bash
git add src/meno_rag/api/main.py tests/test_bench_endpoint.py
git commit -m "feat(bench): install a separate admission controller for benchmark traffic"
```

---

## Task 4: Ветка bench в обработчике

**Files:**
- Modify: `src/meno_rag/api/main.py:652-706` (`chat_completions`)
- Test: `tests/test_bench_endpoint.py` (дополняется)

- [ ] **Step 1: Написать падающий тест**

Дописать в `tests/test_bench_endpoint.py` (импорты в начало файла):

```python
from types import SimpleNamespace
from unittest.mock import AsyncMock

from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime


def _fake_outcome():
    return SimpleNamespace(
        question="Когда начинается сессия?",
        sources=[{"document_title": "Устав НГУ", "source_url": "https://nsu.ru/x"}],
        stage_durations_ms={"rewrite": 12.0, "retrieval": 34.0},
        stage_details={},
        qa_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
        search_queries=["начало сессии"],
        trace={
            "retrieval": {"per_query": [{"query": "начало сессии", "dense": [], "lexical": []}]},
            "fusion": {"per_query": []},
            "rerank": [{"chunk_id": 1, "score": 0.9, "rank": 0}],
            "prompt": {"system": "sys", "user": "usr"},
            "chunks": {"1": {"title": "Устав НГУ", "url": "https://nsu.ru/x", "text": "..."}},
        },
    )


@pytest.fixture
def bench_client(monkeypatch):
    """TestClient с включённым bench-токеном и полностью замоканным пайплайном."""
    monkeypatch.setenv("BENCH_API_TOKEN", BENCH_TOKEN)
    from meno_rag.config import get_settings

    get_settings.cache_clear()
    from meno_rag.api import main as main_mod

    runtime = PipelineRuntime.uniform(
        ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://fake/v1")
    )
    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(return_value=runtime))

    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = AsyncMock()
        c.app.state.pipeline.prepare = AsyncMock(return_value=_fake_outcome())
        c.app.state.pipeline.generate_text = AsyncMock(return_value="Сессия начинается в январе.")
        yield c

    get_settings.cache_clear()


def _post(client, *, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/v1/chat/completions",
        json={"model": "menon-1", "messages": [{"role": "user", "content": "Когда сессия?"}]},
        headers=headers,
    )


def test_bench_request_takes_a_slot_from_the_bench_budget(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())
    saturated = AdmissionController(1)
    assert saturated.try_acquire() is True  # занять единственный слот
    bench_client.app.state.bench_admission = saturated

    r = _post(bench_client, token=BENCH_TOKEN)

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "overloaded"


def test_normal_request_is_unaffected_by_a_saturated_bench_budget(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())
    saturated = AdmissionController(1)
    assert saturated.try_acquire() is True
    bench_client.app.state.bench_admission = saturated

    r = _post(bench_client, token=None)

    assert r.status_code == 200
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_bench_endpoint.py -v`
Expected: FAIL — `test_bench_request_takes_a_slot_from_the_bench_budget` вернёт 200 вместо 503 (bench-запрос сейчас берёт слот из общего пула).

- [ ] **Step 3: Реализовать ветвление**

В `src/meno_rag/api/main.py` добавить импорт к строке 21:

```python
from meno_rag.api import arena, auth, bench as bench_mod, feedback, guest, history, legal, privacy
```

Заменить блок admission в `chat_completions`:

```python
    # Admission control: fast-fail under overload rather than queueing forever.
    admission: AdmissionController = request.app.state.admission
    if not admission.try_acquire():
        metrics_mod.record_error("overloaded")
        return _overloaded_response(active=admission.active, limit=admission.max_concurrent)
```

на:

```python
    # Benchmark requests are recognised before anything else: they pick a
    # different admission budget, force trace capture, and skip persistence.
    bench = bench_mod.is_bench_request(request)

    # Admission control: fast-fail under overload rather than queueing forever.
    admission: AdmissionController = (
        request.app.state.bench_admission if bench else request.app.state.admission
    )
    if not admission.try_acquire():
        if not bench:
            metrics_mod.record_error("overloaded")
        return _overloaded_response(active=admission.active, limit=admission.max_concurrent)
```

Заменить строку вычисления `capture_trace`:

```python
        capture_trace = settings.capture_pipeline_trace and random.random() < settings.pipeline_trace_sample_rate
```

на:

```python
        # Benchmarks always get the trace: it is the reason the endpoint exists,
        # and it must not depend on the production capture toggle or sample rate.
        capture_trace = bench or (
            settings.capture_pipeline_trace and random.random() < settings.pipeline_trace_sample_rate
        )
```

Прокинуть флаг в оба вызова — добавить `bench=bench,` в аргументы `_stream_response(...)` (рядом с `capture_trace=capture_trace,`) и `_non_stream_response(...)` (там же).

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `uv run pytest tests/test_bench_endpoint.py -v`
Expected: FAIL с `TypeError: _non_stream_response() got an unexpected keyword argument 'bench'` — параметр добавляется в Task 5. Это ожидаемое промежуточное состояние; коммит делается после Task 5.

- [ ] **Step 5: Не коммитить**

Ветвление и приёмники флага — одно изменение. Коммит в Task 5, Step 6.

---

## Task 5: Non-stream — трейс в ответе, ноль записи

**Files:**
- Modify: `src/meno_rag/api/main.py:770-880` (`_non_stream_response`)
- Test: `tests/test_bench_endpoint.py` (дополняется)

- [ ] **Step 1: Написать падающие тесты**

Дописать в `tests/test_bench_endpoint.py`:

```python
def test_bench_response_carries_the_full_pipeline_trace(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    r = _post(bench_client, token=BENCH_TOKEN)

    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Сессия начинается в январе."
    assert body["sources"][0]["document_title"] == "Устав НГУ"
    trace = body["pipeline_trace"]
    assert set(trace) == {"retrieval", "fusion", "rerank", "prompt", "chunks"}
    assert trace["prompt"]["system"] == "sys"


def test_normal_response_has_no_pipeline_trace(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    r = _post(bench_client, token=None)

    assert r.status_code == 200
    assert "pipeline_trace" not in r.json()


def test_bench_request_writes_nothing_to_production_tables(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    persist = AsyncMock()
    monkeypatch.setattr(main_mod, "_persist_success", persist)

    assert _post(bench_client, token=BENCH_TOKEN).status_code == 200
    assert persist.await_count == 0

    assert _post(bench_client, token=None).status_code == 200
    assert persist.await_count == 1


def test_bench_request_does_not_record_prometheus_metrics(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())
    from unittest.mock import Mock

    record = Mock()
    monkeypatch.setattr(main_mod.metrics_mod, "record_chat_request", record)

    assert _post(bench_client, token=BENCH_TOKEN).status_code == 200
    assert record.call_count == 0

    assert _post(bench_client, token=None).status_code == 200
    assert record.call_count == 1


def test_bench_trace_is_returned_even_when_capture_is_globally_off(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())
    bench_client.app.state.settings = bench_client.app.state.settings.model_copy(
        update={"capture_pipeline_trace": False, "pipeline_trace_sample_rate": 0.0}
    )

    r = _post(bench_client, token=BENCH_TOKEN)

    assert r.status_code == 200
    assert "pipeline_trace" in r.json()
    bench_client.app.state.pipeline.prepare.assert_awaited()
    assert bench_client.app.state.pipeline.prepare.await_args.kwargs["capture_trace"] is True
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_bench_endpoint.py -v`
Expected: FAIL — `TypeError: _non_stream_response() got an unexpected keyword argument 'bench'`

- [ ] **Step 3: Реализовать**

В сигнатуру `_non_stream_response` добавить параметр после `capture_trace: bool = False,`:

```python
    bench: bool = False,
```

Заменить `metrics_mod.inc_chat_in_flight()` на:

```python
    if not bench:
        metrics_mod.inc_chat_in_flight()
```

Обернуть успешную запись — заменить `await _persist_success(` … `)` целиком на:

```python
        if not bench:
            await _persist_success(
                database=database,
                run_id=completion_id,
                session_id=session_id,
                model=runtime.generation.model_id,
                generation_model=runtime.generation.model_id,
                core_model=runtime.core.model_id,
                endpoint=runtime.generation.base_url,
                question=outcome.question,
                answer=answer,
                outcome=outcome,
                generation_ms=generation_ms,
                total_ms=total_ms,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                user_id=user_id,
                guest_session_id=guest_session_id,
                arena=payload.arena,
                trace_writer=request.app.state.trace_writer,
            )
```

В блоке `except Exception as exc:` обернуть метрики и запись отказа:

```python
        if not bench:
            metrics_mod.record_error(classified.code)
            metrics_mod.record_chat_request(
                provider=runtime.generation.provider,
                stream=False,
                status="error",
                seconds=time.perf_counter() - started,
            )
            await _persist_failure(
                database,
                completion_id,
                session_id,
                runtime,
                payload,
                str(exc),
                stream=False,
                classified=classified,
                stage=stage,
                user_id=user_id,
                guest_session_id=guest_session_id,
            )
        return _classified_error_response(classified, retry_id=completion_id, stage=stage)
```

В `finally:` заменить `metrics_mod.dec_chat_in_flight()` на:

```python
        if not bench:
            metrics_mod.dec_chat_in_flight()
```

Заменить финальный блок метрик и `return` на:

```python
    if not bench:
        metrics_mod.record_chat_request(
            provider=runtime.generation.provider,
            stream=False,
            status="ok",
            seconds=time.perf_counter() - started,
        )
    response: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": runtime.generation.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "sources": outcome.sources if outcome is not None else [],
        "pipeline": {
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
            "stages": {
                **(outcome.stage_durations_ms if outcome is not None else {}),
                StageName.GENERATION: generation_ms,
            },
        },
    }
    if bench and outcome is not None:
        trace = getattr(outcome, "trace", None)
        if trace is not None:
            response["pipeline_trace"] = trace
    return response
```

`Any` уже импортирован в `main.py:10` — дополнительного импорта не требуется.

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `uv run pytest tests/test_bench_endpoint.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Прогнать смежные наборы на отсутствие регрессий**

Run: `uv run pytest tests/test_admission.py tests/test_chat_or_errors.py tests/test_api_errors.py -v`
Expected: PASS, всё зелёное

- [ ] **Step 6: Коммит**

```bash
git add src/meno_rag/api/main.py tests/test_bench_endpoint.py
git commit -m "feat(bench): return the pipeline trace inline and skip prod writes"
```

---

## Task 6: Streaming — трейс финальным SSE-событием

**Files:**
- Modify: `src/meno_rag/api/main.py:883-1068` (`_stream_response`)
- Test: `tests/test_bench_endpoint.py` (дополняется)

- [ ] **Step 1: Написать падающий тест**

Дописать в `tests/test_bench_endpoint.py`:

```python
def _post_stream(client, *, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "menon-1",
            "messages": [{"role": "user", "content": "Когда сессия?"}],
            "stream": True,
        },
        headers=headers,
    )


def test_bench_stream_emits_a_pipeline_trace_event(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    async def _tokens(*args, **kwargs):
        for token in ("Сессия ", "в январе."):
            yield token

    bench_client.app.state.pipeline.stream_text = _tokens

    body = _post_stream(bench_client, token=BENCH_TOKEN).text

    assert "event: pipeline_trace" in body
    assert '"rerank"' in body
    assert body.rstrip().endswith("data: [DONE]")


def test_normal_stream_has_no_pipeline_trace_event(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    async def _tokens(*args, **kwargs):
        yield "Сессия в январе."

    bench_client.app.state.pipeline.stream_text = _tokens

    body = _post_stream(bench_client, token=None).text

    assert "event: pipeline_trace" not in body
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_bench_endpoint.py -k stream -v`
Expected: FAIL — `TypeError: _stream_response() got an unexpected keyword argument 'bench'`

- [ ] **Step 3: Реализовать**

В сигнатуру `_stream_response` добавить после `capture_trace: bool = False,`:

```python
    bench: bool = False,
```

Заменить `metrics_mod.inc_chat_in_flight()` на:

```python
    if not bench:
        metrics_mod.inc_chat_in_flight()
```

Вставить эмиссию трейса между `StageSummary` и `[DONE]`:

```python
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        yield StageSummary(total_ms=total_ms, stages=stage_durations).to_sse()
        if bench:
            # After the answer, not before it: the blob is large and would
            # otherwise delay the first token for no benefit.
            trace = getattr(outcome, "trace", None)
            if trace is not None:
                yield sse_event("pipeline_trace", {"pipeline_trace": trace})
        yield sse_data("[DONE]")
```

Обернуть успешную запись — заменить `await _persist_success(` … `)` и следующий за ним `metrics_mod.record_chat_request(...)` на:

```python
        answer = "".join(answer_parts)
        if not bench:
            await _persist_success(
                database=database,
                run_id=completion_id,
                session_id=session_id,
                model=runtime.generation.model_id,
                generation_model=runtime.generation.model_id,
                core_model=runtime.core.model_id,
                endpoint=runtime.generation.base_url,
                question=outcome.question,
                answer=answer,
                outcome=outcome,
                generation_ms=generation_ms,
                total_ms=total_ms,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                user_id=user_id,
                guest_session_id=guest_session_id,
                arena=payload.arena,
                trace_writer=request.app.state.trace_writer,
            )
            metrics_mod.record_chat_request(
                provider=runtime.generation.provider,
                stream=True,
                status="ok",
                seconds=time.perf_counter() - started,
            )
```

В блоке `except Exception as exc:` обернуть хвост (метрики и запись отказа) — SSE-события об ошибке остаются безусловными, их видит клиент:

```python
        if not bench:
            metrics_mod.record_error(classified.code)
            metrics_mod.record_chat_request(
                provider=runtime.generation.provider,
                stream=True,
                status="error",
                seconds=time.perf_counter() - started,
            )
            await _persist_failure(
                database,
                completion_id,
                session_id,
                runtime,
                payload,
                str(exc),
                stream=True,
                classified=classified,
                stage=stage,
                user_id=user_id,
                guest_session_id=guest_session_id,
            )
```

В `finally:` заменить `metrics_mod.dec_chat_in_flight()` на:

```python
        if not bench:
            metrics_mod.dec_chat_in_flight()
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `uv run pytest tests/test_bench_endpoint.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Коммит**

```bash
git add src/meno_rag/api/main.py tests/test_bench_endpoint.py
git commit -m "feat(bench): emit the pipeline trace as a final SSE event when streaming"
```

---

## Task 7: Документация переменных окружения

**Files:**
- Modify: `example.env` (после блока `PIPELINE_TRACE_QUEUE_MAX`)

- [ ] **Step 1: Дописать блок**

В `example.env` после строки `PIPELINE_TRACE_QUEUE_MAX=1000`:

```bash
# --- Benchmark endpoint (developers only) ---
# A hidden, OpenAI-compatible path for benchmark runs. A request carrying this
# token in `Authorization: Bearer` gets the FULL RAG pipeline trace inline
# (rewrite queries, dense/lexical ranks, fusion, rerank, prompt, chunk texts)
# and writes NOTHING to production tables — no conversations, no pipeline_runs,
# no metrics.
#
# EMPTY = OFF. Turn it on only where you also gate the path at the edge: nginx
# 404s /bench/v1/ without this same token (see Meno-Deploy). A leaked token
# gives a stranger your GPU and your OpenRouter billing.
# Generate with: openssl rand -hex 32
BENCH_API_TOKEN=
# Separate concurrency budget for benchmark traffic, so a benchmark run can
# never exhaust the pool serving real users. Independent of MAX_CONCURRENT_CHATS.
BENCH_MAX_CONCURRENT=4
```

- [ ] **Step 2: Проверить, что дефолты в коде и в примере совпадают**

Run: `uv run python -c "from meno_rag.config import Settings; s=Settings(); print(repr(s.bench_api_token), s.bench_max_concurrent)"`
Expected: `'' 4`

- [ ] **Step 3: Коммит**

```bash
git add example.env
git commit -m "docs(bench): document BENCH_API_TOKEN and BENCH_MAX_CONCURRENT"
```

---

## Task 8: Видимость bench-трафика в метриках (дополнение к спеке)

**Обоснование и статус:** спека (§1.3) решила «метрики не пишутся», чтобы прогон не искажал продовые ряды. Побочный эффект: во время прогона дашборд показывает загруженный GPU при нулевом трафике — оператор видит аномалию без объяснения. Отдельный счётчик закрывает пробел, не смешиваясь с продовыми рядами. **Это добавление к утверждённой спеке — согласовать перед выполнением или пропустить задачу целиком.**

**Files:**
- Modify: `src/meno_rag/api/metrics.py`
- Modify: `src/meno_rag/api/main.py`
- Test: `tests/test_bench_endpoint.py` (дополняется)

- [ ] **Step 1: Написать падающий тест**

```python
def test_bench_requests_are_counted_in_their_own_metric(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod
    from meno_rag.api import metrics as metrics_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    def _value():
        return metrics_mod.REGISTRY.get_sample_value("meno_bench_requests_total", {"status": "ok"}) or 0.0

    def _prod_value():
        return metrics_mod.REGISTRY.get_sample_value(
            "meno_chat_requests_total", {"provider": "vllm", "stream": "false", "status": "ok"}
        ) or 0.0

    # Реестр глобальный и живёт всю сессию pytest, поэтому сравниваем приросты,
    # а не абсолютные значения — иначе тест зависит от порядка выполнения.
    bench_before, prod_before = _value(), _prod_value()
    assert _post(bench_client, token=BENCH_TOKEN).status_code == 200
    assert _value() == bench_before + 1
    assert _prod_value() == prod_before  # продовый ряд не сдвинулся
```

- [ ] **Step 2: Запустить тест, убедиться что падает**

Run: `uv run pytest tests/test_bench_endpoint.py -k bench_requests_are_counted -v`
Expected: FAIL — значение `meno_bench_requests_total` не меняется (метрики нет)

- [ ] **Step 3: Добавить счётчик**

В `src/meno_rag/api/metrics.py` после `_PIPELINE_TRACE`:

```python
_BENCH_REQUESTS = Counter(
    "meno_bench_requests",
    "Benchmark-endpoint requests by outcome. Deliberately a separate series from "
    "meno_chat_requests: benchmark load must not distort production numbers, but "
    "it does consume the same GPU, so an operator needs to see that it is running.",
    labelnames=("status",),
    registry=REGISTRY,
)
```

И функцию рядом с `record_chat_request`:

```python
def record_bench_request(*, status: str) -> None:
    _BENCH_REQUESTS.labels(status=status).inc()
```

- [ ] **Step 4: Вызвать из обработчиков**

В `_non_stream_response`, в успешном хвосте — заменить

```python
    if not bench:
        metrics_mod.record_chat_request(
            provider=runtime.generation.provider,
            stream=False,
            status="ok",
            seconds=time.perf_counter() - started,
        )
```

на

```python
    if bench:
        metrics_mod.record_bench_request(status="ok")
    else:
        metrics_mod.record_chat_request(
            provider=runtime.generation.provider,
            stream=False,
            status="ok",
            seconds=time.perf_counter() - started,
        )
```

В блоке `except` того же метода добавить перед `if not bench:`:

```python
        if bench:
            metrics_mod.record_bench_request(status="error")
```

В `_stream_response`, в успешном хвосте — заменить

```python
        if not bench:
            await _persist_success(
```

на

```python
        if bench:
            metrics_mod.record_bench_request(status="ok")
        else:
            await _persist_success(
```

(остальное тело вызова `_persist_success` и следующий за ним `record_chat_request` остаются как есть, внутри ветки `else`).

В блоке `except Exception as exc:` метода `_stream_response` добавить непосредственно перед `if not bench:`:

```python
        if bench:
            metrics_mod.record_bench_request(status="error")
```

- [ ] **Step 5: Запустить тесты, убедиться что проходят**

Run: `uv run pytest tests/test_bench_endpoint.py -v`
Expected: PASS, 11 passed

- [ ] **Step 6: Коммит**

```bash
git add src/meno_rag/api/metrics.py src/meno_rag/api/main.py tests/test_bench_endpoint.py
git commit -m "feat(bench): count benchmark requests in a dedicated metric series"
```

---

## Финальная проверка

- [ ] **Линт и типы**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src`
Expected: без ошибок

- [ ] **Локальный прогон релевантного подмножества**

Run: `uv run pytest tests/test_bench_token.py tests/test_bench_endpoint.py tests/test_admission.py tests/test_chat_or_errors.py tests/test_api_errors.py tests/test_api_events.py -v`
Expected: всё зелёное

**Локальный прогон на macOS требует `STAND_RESOURCES_DIR=/nonexistent`.** Проверено
2026-09-09: без него любой тест, поднимающий приложение через `TestClient`, падает
с `Fatal Python error: Segmentation fault` при выгрузке faiss-индекса на 938 МБ.
Сегфолт воспроизводится и на чистом `main`, то есть к нашему коду отношения не имеет.
Тесты от реальных ресурсов не зависят — они подменяют `app.state.pipeline` моком,
а `lifespan` при отсутствии индекса просто выставляет `pipeline = None`.

```bash
STAND_RESOURCES_DIR=/nonexistent uv run pytest tests/test_bench_token.py tests/test_bench_endpoint.py tests/test_admission.py tests/test_chat_or_errors.py tests/test_api_errors.py tests/test_api_events.py -v
```

Полный набор локально всё равно не гонять — в CI ресурсов стенда нет, и он прогоняет всё.

- [ ] **Ручная проверка через SSH-туннель (на хосте, после выката)**

На хосте прописать `BENCH_API_TOKEN` в `.env`, затем `./scripts/run_backend.sh restart`. С локальной машины:

```bash
ssh -L 9006:127.0.0.1:9006 meno
```

В другом терминале:

```bash
curl -s http://127.0.0.1:9006/v1/chat/completions \
  -H "Authorization: Bearer $BENCH_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"meno-lite","messages":[{"role":"user","content":"Когда начинается сессия?"}]}' \
  | python3 -m json.tool | head -40
```

Expected: в ответе присутствует ключ `pipeline_trace` с непустыми `retrieval`, `rerank`, `prompt`, `chunks`.

Тот же запрос без заголовка `Authorization` — ответ без `pipeline_trace`.

- [ ] **PR**

```bash
git push -u origin feat/bench-endpoint-monitoring
gh pr create --title "feat(bench): token-gated benchmark mode with inline RAG trace" --body "Реализует План A спеки docs/superpowers/specs/2026-09-09-bench-endpoint-and-monitoring-design.md"
```

На этом шаге публичного поведения не меняется: эндпоинт доступен только через SSH-туннель, пока План B не добавит `location /bench/v1/` в nginx.
