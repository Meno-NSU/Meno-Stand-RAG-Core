# Шлюз и стенды — План 1: контракт и шлюз в текущем процессе

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Контракт `meno-backend/1` с проверкой `check-backend` и шлюз в текущем процессе: стенд другого автора подключается одной записью в файле и работает в меню, истории, рейтинге и метриках, а его ошибки не задевают остальных.

**Architecture:** Лёгкий пакет `packages/meno-contract` (модели, разбор SSE, проверка, пример стенда) используют и шлюз, и авторы, поэтому проверка и реальное поведение не расходятся. Код шлюза в `src/meno_rag/api/stands/` заменяет `api/external.py`: реестр читает манифесты, реле разбирает поток и собирает его заново, участник (`стенд@версия` + LLM) задаёт ключи истории, рейтинга и метрик. v1 и v2 остаются в процессе и описываются такой же записью участника.

**Tech Stack:** Python 3.12, FastAPI, pydantic 2, httpx, uv workspace, alembic, prometheus-client, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-gateway-and-stands-design.md`, разделы 7 и 8, фазы 1–2 раздела 13.

**Не входит в этот план:** правки Meno-Web (фаза 3, сначала макеты), раннер бенчмарка и прямой путь в стенд (фаза 4), v1 и v2 как стенды (фазы 5–6, ждут решения по `feat/adaptive-search`), общие сервисы (фаза 7). Для них будут отдельные планы.

## Global Constraints

- Работа в `~/Meno-RAG-Core-gateway`, ветка `feat/gateway-stands` от `design/gateway-stands`. Каталог `~/Meno-RAG-Core` с его незакоммиченными правками не трогать.
- Тесты запускать только по отдельным файлам: `cd ~/Meno-RAG-Core-gateway && ~/Meno-RAG-Core/.venv/bin/python -m pytest <файл> -q`. Весь набор на этой машине запускать нельзя (3,9 ГБ RAM, уже было падение), это касается и субагентов.
- Перед каждым коммитом: `~/Meno-RAG-Core/.venv/bin/ruff check <файлы>` и `~/Meno-RAG-Core/.venv/bin/ruff format <файлы>`, длина строки 120.
- Функции с подчёркиванием в примерах тестов (`_registry`, `_vote`, `_ratings`, `_count` и подобные) — локальные помощники файла теста, их пишет исполнитель. Тело `...` с комментарием означает: написать тест, проверяющий сказанное в комментарии.
- Git: автор и коммиттер `sckwoky <sckwokyboom@gmail.com>`, без трейлера `Co-Authored-By`. Пуш только по просьбе владельца. Сервер и прод в этом плане не трогаем.
- Не менять: Meno-Web, `src/meno_rag/v2/`, `src/meno_rag/stand/`.
- При пустом `STANDS_FILE` меню, ответы и потоки как сейчас. Меняются только метки метрик и метка базы в сохранённых записях.
- Миграции только добавляющие, новые колонки допускают NULL.
- `lock` только версией из CI: `uvx uv@0.9.10 lock`.
- Значения контракта: протокол `meno-backend/1`; манифест `GET /meno/v1/manifest`; `version` `[A-Za-z0-9._+-]{1,64}`; имя этапа `[a-z0-9_]{1,64}`; id стенда `[a-z0-9][a-z0-9.-]{0,31}`; заголовки `X-Meno-Request-Id`, `X-Meno-Trace: 1`.
- Стандартные этапы (подпись в манифесте не обязательна): `abbreviation_expansion`, `anaphora_resolution`, `query_rewrite`, `retrieval`, `fusion`, `rerank`, `context_assembly`, `generation`.
- Лимиты шлюза: подключение 5 с; пауза между чанками 300 с; весь ответ 900 с (`STAND_TOTAL_TIMEOUT_SECONDS`); ответ 40 000 символов; блок SSE 1 МБ, для `contexts` и `pipeline_trace` 16 МБ; `max_concurrent` 32 на стенд; `detail.lines` 50 строк по 500 символов; 20 источников; 32 версии на стенд в метках. Повтор недоступного стенда через 10 с.
- Id участника в API: `<llm>@<stand>`, разбор через `rpartition("@")`. Метка базы: `<stand>@<version>`. Версия эндпоинта без манифеста: `plain`. Неизвестная версия: `unknown`.

## Review Focus

1. Стенд шлёт SSE как настоящие серверы: комментарии `: ping`, CRLF, блок разорван между чтениями из сети, кириллица разорвана посреди символа. Ответ приходит целым, нарушений не считается. Тесты: Task 2 `test_iter_blocks_survives_real_servers`, Task 5 `test_relay_stream_handles_a_split_multibyte_character`.
2. Пользователь нажал «Стоп» или закрыл вкладку посреди ответа. Соединение со стендом закрывается, общий слот и слот стенда свободны. Тест: Task 6 `test_client_disconnect_frees_the_slots_and_closes_the_stand`.
3. Выбранная модель исчезла: LLM убрали из манифеста или стенд удалён из файла. Понятный 400 `model_not_found`, и пайплайн v1 молча не запускается. Тест: Task 6 `test_a_vanished_model_is_a_clear_error`.
4. Файл реестра правят на ходу: стенд удалён посреди его ответа или файл сломан. Ответ доходит и сохраняется, слот свободен, при сломанном файле работает прошлый список. Тесты: Task 4 `test_a_broken_file_keeps_the_previous_list`, Task 6 `test_a_stand_removed_mid_answer_finishes_it`.
5. Нынешний фронтенд не знает про `arena` и выбирает для арены стенд с `arena: false` или без манифеста. Такой стенд не попадает в голосование: шлюз отвечает 403 до начала потока, а фронтенд сам подставляет другого участника. Тест: Task 6 `test_a_stand_outside_the_arena_refuses_arena_requests`.

---

## Подготовка

- [ ] Создать рабочую ветку: `cd ~/Meno-RAG-Core-gateway && git switch -c feat/gateway-stands`.
- [ ] Убедиться, что идентичность верна: `git config user.name` и `git config user.email` печатают `sckwoky` и `sckwokyboom@gmail.com`.

## Структура файлов

```
packages/meno-contract/pyproject.toml
packages/meno-contract/src/meno_contract/{__init__,models,sse,check,example_stand}.py
docs/contract/meno-backend-v1.md          описание для авторов стендов
stands.example.json
src/meno_rag/api/stands/__init__.py
src/meno_rag/api/stands/config.py         запись стенда и чтение файла
src/meno_rag/api/stands/registry.py       манифесты, записи /v1/models, разрешение id
src/meno_rag/api/stands/relay.py          разбор потока, лимиты, запрос к стенду
src/meno_rag/api/stands/participant.py    стенд, версия и LLM для истории, рейтинга, метрик
alembic/versions/0018_pipeline_run_stand_build.py
tests/contract/{__init__,test_contract_models,test_contract_sse,test_contract_check}.py
tests/stands/{__init__,_helpers,test_stand_config,test_stand_registry,test_stand_relay,test_stand_chat,test_stand_trace,test_stand_participant}.py
```

Удаляются: `src/meno_rag/api/external.py`, `tests/v2/test_external_backends.py`.

---

### Task 1: Пакет `meno-contract` и модели контракта

**Files:**
- Create: `packages/meno-contract/pyproject.toml`, `packages/meno-contract/src/meno_contract/__init__.py`, `packages/meno-contract/src/meno_contract/models.py`, `tests/contract/__init__.py`, `tests/contract/test_contract_models.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces, в `meno_contract.models`, все модели с `extra="ignore"`:
  - константы `PROTOCOL`, `STANDARD_STAGES: frozenset[str]`, `MAX_DETAIL_LINES = 50`, `MAX_DETAIL_LINE_CHARS = 500`, `MAX_SOURCES = 20`;
  - `StageLabel(ru: str, en: str)`;
  - `LlmStatus(state: Literal["available","rate_limited","unreachable"], until: str | None = None)`;
  - `Llm(id: str, name: str, context_length: int | None = None, requires_auth: bool = False, status: LlmStatus | None = None)`;
  - `Manifest(protocol, name, description: str | None, version, build: str | None, kb: str | None, llms: list[Llm], default_llm: str, stages: dict[str, StageLabel] = {})`;
  - `StageDetail(lines: list[str] = [])`;
  - `StageEvent(stage: str, status: Literal["started","completed","failed","skipped"], duration_ms: float | None = None, detail: StageDetail | None = None)`;
  - `Context(id: str, url: str = "", title: str = "", text: str, rank: int | None = None)`;
  - `ErrorEvent(code: str = "stand_error", message: str = "", retryable: bool = False, retry_after_sec: int | None = None)`;
  - `clean_sources(raw: Any) -> tuple[list[dict[str, str]], int]`: строки для показа и число отброшенных.

- [ ] **Step 1: Написать тесты**

```python
from meno_contract.models import PROTOCOL, Context, Manifest, StageEvent, clean_sources

GOOD = {"protocol": PROTOCOL, "name": "RAGU", "version": "2026-11-09-kb36",
        "llms": [{"id": "gemma", "name": "Gemma 4"}], "default_llm": "gemma",
        "stages": {"graph_walk": {"ru": "Обход графа", "en": "Graph walk"}}}

def test_manifest_accepts_the_minimum_and_ignores_unknown_fields():
    m = Manifest.model_validate({**GOOD, "future": 1})
    assert m.version == "2026-11-09-kb36" and m.llms[0].requires_auth is False

@pytest.mark.parametrize("patch", [
    {"protocol": "meno-backend/2"}, {"version": "with space"}, {"version": "x" * 65},
    {"llms": []}, {"default_llm": "absent"},
    {"llms": [{"id": "a", "name": "A"}, {"id": "a", "name": "B"}], "default_llm": "a"},
    {"stages": {"Bad-Name": {"ru": "x", "en": "x"}}},
])
def test_manifest_rejects(patch):
    with pytest.raises(ValidationError):
        Manifest.model_validate({**GOOD, **patch})

def test_stage_detail_is_trimmed_to_the_limits():
    e = StageEvent.model_validate({"stage": "retrieval", "status": "completed",
                                   "detail": {"lines": ["я" * 600] * 60}})
    assert len(e.detail.lines) == 50 and len(e.detail.lines[0]) == 500

def test_stage_detail_without_lines_is_empty():
    assert StageEvent.model_validate({"stage": "rerank", "status": "started", "detail": {"round": 2}}).detail.lines == []

@pytest.mark.parametrize("patch", [{"status": "done"}, {"stage": "Bad Name"}, {"duration_ms": -1}])
def test_stage_event_rejects(patch):
    with pytest.raises(ValidationError):
        StageEvent.model_validate({"stage": "rerank", "status": "started", **patch})

def test_clean_sources_keeps_only_what_can_be_shown():
    rows, dropped = clean_sources([
        {"document_title": "График", "source_url": "https://nsu.ru/g", "date": None, "score": 0.9},
        {"document_title": 7, "source_url": "", "date": "2026-01-31"},
        {"document_title": "x", "source_url": "javascript:alert(1)"},
        "мусор",
    ])
    assert rows == [{"document_title": "График", "source_url": "https://nsu.ru/g"},
                    {"document_title": "7", "source_url": "", "date": "2026-01-31"}]
    assert dropped == 2

def test_clean_sources_caps_the_count_and_survives_a_non_list():
    rows, dropped = clean_sources([{"document_title": "t", "source_url": ""}] * 25)
    assert len(rows) == 20 and dropped == 5
    assert clean_sources(None) == ([], 0)

def test_context_id_becomes_a_string():
    assert Context.model_validate({"id": 12, "text": "т"}).id == "12"
```

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/contract/test_contract_models.py -q`. Ожидание: ошибка импорта.

- [ ] **Step 3: Создать пакет и подключить workspace**

`packages/meno-contract/pyproject.toml`: имя `meno-contract`, версия `0.1.0`, `requires-python = ">=3.12"`, зависимости `pydantic>=2.10.0`, `httpx>=0.28.0`, необязательная группа `example = ["fastapi>=0.115.0", "uvicorn>=0.30.0"]`, команда `check-backend = "meno_contract.check:main"`, сборка как в корне: `uv_build>=0.8.0,<0.10.0`, `[tool.uv.build-backend] module-name = "meno_contract"`.

В корневом `pyproject.toml`: зависимость `"meno-contract"`, секции `[tool.uv.workspace] members = ["packages/*"]` и `[tool.uv.sources] meno-contract = { workspace = true }`, `pythonpath = ["src", ".", "packages/meno-contract/src"]`, в mypy `files = ["src/meno_rag", "packages/meno-contract/src"]`.

- [ ] **Step 4: Реализовать `models.py`** по блоку Interfaces. Значения из Global Constraints. `Manifest` проверяет: протокол равен `PROTOCOL`, `llms` не пуст и id уникальны, `default_llm` есть в `llms`, имена этапов по шаблону. `StageDetail` обрезает строки в `field_validator(mode="after")`, а не отвергает их. В `clean_sources` строка для показа содержит только `document_title`, `source_url` и непустой `date`, значения приведены к строкам; адрес только `http://`, `https://` или пустой.

- [ ] **Step 5: Запустить тесты.** Ожидание: PASS.

- [ ] **Step 6: Обновить lock и проверить, что менялось только нужное.** `uvx uv@0.9.10 lock`, затем `uvx uv@0.9.10 lock --check` и `git diff --stat uv.lock`. Ожидание: `--check` успешен, в диффе только записи `meno-contract` и зависимость корня (десятки строк, не тысячи). Если обновились посторонние пакеты, откатить `uv.lock` и разобраться.

- [ ] **Step 7: Коммит:** `feat(contract): the meno-backend/1 models in a workspace package`.

---

### Task 2: Разбор SSE и пример стенда

**Files:**
- Create: `packages/meno-contract/src/meno_contract/sse.py`, `packages/meno-contract/src/meno_contract/example_stand.py`, `tests/contract/test_contract_sse.py`

**Interfaces:**
- Consumes: `meno_contract.models` (Task 1).
- Produces:
  - `SseBlock(event: str, data: str, raw_size: int)`, frozen; событие по умолчанию `"message"`;
  - `parse_block(text: str) -> SseBlock | None`: `None` для пустых блоков и блоков только из комментариев;
  - `class SseBlockTooLarge(Exception)`;
  - `async def iter_blocks(pieces: AsyncIterator[str], *, max_bytes: int) -> AsyncIterator[SseBlock]`;
  - `example_stand.create_app(version: str = "example-1") -> FastAPI`, `example_stand.main() -> None`. Стенд с `id` LLM `echo`.

- [ ] **Step 1: Написать тесты**

```python
async def _pieces(*parts):
    for p in parts:
        yield p

async def _collect(*parts, max_bytes=1_000_000):
    return [b async for b in iter_blocks(_pieces(*parts), max_bytes=max_bytes)]

def test_parse_block_reads_event_and_joins_data_lines():
    b = parse_block('event: stage\ndata: {"a":\ndata: 1}')
    assert b.event == "stage" and b.data == '{"a":\n1}'

def test_parse_block_accepts_data_without_a_space_and_skips_comments():
    assert parse_block("data:x").data == "x"
    assert parse_block(": ping") is None and parse_block("") is None

async def test_iter_blocks_survives_real_servers():
    blocks = await _collect(": ping\r\n\r\nevent: st", 'age\r\ndata: {"x"', ': 1}\r\n\r\ndata: [DONE]')
    assert [(b.event, b.data) for b in blocks] == [("stage", '{"x": 1}'), ("message", "[DONE]")]

async def test_iter_blocks_refuses_an_oversized_block():
    with pytest.raises(SseBlockTooLarge):
        await _collect("data: " + "я" * 2000, max_bytes=1000)

async def test_example_stand_answers_by_the_contract():
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://stand")
    m = Manifest.model_validate((await client.get("/meno/v1/manifest")).json())
    assert m.default_llm == "echo"
    body = {"model": "echo", "stream": True, "messages": [{"role": "user", "content": "Привет"}]}
    plain = (await client.post("/v1/chat/completions", json=body)).text
    assert "event: sources" in plain and "event: contexts" not in plain and "data: [DONE]" in plain
    traced = (await client.post("/v1/chat/completions", json=body, headers={"X-Meno-Trace": "1"})).text
    assert "event: contexts" in traced
    assert (await client.post("/v1/chat/completions", json={**body, "model": "absent"})).status_code == 400
```

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/contract/test_contract_sse.py -q`.

- [ ] **Step 3: Реализовать `sse.py`.** Буфер копит текст, CRLF заменяется на LF, блоки делятся по пустой строке, остаток без финальной пустой строки разбирается в конце. `raw_size` — длина блока в байтах UTF-8. Превышение `max_bytes` проверяется и для ещё не завершённого блока в буфере.

- [ ] **Step 4: Реализовать `example_stand.py`.** Один файл на FastAPI, без LLM, это образец для авторов, поэтому код простой и с комментариями. Отвечает эхом последней реплики. Этапы: стандартный `retrieval` и свой `echo` с подписью в манифесте. Один источник. `contexts` только при `X-Meno-Trace`. Неизвестная модель: 400 и `{"error": {"message": ..., "code": "model_not_found"}}`. Есть также `/v1/models`. `main()` принимает `--port`, запускает uvicorn.

- [ ] **Step 5: Запустить тесты.** Ожидание: PASS.

- [ ] **Step 6: Коммит:** `feat(contract): SSE parsing and a one-file example stand`.

---

### Task 3: `check-backend` и описание контракта

**Files:**
- Create: `packages/meno-contract/src/meno_contract/check.py`, `docs/contract/meno-backend-v1.md`, `tests/contract/test_contract_check.py`

**Interfaces:**
- Consumes: `Manifest`, `StageEvent`, `Context`, `clean_sources`, `STANDARD_STAGES` (Task 1), `iter_blocks` (Task 2), `example_stand.create_app` (Task 2, только в тестах).
- Produces:
  - `CheckItem(name: str, ok: bool, detail: str = "", warning: bool = False)`;
  - `Report(items: list[CheckItem], plain: bool = False)` со свойством `ok -> bool`: истина, если нет не пройденных пунктов без `warning`;
  - `async def run_checks(client: httpx.AsyncClient, base_url: str, *, api_key: str = "", first_token_limit_s: float = 120.0) -> Report`;
  - `format_report(report: Report) -> str`;
  - `main(argv: list[str] | None = None) -> int`: `check-backend <url> [--key KEY]`, 0 при успехе, иначе 1.

- [ ] **Step 1: Написать тесты.** Плохие стенды строятся из хорошего потока через `httpx.MockTransport`.

```python
async def test_the_example_stand_passes():
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()))
    report = await run_checks(client, "http://stand")
    assert report.ok, format_report(report)

@pytest.mark.parametrize("fault,expected", [
    ("no_done", "[DONE]"),
    ("no_stage", "этап"),
    ("unpaired_stage", "парн"),
    ("unlabelled_stage", "подпис"),
    ("no_sources", "sources"),
    ("no_content", "текст"),
    ("no_contexts_with_trace", "contexts"),
    ("contexts_without_trace", "contexts"),
    ("unknown_model_ok", "неизвестн"),
])
async def test_a_bad_stand_fails_with_a_clear_reason(fault, expected):
    report = await run_checks(_client(fault), "http://stand")
    assert not report.ok
    assert expected in format_report(report)

async def test_a_wrong_protocol_is_reported():
    report = await run_checks(_client(manifest={"protocol": "meno-backend/2"}), "http://stand")
    assert not report.ok and "протокол" in format_report(report).lower()

async def test_a_plain_endpoint_may_enter_the_menu_but_not_the_arena():
    report = await run_checks(_client(manifest=None), "http://stand")
    assert report.plain and not report.ok and "арен" in format_report(report)

async def test_think_in_the_answer_is_a_warning_only():
    report = await run_checks(_client("think_in_content"), "http://stand")
    assert report.ok and "<think>" in format_report(report)

def test_main_returns_nonzero_when_the_stand_is_down():
    assert main(["http://127.0.0.1:9"]) == 1
```

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/contract/test_contract_check.py -q`.

- [ ] **Step 3: Реализовать `check.py`.** Порядок: манифест; обычный вопрос к `default_llm` с `stream: true`; тот же вопрос с `X-Meno-Trace: 1`; неизвестная модель. Проверки потока из раздела 7.4 спецификации: хотя бы один этап, парность `started`, подпись у нестандартного этапа, непустой текст, хотя бы одно `sources`, последний блок `[DONE]`, `contexts` только при заголовке и до `[DONE]`, первый токен не позже `first_token_limit_s`. Ответ на неизвестную модель должен быть 4xx. Ошибка соединения превращается в не пройденный пункт, а не в трассировку. Сообщения по-русски и называют, что исправить.

- [ ] **Step 4: Запустить тесты.** Ожидание: PASS.

- [ ] **Step 5: Написать `docs/contract/meno-backend-v1.md`** для автора, который не видел наш код. Содержание берётся из раздела 7 спецификации: что реализовать (манифест, запрос, таблица событий, ошибки), правила 7.7, как проверить себя, что передать нам (URL и ключ), как подключается эндпоинт без манифеста. Команда проверки:

```
uvx --from "git+https://github.com/Meno-NSU/Meno-Stand-RAG-Core@feat/gateway-stands#subdirectory=packages/meno-contract" check-backend http://host:port
```

- [ ] **Step 6: Проверить на настоящих сокетах.**

```bash
cd ~/Meno-RAG-Core-gateway/packages/meno-contract/src
timeout 20 ~/Meno-RAG-Core/.venv/bin/python -m meno_contract.example_stand --port 9299 &
sleep 3; ~/Meno-RAG-Core/.venv/bin/python -m meno_contract.check http://127.0.0.1:9299; echo "exit=$?"
```

Ожидание: все пункты пройдены, `exit=0`.

- [ ] **Step 7: Коммит:** `feat(contract): check-backend and the contract description for stand authors`.

---

### Task 4: Реестр стендов из файла

**Files:**
- Create: `src/meno_rag/api/stands/__init__.py`, `src/meno_rag/api/stands/config.py`, `src/meno_rag/api/stands/registry.py`, `tests/stands/__init__.py`, `tests/stands/_helpers.py`, `tests/stands/test_stand_config.py`, `tests/stands/test_stand_registry.py`
- Modify: `src/meno_rag/api/external.py` (удалить `ExternalBackend`, `parse_external_backends`, `external_model_id`, `ExternalTarget`, `ExternalRegistry`), `src/meno_rag/api/main.py:253-259,607-609,635-637,1354-1367`, `src/meno_rag/api/metrics.py`, `src/meno_rag/config.py:222-225`, `tests/v2/test_external_backends.py`

**Interfaces:**
- Consumes: `Manifest` (Task 1), `AdmissionController` (`api/admission.py`).
- Produces:
  - `config.StandEntry(id: str, url: str, api_key: str = "", arena: bool = False, max_concurrent: int = 32, name: str = "")`, frozen, с `headers() -> dict[str, str]`;
  - `config.parse_stands(raw: str, *, reserved_ids: frozenset[str] = frozenset()) -> list[StandEntry]`, при ошибке `ValueError` с именем записи;
  - `config.read_stands_file(path: str, *, reserved_ids=...) -> list[StandEntry]`: пустой путь даёт `[]`, отсутствующий файл `ValueError`;
  - `registry.PLAIN_VERSION = "plain"`;
  - `registry.StandTarget(backend: StandEntry, model: str, public_id: str, version: str, build: str | None, manifest: Manifest | None, limiter: AdmissionController, requires_auth: bool)`, frozen, со свойствами `kb_id -> f"{backend.id}@{version}"` и `arena -> bool`: `backend.arena and manifest is not None`;
  - `registry.StandRegistry(path: str, *, http_client: httpx.AsyncClient, timeout: float = 5.0, cache_ttl: float = 300.0, reserved_ids: frozenset[str] = frozenset())`, при ошибке файла в конструкторе `ValueError`. Методы: `async refresh() -> None`, `async entries() -> list[dict]`, `resolve(public_id: str | None) -> StandTarget | None`, `knows_stand(stand_id: str) -> bool`, `live_kbs() -> set[str]`, `problems() -> dict[str, str]`, атрибут `http`;
  - `metrics.set_stand_up(stand: str, up: bool)`, `metrics.record_stand_config_error()`, `metrics.record_stand_violation(stand: str, kind: str)`; серии `meno_stand_up{stand}`, `meno_stand_config_errors`, `meno_stand_contract_violations{stand,kind}`;
  - `Settings.stands_file` (`STANDS_FILE`, по умолчанию `""`), `Settings.stand_total_timeout_seconds` (`STAND_TOTAL_TIMEOUT_SECONDS`, 900.0); `EXTERNAL_BACKENDS` удаляется;
  - `main._stand_registry(app)`, `main._stand_target(app, model)`, `main._stand_runtime(target)`, `app.state.stand_registry`;
  - в `tests/stands/_helpers.py`: `LAB`, `manifest(version="v1", **kw) -> dict`, `block(event=None, data=None) -> str`, `chunk(content=None, *, cid="up-1", model="gemma", finish=None) -> dict`, `sse(*blocks) -> bytes`, `write_stands(path, stands) -> Path`, `stand_handler(seen, *, stream_body=..., manifest=..., down=False)`, `stand_client(make_client, tmp_path, handler, stands=None) -> (client, db, registry)`.

- [ ] **Step 1: Написать тесты**

`test_stand_config.py`:

```python
def test_parse_reads_an_entry():
    lab, = parse_stands(json.dumps([{"id": "lab", "url": "http://lab:9006/", "api_key": "k", "arena": True}]))
    assert lab == StandEntry(id="lab", url="http://lab:9006", api_key="k", arena=True)
    assert lab.headers() == {"Authorization": "Bearer k"} and lab.max_concurrent == 32

def test_a_dot_is_allowed_in_the_id_unless_reserved():
    assert parse_stands('[{"id": "menon-1.1", "url": "http://a"}]')[0].id == "menon-1.1"
    with pytest.raises(ValueError, match="menon-1.1"):
        parse_stands('[{"id": "menon-1.1", "url": "http://a"}]', reserved_ids=frozenset({"menon-1.1"}))

@pytest.mark.parametrize("raw", [
    "[{", '{"id": "lab"}', '[{"id": "lab"}]', '[{"id": "Lab X", "url": "http://a"}]',
    '[{"id": "lab", "url": "ftp://a"}]', '[{"id": "lab", "url": "http://a", "max_concurrent": 0}]',
    '[{"id": "lab", "url": "http://a"}, {"id": "lab", "url": "http://b"}]',
])
def test_parse_rejects_a_bad_file(raw):
    with pytest.raises(ValueError):
        parse_stands(raw)

def test_no_path_means_no_stands_and_a_missing_file_is_an_error(tmp_path):
    assert read_stands_file("") == []
    with pytest.raises(ValueError):
        read_stands_file(str(tmp_path / "absent.json"))
```

`test_stand_registry.py`:

```python
def test_a_manifest_gives_the_menu_entries():
    reg, calls = _registry(manifest(llms=[{"id": "lite", "name": "Лайт"}, {"id": "gemma", "name": "Gemma 4"}],
                                    default_llm="gemma", name="RAGU"), arena=True, api_key="k")
    entries = _run(reg.entries())
    assert [e["id"] for e in entries] == ["gemma@lab", "lite@lab"]          # default LLM first
    e = entries[0]
    assert (e["provider"], e["backend_id"], e["backend_name"]) == ("external", "lab", "RAGU")
    assert (e["display_name"], e["llm_id"], e["version"], e["arena"]) == ("Gemma 4", "gemma", "v1", True)
    assert calls[0].url.path == "/meno/v1/manifest"
    assert calls[0].headers["authorization"] == "Bearer k"

def test_no_manifest_falls_back_to_the_model_list_and_stays_out_of_the_arena():
    reg, _ = _registry(None, models=["gemma"], arena=True, name="Лаба")
    e = _run(reg.entries())[0]
    assert (e["id"], e["version"], e["arena"], e["backend_name"]) == ("gemma@lab", "plain", False, "Лаба")

def test_a_manifest_of_another_protocol_hides_the_stand_and_explains_why():
    reg, _ = _registry(manifest(protocol="meno-backend/2"))
    assert _run(reg.entries()) == [] and "meno-backend/2" in reg.problems()["lab"]

def test_llm_status_and_auth_reach_the_entry():
    reg, _ = _registry(manifest(llms=[{"id": "gemma", "name": "G", "requires_auth": True,
                                       "status": {"state": "rate_limited", "until": "2026-11-09T10:00:00+00:00"}}]))
    e = _run(reg.entries())[0]
    assert e["status"]["state"] == "rate_limited" and e["auth_required"] is True

def test_a_stand_that_stops_answering_keeps_its_last_manifest_marked_unreachable(): ...
    # после ConnectError те же записи с status.state == "unreachable", resolve всё ещё находит цель,
    # meno_stand_up{stand="lab"} == 0

def test_entries_do_not_wait_for_a_refresh_in_progress(): ...   # перенести из test_external_backends.py

def test_a_new_stand_in_the_file_appears_after_refresh(tmp_path): ...
    # дописать запись, os.utime вперёд, await reg.refresh(): появились её записи

def test_a_broken_file_keeps_the_previous_list(tmp_path):
    reg, _ = _registry(manifest(), path=tmp_path / "stands.json")
    _run(reg.refresh())
    before = _metric("meno_stand_config_errors_total")
    (tmp_path / "stands.json").write_text("[{", encoding="utf-8"); _bump_mtime(tmp_path / "stands.json")
    _run(reg.refresh())
    assert [e["id"] for e in _run(reg.entries())] == ["gemma@lab"]
    assert _metric("meno_stand_config_errors_total") == before + 1

def test_resolve_maps_an_id_to_its_target():
    reg, _ = _registry(manifest(build="a1b2"), arena=True)
    t = reg.resolve("gemma@lab")
    assert (t.backend.id, t.model, t.kb_id, t.build, t.arena) == ("lab", "gemma", "lab@v1", "a1b2", True)
    assert reg.resolve("absent@lab") is None and reg.resolve("gemma@other") is None and reg.resolve("gemma") is None

def test_limiter_survives_a_reload_when_the_limit_is_unchanged(tmp_path): ...
    # resolve до и после перечитывания отдаёт один и тот же объект limiter; при новом max_concurrent другой

def test_live_kbs_lists_the_current_versions():
    reg, _ = _registry(manifest())
    assert reg.live_kbs() == {"lab@v1"}
```

Тесты с приложением, в `test_stand_registry.py`: `/v1/models` содержит группу стенда и `requires_auth` у гостя при `auth_required`; `STANDS_FILE` из окружения подхватывается; сломанный файл останавливает старт (`pytest.raises(ValueError)` вокруг `make_client()`); `menon-1.1` при `V2_ENABLED=true` останавливает старт; `POST /v1/models/refresh` подхватывает стенд, даже если при старте файл был пуст; пустой `STANDS_FILE` не добавляет записей с `provider == "external"`.

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/stands/test_stand_config.py tests/stands/test_stand_registry.py -q`.

- [ ] **Step 3: Реализовать `config.py`, `registry.py` и метрики.** Обнаружение: `GET {url}/meno/v1/manifest`. 200 и верный манифест: стенд полный. 404: `GET {url}/v1/models`, стенд без манифеста. Неверный манифест: записей нет, причина в `problems()`, лог, нарушение `bad_manifest`. Ошибка соединения или 5xx: последний манифест сохраняется, статус `unreachable`. `refresh()` сначала перечитывает файл, если изменилось `st_mtime_ns`, при ошибке оставляет прошлый список и пишет лог и метрику. Логику кэша и «запрос не ждёт идущего обновления» перенести из `ExternalRegistry`, убрав выход при пустом списке, иначе файл с первой записью не подхватится.

- [ ] **Step 4: Подключить к приложению.** В `lifespan`: `StandRegistry(settings.stands_file, ..., reserved_ids=frozenset({"menon-1.1"}) if settings.v2_enabled else frozenset())`. В `list_models`: `requires_auth` также для записей с `auth_required`. В `refresh_models`: обновлять реестр всегда. `_external_*` функции в `main.py` продолжают работать: у `StandTarget` те же поля `backend`, `model`, `public_id`, `kb_id`. Переименовать `_external_registry`, `_external_target`, `_external_runtime` в `_stand_*`.

- [ ] **Step 5: Поправить `tests/v2/test_external_backends.py`.** Удалить разделы о конфигурации и списке моделей (перенесены). Оставшимся тестам чата дать реестр через `stand_client`, обработчик отвечает 404 на манифест. Сохранённая метка базы становится `lab@plain`.

- [ ] **Step 6: Запустить тесты.** `pytest tests/stands/test_stand_config.py tests/stands/test_stand_registry.py -q`, затем `pytest tests/v2/test_external_backends.py -q`, затем `pytest tests/test_models_endpoint.py -q`. Ожидание: PASS.

- [ ] **Step 7: Коммит:** `feat(stands): a registry read from a file, with manifests`.

---

### Task 5: Шлюз разбирает поток, а не транслирует его

**Files:**
- Create: `src/meno_rag/api/stands/relay.py`, `tests/stands/test_stand_relay.py`
- Modify: `src/meno_rag/api/external.py` (удалить `Relay`, `relay_stream`), `src/meno_rag/api/errors.py`, `src/meno_rag/api/main.py:1373-1513`, `tests/v2/test_external_backends.py`

**Interfaces:**
- Consumes: `StageEvent`, `Context`, `ErrorEvent`, `clean_sources`, `STANDARD_STAGES` (Task 1), `SseBlock`, `iter_blocks`, `SseBlockTooLarge` (Task 2), `StandTarget`, `metrics.record_stand_violation` (Task 4), `openai_chunk`, `sse_event`, `sse_data`, `StageSummary` (`api/events.py`).
- Produces:
  - `errors.classify_stand_error(code: str, message: str, retryable: bool = False, retry_after_sec: int | None = None) -> ClassifiedError`: известные коды получают существующие тексты, остальные `_USER_MSG_STAND = "Не удалось получить ответ. Повторите запрос или выберите другую модель."`, `http_status=502`. Текст стенда идёт только в `message` (лог и `detail`);
  - `relay.StandStreamError(code, message, retryable=False, retry_after_sec=None)` с методом `classified() -> ClassifiedError`;
  - `relay.stand_request_body(target: StandTarget, payload: ChatCompletionRequest) -> dict`: всегда `"stream": True`;
  - `relay.Relay(completion_id: str, created: int, target: StandTarget, pass_trace: bool = False, record_metrics: bool = True)` с полями `parts`, `sources`, `stage_durations: dict[str, float]`, `stage_details: dict[str, dict]`, `contexts: list[dict]`, `trace: dict | None`, `finished: bool`, `done: bool`; свойство `answer`; методы `handle(block: SseBlock) -> list[str]`, `closing() -> list[str]`, `summary(total_ms: float) -> str`;
  - `async relay.relay_stream(http, target, body, relay, *, request_id: str, trace: bool = False, total_timeout_s: float = 900.0) -> AsyncIterator[str]`;
  - `main._stand_stream_response(...)` вместо `_external_stream_response`, те же параметры.

- [ ] **Step 1: Написать тесты**

```python
def _relay(**kw):
    target = _target(manifest(stages={"graph_walk": {"ru": "Обход графа", "en": "Graph walk"}}))
    return Relay(completion_id="ours", created=1, target=target, **kw)

def _feed(relay, event=None, data=None):
    return relay.handle(parse_block(block(event, data)))

def test_a_chunk_is_rebuilt_with_our_ids_and_no_extra_fields():
    r = _relay()
    out = _feed(r, data={**chunk("Сессия"), "system_fingerprint": "ragu-7"})
    assert r.answer == "Сессия"
    sent = json.loads(out[0].removeprefix("data: "))
    assert sent["id"] == "ours" and sent["model"] == "gemma@lab" and "system_fingerprint" not in sent

def test_a_stage_gets_labels_from_the_manifest_and_generic_details():
    r = _relay()
    out = _feed(r, "stage", {"stage": "graph_walk", "status": "completed", "duration_ms": 7.5,
                             "detail": {"lines": ["3 узла"], "secret": 1}})
    sent = json.loads(out[0].split("data: ", 1)[1])
    assert sent["labels"] == {"ru": "Обход графа", "en": "Graph walk"}
    assert sent["detail"] == {"lines": ["3 узла"]}
    assert r.stage_durations == {"graph_walk": 7.5}

def test_repeated_stages_add_up():
    r = _relay()
    for ms in (10.0, 5.0):
        _feed(r, "stage", {"stage": "rerank", "status": "completed", "duration_ms": ms})
    assert r.stage_durations["rerank"] == 15.0

def test_an_undeclared_stage_passes_and_counts_as_a_violation():
    r, before = _relay(), _violations("lab", "unknown_stage")
    assert _feed(r, "stage", {"stage": "mystery", "status": "started"})
    assert _violations("lab", "unknown_stage") == before + 1

def test_a_bad_stage_is_dropped_and_counted():
    r, before = _relay(), _violations("lab", "bad_event")
    assert _feed(r, "stage", {"stage": "Bad Name", "status": "started"}) == []
    assert _violations("lab", "bad_event") == before + 1

def test_sources_are_cleaned_before_anyone_sees_them():
    r = _relay()
    out = _feed(r, "sources", {"sources": [{"document_title": "Г", "source_url": "https://a", "date": None},
                                           {"document_title": "x", "source_url": "javascript:1"}]})
    assert r.sources == [{"document_title": "Г", "source_url": "https://a"}]
    assert json.loads(out[0].split("data: ", 1)[1]) == {"sources": r.sources}

def test_details_stay_with_the_gateway_unless_it_is_a_benchmark():
    r = _relay()
    ctx = {"contexts": [{"id": 1, "text": "т"}]}
    assert _feed(r, "contexts", ctx) == [] and _feed(r, "pipeline_trace", {"pipeline_trace": {"a": 1}}) == []
    assert r.contexts[0]["id"] == "1" and r.trace == {"a": 1}
    bench = _relay(pass_trace=True)
    assert _feed(bench, "contexts", ctx) and _feed(bench, "pipeline_trace", {"pipeline_trace": {"a": 1}})

def test_a_stand_summary_and_unknown_events_are_ignored_silently():
    r, before = _relay(), _violations("lab", "bad_event")
    assert _feed(r, "summary", {"total_ms": 1}) == [] and _feed(r, "future", {"x": 1}) == []
    assert _violations("lab", "bad_event") == before

def test_broken_json_is_dropped_and_counted(): ...       # kind == "bad_json"

@pytest.mark.parametrize("event,data", [
    ("error", {"code": "oom", "message": "CUDA out of memory"}),
    (None, {"error": {"code": "oom", "message": "CUDA out of memory"}}),
])
def test_an_error_from_the_stand_stops_the_answer_and_its_text_stays_private(event, data):
    with pytest.raises(StandStreamError) as e:
        _feed(_relay(), event, data)
    classified = e.value.classified()
    assert "CUDA" not in classified.user_message and "CUDA" in classified.message

def test_an_oversized_answer_stops_with_an_error():
    r = _relay()
    _feed(r, data=chunk("я" * 40_000))
    with pytest.raises(StandStreamError, match="too_long"):
        _feed(r, data=chunk("я"))

def test_a_large_ordinary_event_is_dropped_but_details_may_be_large(): ...   # kind == "block_too_large"

async def test_relay_stream_sends_the_contract_headers():
    # POST на /v1/chat/completions; X-Meno-Request-Id == "chatcmpl-1"; X-Meno-Trace только при trace=True;
    # Authorization только при заданном ключе; тело stream == True
    ...

@pytest.mark.parametrize("status,code", [(429, "model_rate_limited"), (503, "model_unreachable"),
                                          (400, "invalid_upstream_request"), (500, "transient_upstream_5xx")])
async def test_an_http_error_before_the_stream_maps_to_our_codes(status, code): ...

async def test_retry_after_reaches_the_error(): ...
    # 429 и заголовок Retry-After: 30 дают StandStreamError с retry_after_sec == 30

async def test_relay_stream_handles_a_split_multibyte_character():
    raw = sse(block(data=chunk("Привет")), block(data="[DONE]"))
    cut = raw.index("Привет".encode()) + 1          # внутри буквы
    r = await _run_relay(pieces=[raw[:cut], raw[cut:]])
    assert r.answer == "Привет" and r.done

async def test_a_silent_stand_hits_the_total_limit():
    # поток выдал один чанк и уснул; total_timeout_s=0.05
    with pytest.raises(StandStreamError, match="transient_timeout"):
        await _run_relay(pieces=[...], total_timeout_s=0.05)
```

Интеграционные, с приложением, в том же файле: поток через `/v1/chat/completions` даёт наш `summary` с `total_ms` шлюза, а не стенда (стенд прислал 30.0); без `finish_reason` от стенда шлюз досылает завершающий чанк; `delta.role` проходит; текст ошибки стенда не виден клиенту; пустой ответ (только `[DONE]`) идёт в путь ошибки с кодом `stand_empty_answer` и не сохраняется как ответ; поток с текстом, закрытый без `[DONE]`, сохраняет ответ и считает нарушение `no_done`.

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/stands/test_stand_relay.py -q`.

- [ ] **Step 3: Реализовать `classify_stand_error` и `relay.py`.** Нарушения считаются только при `record_metrics`, `main` передаёт `record_metrics=not bench`. Если поток закончился без `[DONE]`, `relay_stream` считает нарушение `no_done`. Сверка этапа с манифестом только у стендов с манифестом; для эндпоинта без манифеста этап проходит без подписи и без нарушения. Лимит размера применять так: `iter_blocks(..., max_bytes=16_000_000)`, превышение даёт `stand_block_too_large`; прочие события больше 1 МБ отбрасываются внутри `handle`. Статусы до потока: 429 в `model_rate_limited` с `Retry-After`, 503 в `model_unreachable`, 400 и 404 в `invalid_upstream_request`, остальное в `transient_upstream_5xx`; тело ответа идёт в `message`.

Общий предел времени считать так, потому что `asyncio.timeout` вокруг `yield` отменил бы задачу потребителя:

```python
deadline = loop.time() + total_timeout_s
blocks = iter_blocks(...).__aiter__()
while True:
    try:
        block = await asyncio.wait_for(anext(blocks), max(0.0, deadline - loop.time()))
    except StopAsyncIteration:
        break
    except TimeoutError as exc:
        raise StandStreamError("transient_timeout", ...) from exc
    for text in relay.handle(block):
        yield text
```

- [ ] **Step 4: Переписать `_stand_stream_response` в `main.py`.** Порядок: блоки из `relay_stream`; если `relay.parts` пуст, `StandStreamError("stand_empty_answer", ..., retryable=True)`; `relay.closing()`; `relay.summary(total_ms)`; `[DONE]`; метрики и сохранение. При ошибке та же последовательность, что в `_stream_response` (`api/main.py:1295-1316`): этап `failed`, `event: error`, чанк с `error`, `[DONE]`. Классификация: `exc.classified()` для `StandStreamError`, иначе `classify_error(exc)`. Подключение и чтение: `httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)`, предел всего ответа из `settings.stand_total_timeout_seconds`.

- [ ] **Step 5: Перенести и обновить тесты.** Удалить из `tests/v2/test_external_backends.py` разделы про `Relay`. Потоковые тесты чата обновить: `summary` теперь от шлюза, ошибка в формате шлюза.

- [ ] **Step 6: Запустить тесты.** `pytest tests/stands/test_stand_relay.py -q`, затем `pytest tests/v2/test_external_backends.py -q`. Ожидание: PASS.

- [ ] **Step 7: Коммит:** `feat(stands): the gateway validates the stream instead of passing it through`.

---

### Task 6: Запросы к стенду: ответ без потока, лимиты, арена, вход

**Files:**
- Create: `tests/stands/test_stand_chat.py`
- Modify: `src/meno_rag/api/main.py:778-958,1515-1597`
- Delete: `src/meno_rag/api/external.py`, `tests/v2/test_external_backends.py` (оставшиеся тесты чата перенести в `test_stand_chat.py`)

**Interfaces:**
- Consumes: `relay_stream`, `Relay`, `stand_request_body`, `StandStreamError` (Task 5), `StandTarget.limiter`, `.arena`, `.requires_auth`, `StandRegistry.resolve` (Task 4), `_overloaded_response`, `_error_response`, `split_virtual_model`.
- Produces: `main._stand_non_stream_response(...)`, собирает ответ из `relay_stream`. Код ошибки `stand_not_in_arena` (403).

- [ ] **Step 1: Написать тесты**

```python
def test_the_request_carries_no_user_identity(make_client, tmp_path):
    seen = []
    c, _, _ = stand_client(make_client, tmp_path, stand_handler(seen))
    c.post("/v1/chat/completions", json=ASK, headers={"X-Auth-Token": "jwt", "X-Guest-Token": "g", "Cookie": "a=b"})
    request, body = seen[0]
    assert not {"x-auth-token", "x-guest-token", "cookie"} & set(request.headers)
    assert not {"user", "knowledge_base_id", "arena"} & body.keys()
    assert body["model"] == "gemma" and body["stream"] is True
    assert request.headers["x-meno-request-id"].startswith("chatcmpl-")

def test_a_client_without_streaming_gets_a_collected_answer(make_client, tmp_path):
    seen = []
    c, db, _ = stand_client(make_client, tmp_path, stand_handler(seen))
    r = c.post("/v1/chat/completions", json={**ASK, "stream": False}).json()
    assert seen[0][1]["stream"] is True
    assert r["choices"][0]["message"]["content"] == "Сессия в январе"
    assert r["model"] == "gemma@lab" and r["sources"][0]["source_url"] == "https://nsu.ru/g"
    assert _rows(db)[-1][2] == "gemma@lab"

def test_the_stand_limit_answers_overloaded_and_holds_no_slot(make_client, tmp_path):
    c, _, reg = stand_client(make_client, tmp_path, stand_handler([]), [{**LAB, "max_concurrent": 1}])
    target = reg.resolve("gemma@lab")
    assert target.limiter.try_acquire()
    r = c.post("/v1/chat/completions", json=ASK)
    assert r.status_code == 503 and r.json()["error"]["code"] == "overloaded"
    assert c.app.state.admission.active == 0
    target.limiter.release()

def test_slots_are_free_after_an_answer(make_client, tmp_path): ...   # для stream и non-stream: active == 0 у обоих

@pytest.mark.parametrize("model", ["gone@lab", "gemma@removed"])
def test_a_vanished_model_is_a_clear_error(make_client, tmp_path, model):
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([]))
    r = c.post("/v1/chat/completions", json={**ASK, "model": model})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_found"
    assert not c.app.state.pipeline.prepare_called

def test_a_stand_outside_the_arena_refuses_arena_requests(make_client, tmp_path):
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([]), [{**LAB, "arena": False}])
    r = c.post("/v1/chat/completions", json={**ASK, "arena": True})
    assert r.status_code == 403 and r.json()["error"]["code"] == "stand_not_in_arena"
    assert c.post("/v1/chat/completions", json=ASK).status_code == 200

def test_an_arena_stand_accepts_arena_requests(make_client, tmp_path): ...

def test_a_guest_cannot_use_an_llm_that_requires_login(make_client, tmp_path): ...  # 403 auth_required при AUTH_ENABLED

async def test_client_disconnect_frees_the_slots_and_closes_the_stand(...):
    # вызвать _stand_stream_response напрямую, получить первый блок, затем gen.aclose();
    # target.limiter.active == 0, admission.active == 0, поток стенда закрыт (своя async-итерация с флагом)

async def test_a_stand_removed_mid_answer_finishes_it(...):
    # получить первый блок, удалить стенд из файла, reg.refresh(), дочитать;
    # ответ сохранён с lab@v1, limiter.active == 0

@pytest.mark.parametrize("fault", ["hang", "garbage", "torn_connection", "huge_answer", "http_500"])
def test_a_bad_stand_does_not_touch_the_others(make_client, tmp_path, fault):
    # два стенда: bad и ok; запрос к bad завершается ошибкой или отброшенным мусором,
    # история не содержит ответа от bad при ошибке, все слоты свободны;
    # следующий запрос к gemma@ok отвечает и сохраняется.
    # hang использует малый STAND_TOTAL_TIMEOUT_SECONDS
    ...
```

`prepare_called` добавить в `FakeLegacy` через подкласс в `_helpers.py` или фиксируя вызовы мока. `tests/v2/test_api_v2.py` не менять.

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/stands/test_stand_chat.py -q`.

- [ ] **Step 3: Изменить `chat_completions`.** После определения `bench` и до общего лимита:
  - id с `@`, не разрешённый реестром и не виртуальный (`split_virtual_model`), сразу даёт 400 `model_not_found`, `param="model"`;
  - `payload.arena and not target.arena` даёт 403 `stand_not_in_arena`;
  - после общего лимита `target.limiter.try_acquire()`, при отказе вернуть общий слот и `_overloaded_response`;
  - после определения пользователя: `target.requires_auth and settings.auth_enabled and current_user is None` даёт 403 `auth_required`.

Один `release_all()` снимает оба слота, он идёт в `on_finish` для потока и в `finally` для остального. Причины отказа считаются через `_record_preflight_failure`.

- [ ] **Step 4: Реализовать `_stand_non_stream_response`.** Читает `relay_stream` до конца, отбрасывая текст блоков. Ответ: `{"id", "object": "chat.completion", "created", "model": public_id, "choices": [{"index": 0, "message": {"role": "assistant", "content": ...}, "finish_reason": "stop"}], "sources": relay.sources}`. Ошибки через `_classified_error_response`. Удалить `complete`, `upstream_body`, файл `external.py` и импорты из него.

- [ ] **Step 5: Запустить тесты.** `pytest tests/stands/test_stand_chat.py -q`, затем `pytest tests/test_admission.py -q`, затем `pytest tests/v2/test_api_v2.py -q`, затем `pytest tests/v2/test_virtual_models.py -q`. Ожидание: PASS.

- [ ] **Step 6: Коммит:** `feat(stands): per-stand limits, arena and login gates, answers collected from the stream`.

---

### Task 7: Согласие, детали запуска и бенчмарк через шлюз

**Files:**
- Create: `alembic/versions/0018_pipeline_run_stand_build.py`, `tests/stands/test_stand_trace.py`
- Modify: `src/meno_rag/db/orm.py` (`PipelineRun`), `src/meno_rag/db/repositories.py:312-345`, `src/meno_rag/api/main.py:1373-1386,1612-1757`

**Interfaces:**
- Consumes: `Relay.contexts`, `.trace`, `.stage_details`, `.pass_trace`, `relay_stream(..., trace=)` (Task 5), `StandTarget.build`.
- Produces:
  - колонка `pipeline_runs.stand_build: String(128)`, NULL; `repositories.create_pipeline_run(..., stand_build: str | None = None)`; `_persist_success(..., stand_build: str | None = None)`;
  - `main._has_improvement_consent(database, *, user_id, guest_session_id) -> bool`, при сбое БД `False`;
  - `main._stand_outcome(payload, relay, *, capture_trace: bool)`, заменяет `_external_outcome`.

- [ ] **Step 1: Написать тесты**

```python
TRACE = {"search_queries": ["сессия"], "system_prompt": "S", "user_prompt": "U", "dialogue_history": "H",
         "fewshots": [], "stage_details": {"retrieval": {"round": 1}}, "generation_model": "gemma-x",
         "core_model": "lite-x", "my_debug": 42}

def test_with_consent_the_details_reach_the_analysis_tables(make_client, tmp_path):
    seen = []
    c, db, _ = stand_client(make_client, tmp_path, stand_handler(seen, stream_body=_traced_stream(TRACE),
                                                                 manifest=manifest(build="a1b2")))
    headers = _consenting_guest(c, db)
    assert c.post("/v1/chat/completions", json=ASK, headers=headers).status_code == 200
    assert seen[0][0].headers["x-meno-trace"] == "1"
    run = _run_row(db)
    assert (run.knowledge_base_id, run.stand_build) == ("lab@v1", "a1b2")
    assert (run.generation_model, run.core_model, run.search_queries) == ("gemma-x", "lite-x", ["сессия"])
    record = _generation_row(db)
    assert (record.system_prompt, record.user_prompt, record.dialogue_history) == ("S", "U", "H")
    assert record.retrieved[0]["id"] == "1"                                 # из contexts

def test_without_consent_nothing_is_asked_and_only_history_is_stored(make_client, tmp_path):
    seen = []
    c, db, _ = stand_client(make_client, tmp_path, stand_handler(seen, stream_body=_traced_stream(TRACE)))
    c.post("/v1/chat/completions", json=ASK)
    assert "x-meno-trace" not in seen[0][0].headers
    assert _count(db, "pipeline_runs") == 0 and _count(db, "generation_records") == 0
    assert _count(db, "messages") == 2

def test_the_browser_never_sees_the_details(make_client, tmp_path):
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=_traced_stream(TRACE)))
    text = c.post("/v1/chat/completions", json=ASK, headers=_consenting_guest(c, db)).text
    assert "event: contexts" not in text and "pipeline_trace" not in text and "my_debug" not in text

def test_a_benchmark_gets_the_details_and_writes_nothing(bench_stand_client):
    r = _bench_post(stream=True)
    assert "event: contexts" in r.text and "event: pipeline_trace" in r.text
    assert _count(db, "messages") == 0 and r.headers["x-bench-mode"] == "1"

def test_a_benchmark_without_streaming_gets_the_details_in_the_body(bench_stand_client):
    body = _bench_post(stream=False).json()
    assert body["contexts"][0]["id"] == "1" and body["pipeline_trace"]["my_debug"] == 42

def test_the_opaque_trace_follows_the_sample_rate(...):
    # capture_trace=False: trace_writer.enqueue не вызван, колонки заполнены; True: вызван и содержит my_debug
    ...
```

`_consenting_guest` взять по образцу `tests/test_arena_turn_persistence.py:82`, он должен выдать согласие и `SERVICE_AND_HISTORY`, и `MENO_IMPROVEMENT`.

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/stands/test_stand_trace.py -q`.

- [ ] **Step 3: Добавить миграцию и колонку.** `revision = "0018_pipeline_run_stand_build"`, `down_revision = "0017_pipeline_run_owner"`, `op.add_column("pipeline_runs", sa.Column("stand_build", sa.String(length=128), nullable=True))`. Формат как у `0017`.

- [ ] **Step 4: Реализовать.** `want_trace = bench or await _has_improvement_consent(...)` перед запросом к стенду, `Relay(pass_trace=bench)`. `_stand_outcome` раскладывает ключи раздела 7.5 спецификации: `search_queries`; `system_prompt` и `user_prompt` как `qa_messages`; `dialogue_history`; `retrieved`, при отсутствии `relay.contexts`; `fewshots`; `stage_details`, при отсутствии `relay.stage_details`. `generation_model` и `core_model` из следа, иначе `target.model`. `trace` — весь след с добавленным `contexts`, только при `capture_trace`. В `_stand_non_stream_response` при `bench` добавить в ответ `contexts` и `pipeline_trace`.

- [ ] **Step 5: Запустить тесты.** `pytest tests/stands/test_stand_trace.py -q`, затем `pytest tests/test_migrate.py -q`, затем `pytest tests/test_generation_records_schema.py -q`, затем `pytest tests/test_bench_endpoint.py -q`. Ожидание: PASS.

- [ ] **Step 6: Коммит:** `feat(stands): consent decides what a stand is asked for, details stored as for our pipelines`.

---

### Task 8: Участник и рейтинг

**Files:**
- Create: `src/meno_rag/api/stands/participant.py`, `tests/stands/test_stand_participant.py`
- Modify: `src/meno_rag/config.py`, `src/meno_rag/schemas.py:110-113`, `src/meno_rag/api/arena.py`, `src/meno_rag/db/repositories.py`, `src/meno_rag/api/main.py` (`chat_completions`, `_stream_response`, `_non_stream_response`), `tests/test_vote_request_validation.py` и тесты из Step 5

**Interfaces:**
- Consumes: `StandRegistry.resolve`, `.live_kbs` (Task 4), `split_virtual_model`, `V2_KB_ID`.
- Produces:
  - `LEGACY_STAND_ID = "menon-1.0"`, `V2_STAND_ID = "menon-1.1"`, `UNKNOWN_VERSION = "unknown"`;
  - `Participant(stand: str, version: str, llm: str)`, frozen, со свойством `kb -> f"{stand}@{version}"`;
  - `participant_for(public_model: str, *, registry: StandRegistry | None, settings: Settings, kb_hint: str | None = None) -> Participant`;
  - `stored_participant(public_model: str, stored_kb: str | None) -> Participant | None`: `None`, если `stored_kb` не в виде `стенд@версия`;
  - `builtin_live_kbs(settings: Settings) -> set[str]`;
  - `repositories.arena_turn_sides(session, *, conversation_id: str, turn_index: int | None) -> list[dict] | None`;
  - `Settings.legacy_version` (`LEGACY_VERSION`, `"2026-07"`), `Settings.v2_version` (`V2_VERSION`, `"unversioned"`).

Порядок разрешения в `participant_for`:

| Условие | Результат |
|---|---|
| `registry.resolve(id)` нашёл цель | стенд цели, её версия, её LLM |
| `split_virtual_model(id)` виртуальный | `menon-1.1`, `settings.v2_version`, базовый id |
| В id есть `@` | стенд и LLM из `rpartition("@")`, версия `unknown` |
| `kb_hint == V2_KB_ID` | `menon-1.1`, `settings.v2_version`, id |
| Иначе | `menon-1.0`, `settings.legacy_version`, id |

- [ ] **Step 1: Написать тесты**

```python
@pytest.mark.parametrize("model,hint,expected", [
    ("gemma@lab", None, ("lab", "v1", "gemma")),
    ("gemma@menon-1.1", None, ("menon-1.1", "kb-v31h", "gemma")),
    ("gemma@removed", None, ("removed", "unknown", "gemma")),
    ("meno-lite", None, ("menon-1.0", "2026-07", "meno-lite")),
    ("meno-lite", V2_KB_ID, ("menon-1.1", "kb-v31h", "meno-lite")),
])
def test_participant_for(model, hint, expected): ...

def test_stored_participant_reads_only_the_new_form():
    assert stored_participant("gemma@lab", "lab@v1") == Participant("lab", "v1", "gemma")
    assert stored_participant("gemma@lab", "nsu-stand-faiss-bm25") is None
    assert stored_participant("gemma@lab", None) is None

def test_a_late_vote_counts_for_the_version_that_answered(make_client, tmp_path):
    c, db, reg = _arena_client(make_client, tmp_path, version="v1")
    _record_turn(c, turn_index=0)
    _restart_stand_with_version(c, reg, "v2")
    assert _vote(c, turn_index=0).status_code == 200
    assert ("gemma", "lab@v1") in _ratings(db) and ("gemma", "lab@v2") not in _ratings(db)

def test_a_late_vote_survives_the_stand_being_removed(make_client, tmp_path): ...   # рейтинг ("gemma", "lab@v1")

def test_a_vote_without_a_stored_turn_uses_the_registry(make_client, tmp_path): ...  # ("gemma", "lab@v1")

def test_a_vote_with_no_turn_and_no_stand_is_marked_unknown(make_client, tmp_path): ...  # ("gemma", "lab@unknown")

def test_a_stored_turn_for_another_model_is_not_trusted(make_client, tmp_path): ...

def test_the_client_kb_is_ignored_and_may_be_empty(make_client, tmp_path):
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    assert _vote(c, kb_a="bogus", kb_b="").status_code == 200
    assert not any(kb == "bogus" for _, kb in _ratings(db))

def test_the_stored_turn_records_the_participant(make_client, tmp_path):
    c, db, _ = _arena_client(make_client, tmp_path, version="v1")
    _record_turn(c, turn_index=0, kb="nsu-stand-faiss-bm25")
    assert _stored_sides(db)[0]["knowledge_base_id"] == "lab@v1"

def test_in_process_answers_are_labelled_by_participant(make_client, monkeypatch):
    monkeypatch.setenv("V2_VERSION", "kb-v31h")
    c, db = make_client()                                   # v2 в тестовой фикстуре включён по умолчанию
    _post_model(c, "gemma@menon-1.1")
    assert _last_message(db).knowledge_base_id == "menon-1.1@kb-v31h"

def test_the_leaderboard_marks_live_rows(make_client, tmp_path):
    # строка ("gemma", "lab@v1") live True; старая ("meno-lite", "nsu-stand-faiss-bm25") live False
    ...
```

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/stands/test_stand_participant.py -q`.

- [ ] **Step 3: Реализовать `participant.py`, настройки, схему.** `VoteRequest.kb_a`, `kb_b`: `str = Field(default="", max_length=128)`. Комментарий над полями обновить: ключ рейтинга теперь определяет сервер.

- [ ] **Step 4: Изменить `api/arena.py`.**
  - `record_turn`: для каждой стороны с `model` записать `knowledge_base_id = participant_for(side.model, ..., kb_hint=side.knowledge_base_id).kb`.
  - `submit_vote`: внутри транзакции прочитать `arena_turn_sides`. Для сторон a и b: если сохранённая сторона с тем же `key` имеет тот же `model`, взять `stored_participant`; если `None`, `participant_for`. В `payload` записать `model_a = participant.llm`, `kb_a = participant.kb`, то же для b. Ключ блокировки оставить как есть.
  - `get_leaderboard`: каждой строке добавить `live = row["knowledge_base"] in (registry.live_kbs() | builtin_live_kbs(settings))`.

- [ ] **Step 5: Изменить метку в чате и поправить тесты.** В `chat_completions` после получения `runtime`: `participant = participant_for(_public_model(payload, runtime), ..., kb_hint=kb_id)`, он передаётся в `_stream_response` и `_non_stream_response`. `kb_id` там остаётся для `_late_sources`, а в `_persist_success` и `_persist_failure` идёт `participant.kb`. Найти затронутые тесты: `grep -rl "kb_a\|knowledge_base_id" tests`. В каждом править только утверждения о клиентской метке, сохранённой базе и составе строки лидерборда. Обязательно: `tests/test_vote_request_validation.py` (пустая `kb` теперь допустима, длинная по-прежнему нет), `tests/test_arena_turn_persistence.py`, `tests/test_arena_vote_dedupe.py`, `tests/test_arena_vote_metadata.py`, `tests/test_arena_user_attribution.py`, `tests/test_leaderboard_api.py`, `tests/test_delete_subject_data.py`, `tests/v2/test_virtual_models.py`, `tests/v2/test_api_v2.py`. `tests/test_leaderboard.py` работает на уровне репозитория и меняться не должен.

- [ ] **Step 6: Запустить тесты.** `pytest tests/stands/test_stand_participant.py -q` и каждый файл из Step 5 отдельной командой. Ожидание: PASS.

- [ ] **Step 7: Коммит:** `feat(stands): the participant decides history labels and ratings, fixed when a round is stored`.

---

### Task 9: Метрики по стенду, версии и LLM

**Files:**
- Modify: `src/meno_rag/api/metrics.py:32-59,133-144`, `src/meno_rag/api/main.py` (все вызовы `record_chat_request`, `record_time_to_first_token`), `src/meno_rag/api/stands/relay.py`, `tests/test_metrics.py`, `tests/test_bench_endpoint.py`

**Interfaces:**
- Consumes: `Participant` (Task 8), `StandTarget` (Task 4), `Relay.record_metrics` (Task 5).
- Produces:
  - `metrics.record_chat_request(*, provider, stream, status, seconds, stand: str = "", version: str = "", llm: str = "")`;
  - `metrics.record_time_to_first_token(*, provider, seconds, stand: str = "", version: str = "", llm: str = "")`;
  - `metrics.record_stand_stage(*, stand: str, version: str, stage: str, seconds: float)`; серия `meno_stand_stage_seconds{stand,version,stage}`;
  - `metrics.bounded_version(stand: str, version: str) -> str`: после 32 разных версий стенда возвращает `other` и считает нарушение `too_many_versions`.

- [ ] **Step 1: Написать тесты в `tests/test_metrics.py`**

```python
def test_chat_metrics_carry_the_participant():
    metrics.record_chat_request(provider="external", stream=True, status="ok", seconds=1.0,
                                stand="ragu", version="v1", llm="gemma")
    assert 'llm="gemma",provider="external",stand="ragu",status="ok",stream="true",version="v1"' in _render()

def test_versions_are_capped_per_stand():
    for i in range(33):
        last = metrics.bounded_version("many", f"v{i}")
    assert metrics.bounded_version("many", "v0") == "v0" and last == "other"
    assert metrics.bounded_version("other-stand", "v33") == "v33"

def test_a_stage_time_is_recorded_per_version(): ...
    # record_stand_stage для v1 и v2 одного стенда даёт две серии meno_stand_stage_seconds

def test_a_stand_answer_is_labelled_end_to_end(make_client, tmp_path): ...
    # запрос к gemma@lab: chat и TTFT с stand="lab",version="v1",llm="gemma";
    # stage retrieval в meno_stand_stage_seconds; необъявленный этап как "other"

def test_in_process_answers_are_labelled_too(make_client, monkeypatch): ...
    # gemma@menon-1.1: stand="menon-1.1",version=<V2_VERSION>,llm="gemma"; provider всё ещё "vllm"

def test_a_benchmark_moves_no_stand_series(...): ...
```

- [ ] **Step 2: Запустить и увидеть падение.** `pytest tests/test_metrics.py -q`.

- [ ] **Step 3: Реализовать.** Добавить метки к `meno_chat_requests`, `meno_chat_request_seconds`, `meno_chat_time_to_first_token_seconds`. `provider` не убирать. `bounded_version` вызывается внутри функций записи, чтобы вызывающий не мог про него забыть. Все вызовы в `main.py` получают участника. Для `Relay.handle`: завершённый этап с длительностью идёт в `record_stand_stage` при `record_metrics`, необъявленное имя пишется как `other`. Для этапов v1 и v2 в процессе: то же из `_stream_response` и `outcome.stage_durations_ms` в `_non_stream_response`. Трафик бенчмарка в эти серии не идёт, как сейчас.

- [ ] **Step 4: Поправить утверждения** с полным набором меток в `tests/test_metrics.py`. В `tests/test_bench_endpoint.py` `_metric_sum` суммирует по имени и должен работать без правок, проверить запуском.

- [ ] **Step 5: Запустить тесты.** `pytest tests/test_metrics.py -q`, затем `pytest tests/test_bench_endpoint.py -q`, затем `pytest tests/test_trace_metrics.py -q`, затем `pytest tests/stands/test_stand_relay.py -q`. Ожидание: PASS.

- [ ] **Step 6: Коммит:** `feat(metrics): chat and stage series carry the stand, version and LLM`.

---

### Task 10: Запуск, CI, документация и проверка целиком

**Files:**
- Create: `stands.example.json`
- Modify: `example.env:248-253`, `scripts/run_backend.sh`, `.github/workflows/ci.yml`, `README.md`

**Interfaces:**
- Consumes: всё выше.
- Produces: `STANDS_FILE`, `STAND_TOTAL_TIMEOUT_SECONDS`, `LEGACY_VERSION`, `V2_VERSION` в `example.env`, файл-образец реестра.

- [ ] **Step 1: Обновить `example.env`.** Заменить раздел `EXTERNAL_BACKENDS` переменными из блока Produces, по строке комментария на каждую. Пояснить: запись в файле делает стенд видимым всем; `arena: true` ставится после `check-backend`; файл перечитывается без перезапуска; `LEGACY_VERSION` и `V2_VERSION` меняются при смене базы или настроек. Создать `stands.example.json` с двумя записями из раздела 8.1 спецификации, ключ в образце `"change-me"`.

- [ ] **Step 2: Обеспечить запуск без новых установок.** На проде нельзя ставить пакеты в venv, а новых сторонних зависимостей у `meno-contract` нет. В `scripts/run_backend.sh` после `cd "$ROOT_DIR"` добавить:

```bash
# meno-contract has only dependencies the venv already holds; no install is needed on the server.
export PYTHONPATH="$ROOT_DIR/packages/meno-contract/src${PYTHONPATH:+:$PYTHONPATH}"
```

- [ ] **Step 3: Добавить шаг CI.** Новая задача `contract` в `.github/workflows/ci.yml`, та же настройка uv (`version: "0.9.10"`):

```yaml
- name: The contract package installs and imports without torch
  run: |
    uv sync --package meno-contract --frozen --no-default-groups
    .venv/bin/python -c "import sys, meno_contract.check, meno_contract.sse; assert 'torch' not in sys.modules"
```

В задаче `quality` ничего менять не нужно, она запускает все тесты. Локально проверить в отдельном каталоге, чтобы не создавать `.venv` с `torch` в рабочем:

```bash
V=$(mktemp -d)/venv
UV_PROJECT_ENVIRONMENT="$V" uvx uv@0.9.10 sync --package meno-contract --frozen --no-default-groups
"$V/bin/python" -c "import sys, meno_contract.check, meno_contract.sse; assert 'torch' not in sys.modules"
```

Ожидание: ставятся только `pydantic`, `httpx` и их зависимости, импорт без ошибок. Если эта версия `uv` не принимает какой-то флаг, подобрать рабочий набор и обновить шаг CI.

- [ ] **Step 4: Дополнить `README.md`.** Короткий раздел «Стенды»: что это, как подключить (`STANDS_FILE`, запись, `POST /v1/models/refresh`), ссылка на `docs/contract/meno-backend-v1.md` и `stands.example.json`.

- [ ] **Step 5: Проверить вживую, на настоящих сокетах.** v1 и v2 выключены, поэтому модели не загружаются.

```bash
cd ~/Meno-RAG-Core-gateway
T=$(mktemp -d)
echo '[{"id": "example", "url": "http://127.0.0.1:9299", "arena": true}]' > "$T/stands.json"
export PYTHONPATH=src:packages/meno-contract/src DATABASE_URL="sqlite+aiosqlite:///$T/e2e.sqlite3"
export LEGACY_ENABLED=false V2_ENABLED=false STANDS_FILE="$T/stands.json" VLLM_ENDPOINTS=http://127.0.0.1:9 ENV_FILE=/dev/null
PY=~/Meno-RAG-Core/.venv/bin/python
$PY -m meno_rag.db.migrate
timeout 90 $PY -m meno_contract.example_stand --port 9299 &
timeout 90 $PY -m uvicorn meno_rag.api.main:app --port 9298 &
sleep 25
curl -s localhost:9298/v1/models | $PY -c "import json,sys; print([(m['id'],m['version'],m['arena']) for m in json.load(sys.stdin)['data'] if m['provider']=='external'])"
curl -sN localhost:9298/v1/chat/completions -H 'content-type: application/json' -d '{"model":"echo@example","stream":true,"messages":[{"role":"user","content":"Привет"}]}'
curl -s localhost:9298/metrics | grep 'stand="example"' | head
```

Ожидание: в `/v1/models` запись `('echo@example', 'example-1', True)`; в потоке этапы с `labels`, текст, `sources`, `summary`, `[DONE]`, нет `contexts`; в метриках серии с `stand="example",version="example-1",llm="echo"`. Если старт требует других переменных окружения, задать их и записать сюда итоговый набор.

- [ ] **Step 6: Проверить качество кода по всем изменённым файлам.**

```bash
~/Meno-RAG-Core/.venv/bin/ruff check src tests packages
~/Meno-RAG-Core/.venv/bin/ruff format --check src tests packages
~/Meno-RAG-Core/.venv/bin/mypy src/meno_rag/api/stands packages/meno-contract/src
grep -rn "EXTERNAL_BACKENDS\|external_backends\|api.external\|ExternalRegistry" src tests example.env README.md
```

Ожидание: ruff и mypy без ошибок, `grep` ничего не находит.

- [ ] **Step 7: Коммит:** `chore(stands): configuration, startup, CI and docs for the stand registry`.

---

## После плана: только с разрешения владельца

Эти действия затрагивают сервер, GitHub или других людей и в план исполнения не входят:

1. **Полный набор тестов** на сервере, в копии `~/meno-v2`. Здесь его запускать нельзя.
2. **Пуш ветки** `feat/gateway-stands`.
3. **Перед деплоем проверить на проде:** что запущено, задан ли `EXTERNAL_BACKENDS` (если да, перенести в файл реестра), выкачен ли Meno-Web `feat/external-backends` (`0a7f20d`).
4. **Деплой:** пара скриптов `deploy` и `rollback` в `~/meno-v2`, как раньше. Выставить `V2_VERSION` по реально используемой базе. Файл реестра может быть пустым: стенды подключаются позже, без перезапуска. Проверить запросы в Grafana (Meno-Deploy) с новыми метками.
5. **Передать `docs/contract/meno-backend-v1.md` авторам RAGU и CAG** для отзыва (фаза 1 спецификации).

## Следующие планы

- **План 2:** Meno-Web (фаза 3), после согласования макетов.
- **План 3:** бенчмарк (фаза 4): `contexts` в раннере, версия в записи прогона, прямой путь в стенд.
- **План 4:** v2 и v1 как стенды, отдельный пакет шлюза (фазы 5–6), после решения по `feat/adaptive-search`.
- **План 5:** общие сервисы (фаза 7), когда понадобится ещё один стенд с поиском на GPU.
