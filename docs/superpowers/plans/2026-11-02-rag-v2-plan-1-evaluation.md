# Менон RAG v2 — План 1: инструмент оценки качества и скорости

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Расширить Meno-QA-Benchmark до инструмента, который правдоподобно оценивает качество ответов, источников, удержание контекста и скорость любой версии Менона, и снять на нём базовую линию текущего прода.

**Architecture:** Бенчмарк хранит «серебряный» набор вопросов (`data/v2/items.jsonl`) с ключевыми фактами, привязанными к цитатам из закреплённого релиза `kb/chunks/v1` датасета HF. Прогон идёт потоково через bench-эндпоинт Менона (OpenAI-совместимый `/v1/chat/completions` с Bearer-токеном бенча) и записывает ответ, время до первых слов, стадии, источники и использованный контекст. Оценка — судья (Claude через офлайн-экспорт задач или любой OpenAI-совместимый эндпоинт) плюс детерминированные проверки; отдельно — быстрый тест поиска по файлам выдачи и нагрузочный замер скорости, который запускается прямо на сервере без зависимостей.

**Tech Stack:** Python ≥ 3.12, uv, pytest, httpx (async SSE), стандартная библиотека для `speed.py`; существующие RAGAS-команды не трогаем.

**Spec:** `docs/superpowers/specs/2026-11-02-rag-v2-design.md` (разделы 3.4, 10, 4.1). Это первый из серии планов; планы 2 (индекс v2 и сравнения стека), 3 (пайплайн v2) и 4 (арена и выкатка) пишутся после того, как этот инструмент заработает: их выборы зависят от его замеров.

## Global Constraints

- Репозиторий: `Meno-NSU/Meno-QA-Benchmark`, локальный клон `/home/devuser/Meno-QA-Benchmark`, ветка `feat/bench-v2`. Доступ — read-only deploy key `~/.ssh/meno_qa_benchmark_deploy`; пуш требует права на запись, которые даёт владелец (последний шаг Task 13).
- Все сетевые запросы и все вызовы LLM — только с локальной машины. На прод-сервере (`meno`, доступ через `scratchpad/ssh/p`) — только запуск скриптов: системным `python3` (3.12.3) или `~/Meno-Stand/Meno-Stand-RAG-Core/.venv/bin/python` (bm25s, nltk, razdel, httpx, numpy уже стоят). Ничего не устанавливать в venv прода, не запускать pip/uv на сервере, не останавливать прод, не трогать сервисы пользователя `rsolikhov` и чужой Qdrant.
- Закреплённый релиз базы: датасет `Meno-NSU/NSU-Data-For-Meno`, ревизия `f374a7ed` (KB v2.5), `kb/chunks/v1`: 104 767 документов, 520 414 кусков, 476 869 индексируемых (`stats.json`). Ревизия записывается в `data/v2/README.md` и в `provenance` каждого примера.
- Реальные диалоги из дампа прода и Telegram-бота можно использовать дословно (все авторы — участники команды с письменным согласием). В бенчмарк не переносятся e-mail, телефоны, id пользователей и сессий, никнеймы.
- Прогоны качества по проду: параллелизм ≤ 4. Замеры скорости на проде (параллелизм 1/8/16) — только вечером, когда трафик минимален.
- Судья для решающих прогонов — Claude (вызовы с локальной машины через офлайн-экспорт задач); everest:9022 судьёй не использовать.
- Набор помечается «серебряным» (собран LLM, без проверки человеком).

## Review Focus

1. Поток SSE оборвался посреди ответа или пришло `event: error` — запись хранит частичный текст, помечена `ok=False` и не считается «честным отказом» (тест в Task 4).
2. Очень длинная история (`long_dialog`) не помещается в контекст тестируемой системы и даёт 4xx — ошибка учитывается в покрытии категории, а не пропадает молча (тест в Task 7).
3. URL источника отличается от эталона схемой, `www.`, слешем в конце, `utm_*` или фрагментом — считается тем же источником (тест в Task 2).
4. Судья вернул невалидный JSON или неполный вердикт — одна повторная попытка, затем `judge_error`, пример исключается из метрики с учётом в `coverage` (тест в Task 6).
5. Цитата отличается от текста документа типографикой (ё/е, «» и "", неразрывный пробел, мягкий перенос, `|` и `**` разметки) — проверка проходит; пересказ вместо цитаты — не проходит (тест в Task 3).

---

### Task 1: Окружение и каркас

**Files:**
- Modify: `pyproject.toml` (dev-группа: `pytest>=8`; зависимость `httpx>=0.27`)
- Modify: `.gitignore` (добавить `work/`, `results/`)
- Create: `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: рабочее окружение `uv run pytest`; каталог `work/` (не коммитится) для рабочих файлов сборки набора.

- [ ] **Step 1:** Скачать статический бинарь uv с GitHub Releases в `~/.local/bin/uv` (локально pip нет). Проверка: `~/.local/bin/uv --version` печатает версию.
- [ ] **Step 2:** Склонировать репозиторий deploy-ключом в `/home/devuser/Meno-QA-Benchmark` (`GIT_SSH_COMMAND="ssh -i ~/.ssh/meno_qa_benchmark_deploy -o IdentitiesOnly=yes -o UserKnownHostsFile=<scratchpad>/github_known_hosts"`), создать ветку `feat/bench-v2`.
- [ ] **Step 3:** Написать `tests/test_smoke.py::test_package_imports` — `import meno_qa_benchmark` и наличие `main`.
- [ ] **Step 4:** `uv sync --all-groups` и `uv run pytest -q` → `1 passed`.
- [ ] **Step 5:** Commit `chore: dev environment, pytest, work/ ignored`.

### Task 2: Схема набора v2 и нормализация URL

**Files:**
- Create: `src/meno_qa_benchmark/schema.py`, `src/meno_qa_benchmark/textnorm.py` (в этой задаче — только `norm_url`)
- Test: `tests/test_schema.py`, `tests/test_textnorm.py`

**Interfaces:**
- Produces:
  - `norm_url(u: str) -> str` (в `textnorm.py`) — схема → `https`, хост в нижнем регистре без `www.`, без фрагмента, без параметров `utm_*`, без слеша в конце пути.
  - `CATEGORIES = ("factual", "complex", "multi_turn", "long_dialog", "meta_dialog", "unanswerable", "time_sensitive", "general", "chitchat", "off_topic", "prompt_injection")`
  - `TOPICS = ("schedule", "cii_labs", "people", "phd", "masters", "admission", "student_life", "scholarships", "international", "general_knowledge", "other")`
  - `NO_RETRIEVAL_CATEGORIES = frozenset({"meta_dialog", "chitchat", "off_topic", "prompt_injection"})`
  - `@dataclass(frozen=True) Turn(role: Literal["user","assistant"], content: str)`
  - `@dataclass(frozen=True) KeyFact(fact: str, quote: str, source: Literal["kb","history"], chunk_id: str = "", doc_id: str = "", url: str = "")`
  - `@dataclass(frozen=True) BenchItem(id: str, question: str, ground_truth: str, category: str, topic: str, history: tuple[Turn, ...] = (), key_facts: tuple[KeyFact, ...] = (), as_of: str | None = None, answerable: bool = True, gold_urls: tuple[str, ...] = (), expect_sources: bool = True, provenance: dict = field(default_factory=dict))`
  - `load_items(path: Path) -> list[BenchItem]` — `.json` (старый формат: список `{question, ground_truth}`) или `.jsonl` (v2).
  - `dump_items(items: Sequence[BenchItem], path: Path) -> None` (JSONL, `ensure_ascii=False`).
  - `validate_item(item: BenchItem) -> list[str]` — список проблем; пустой — валиден.

- [ ] **Step 1: Тесты**

```python
def test_legacy_json_loads_as_factual(tmp_path):
    p = tmp_path / "data.json"; p.write_text('[{"question": "Q?", "ground_truth": "A."}]', encoding="utf-8")
    [it] = load_items(p)
    assert (it.id, it.category, it.topic, it.answerable) == ("legacy-000", "factual", "other", True)

def test_roundtrip_jsonl(tmp_path): ...  # dump_items -> load_items даёт равные объекты, включая history и key_facts

def test_validate_rules():
    # каждое правило — отдельный assert на наличие подстроки в validate_item(...):
    # "unknown category", "unknown topic", "answerable item needs key_facts" (factual без фактов),
    # "unanswerable item must not have key_facts", "history must alternate user/assistant",
    # "history must end with assistant", "time_sensitive needs as_of", "as_of must be ISO date",
    # "no_retrieval category must not expect sources", "gold_urls must include key fact urls",
    # "contains e-mail" (адрес в question/history/ground_truth)

def test_norm_url_variants():                     # Review Focus 3
    base = "https://nsu.ru/n/education/stipendii"
    for u in ["http://www.nsu.ru/n/education/stipendii/", "https://NSU.ru/n/education/stipendii#top",
              "https://nsu.ru/n/education/stipendii?utm_source=vk"]:
        assert norm_url(u) == base
```

- [ ] **Step 2:** `uv run pytest tests/test_schema.py tests/test_textnorm.py -v` → FAIL (нет модулей).
- [ ] **Step 3:** Реализовать `norm_url` и `schema.py`. Для категорий из `NO_RETRIEVAL_CATEGORIES` `expect_sources` обязан быть `False`; для `meta_dialog` факты имеют `source="history"`; `gold_urls` и URL фактов сравниваются через `norm_url`.
- [ ] **Step 4:** Тесты → PASS.
- [ ] **Step 5:** Добавить CLI-команду `validate <path>` в `__init__.py` (печатает `id: проблема` и код возврата 1, если проблемы есть). Commit `feat: bench item schema v2 with legacy loader and validator`.

### Task 3: Нормализация текста, загрузка базы, проверка цитат

**Files:**
- Modify: `src/meno_qa_benchmark/textnorm.py`
- Create: `src/meno_qa_benchmark/kb.py`
- Test: `tests/test_textnorm.py`, `tests/test_kb.py` (фикстура: два маленьких `part-*.jsonl.gz` в `tests/fixtures/kb/`)

**Interfaces:**
- Consumes: `norm_url` (Task 2).
- Produces:
  - `norm_text(s: str) -> str` — NFKC, `ё→е`, кавычки «»„“” → `"`, тире/дефисы → `-`, удалить мягкий перенос и нулевую ширину, убрать `|` и `**`/`__` разметки, схлопнуть пробелы, casefold.
  - `quote_in(quote: str, text: str) -> bool` — `norm_text(quote)` подстрока `norm_text(text)`; цитата короче 3 слов → `False`.
  - `@dataclass(frozen=True) KbChunk(chunk_id: str, doc_id: str, url: str, title: str, text: str, date: str | None, indexed: bool)`
  - `load_chunks(parts_dir: Path, wanted: set[str] | None = None) -> dict[str, KbChunk]` — потоково читает `part-*.jsonl.gz` релиза `kb/chunks/v1` (поля `chunk_id, doc_id, url, title, text, date, indexed`), при `wanted` держит в памяти только нужные.
  - `verify_quotes(item: BenchItem, chunks: Mapping[str, KbChunk]) -> list[str]` — для `source="kb"`: чанк существует, `quote_in(quote, chunk.text)`, `url` факта совпадает с `norm_url(chunk.url)`; для `source="history"`: цитата есть в одном из ходов истории.

- [ ] **Step 1: Тесты**

```python
def test_quote_typography_passes():                 # Review Focus 5
    assert quote_in('Стипендия назначается «по итогам» сессии', "| Стипендия назначается \"по итогам\" сессии |")
    assert quote_in("ещё раз проверим", "Еще раз про­верим")

def test_paraphrase_fails():
    assert not quote_in("стипендию платят после сессии", "Стипендия назначается по итогам сессии")

def test_verify_quotes_reports_missing_chunk_and_bad_quote(): ...  # два факта -> две проблемы с chunk_id в тексте
```

- [ ] **Step 2:** Тесты → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** Тесты → PASS (включая `tests/test_schema.py`).
- [ ] **Step 5:** Commit `feat: text/url normalization, kb chunk loader, quote verification`.

### Task 4: Потоковый прогон через bench-эндпоинт

**Files:**
- Create: `src/meno_qa_benchmark/runner.py`
- Modify: `src/meno_qa_benchmark/config.py` (поля `knowledge_base_id: str | None` из `MENO_KB_ID`), `src/meno_qa_benchmark/__init__.py` (команда `run`)
- Test: `tests/test_runner.py` (фейковый SSE-сервер на `httpx.MockTransport`)

**Interfaces:**
- Consumes: `BenchItem`, `load_items` (Task 2).
- Produces:
  - `build_messages(item: BenchItem) -> list[dict]` — история как реплики с ролями + текущий вопрос последним `user`.
  - `parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, str]]` — пары `(event, data)`; строки без `event:` → событие `"message"`.
  - `@dataclass RunRecord(item_id: str, ok: bool, error: str | None, http_status: int | None, response: str, ttft_s: float | None, total_s: float, stages_ms: dict[str, float], sources: list[dict], contexts: list[dict], search_queries: list[str], model: str, knowledge_base_id: str | None)`; `contexts` — куски с `kept=True` из `pipeline_trace.rerank.candidates`, по `rank`, каждый `{chunk_id: str, url, title, text, rank}` из `pipeline_trace.chunks` (id приводится к `str`); `search_queries` — из `pipeline_trace.search_queries` (пустой список, если поиска не было).
  - `async run_items(items: Sequence[BenchItem], *, base_url: str, api_key: str, model: str, knowledge_base_id: str | None, concurrency: int, timeout: float) -> list[RunRecord]`
  - `save_records(records, path: Path) -> None`, `load_records(path: Path) -> list[RunRecord]` (JSONL).
- Контракт эндпоинта (из `Meno-Stand-RAG-Core src/meno_rag/api/main.py`, `events.py`): `POST {base_url}/chat/completions`, тело `{"model", "messages", "stream": true, "knowledge_base_id"?}`, заголовок `Authorization: Bearer <bench token>`; события `stage` (поле `stage`, `status`, `duration_ms`), `summary` (`total_ms`, `stages`), `sources` (`{"sources": [...]}`), `pipeline_trace` (`{"pipeline_trace": {...}}`), `error`; контент — `data:` с OpenAI-чанком (`choices[0].delta.content`), конец — `data: [DONE]`.

- [ ] **Step 1: Тесты**

```python
async def test_records_ttft_sources_contexts_and_stages(): ...
    # фейковый поток: stage(query_rewrite, 380ms), sources, 2 content-дельты, pipeline_trace c двумя kept-кусками, [DONE]
    # assert rec.ok and rec.response == "Привет, мир" and rec.ttft_s is not None and rec.ttft_s <= rec.total_s
    # assert [c["chunk_id"] for c in rec.contexts] == ["7", "3"]   # порядок по rank, id строками
    # assert rec.stages_ms["query_rewrite"] == 380.0 and rec.sources[0]["source_url"]

async def test_stream_error_keeps_partial_text():          # Review Focus 1
    # одна content-дельта, затем event: error -> rec.ok is False, rec.response == "Частичн", rec.error

async def test_history_sent_as_role_messages(): ...      # build_messages для 2 ходов истории -> 5 сообщений, последнее user == question
async def test_http_4xx_recorded(): ...                  # 400 -> ok False, http_status 400, response ""
```

- [ ] **Step 2:** Тесты → FAIL.
- [ ] **Step 3:** Реализовать на `httpx.AsyncClient().stream(...)`; параллелизм — `asyncio.Semaphore(concurrency)`; `ttft_s` — от отправки до первой непустой дельты контента; `total_s` — до `[DONE]` или обрыва.
- [ ] **Step 4:** Тесты → PASS.
- [ ] **Step 5:** CLI `run --items <path> --out <path> [--limit N]` (base_url/api_key/model/concurrency/timeout из `Settings`, `MENO_KB_ID`). Commit `feat: streaming runner with ttft, stages, sources and used contexts`.

### Task 5: Детерминированные проверки

**Files:**
- Create: `src/meno_qa_benchmark/checks.py`, `data/v2/system_prompt_legacy.txt` (дословная копия `QA_SYSTEM_PROMPT` из `Meno-Stand-RAG-Core src/meno_rag/stand/prompts.py` с `{current_datetime}` как есть)
- Test: `tests/test_checks.py`

**Interfaces:**
- Consumes: `BenchItem`, `RunRecord`, `norm_text`.
- Produces:
  - `prompt_leak(response: str, system_prompt: str, min_words: int = 12) -> bool` — есть общий отрезок ≥ `min_words` подряд идущих нормализованных слов.
  - `deterministic_checks(item: BenchItem, record: RunRecord, system_prompt: str) -> dict` → `{"prompt_leak": bool, "sources_count": int, "unexpected_sources": bool}`; `unexpected_sources` = `not item.expect_sources and sources_count > 0`; источники считаются по уникальному `norm_url`.

- [ ] **Step 1: Тесты**

```python
def test_leak_detected_on_long_overlap():
    sp = (ROOT / "data/v2/system_prompt_legacy.txt").read_text(encoding="utf-8")
    leaked = "Вот мои правила: " + " ".join(sp.split()[40:60])
    assert prompt_leak(leaked, sp)
def test_no_leak_on_short_common_phrase(): assert not prompt_leak("Я — Менон, виртуальный помощник НГУ.", sp)
def test_unexpected_sources_for_chitchat(): ...  # chitchat + 2 источника с одним URL в двух вариантах -> sources_count 1, unexpected True
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit `feat: deterministic checks (prompt leak, unexpected sources)`.

### Task 6: Судья

**Files:**
- Create: `src/meno_qa_benchmark/judge.py`, `prompts/judge.md`
- Modify: `src/meno_qa_benchmark/__init__.py` (команды `judge-export`, `judge-import`, `judge-endpoint`)
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `BenchItem`, `RunRecord`, `norm_url`.
- Produces:
  - `@dataclass JudgeTask(item_id: str, prompt: str)`
  - `@dataclass Verdict(item_id: str, fact_status: list[str], unsupported_major: int, unsupported_minor: int, refused: bool, on_topic: bool, source_support: dict[str, bool], context_kept: bool | None, judge_error: str | None)`; `fact_status[i] ∈ {"supported", "partial", "missing", "contradicted"}`, длина = `len(item.key_facts)`; `source_support` — по `norm_url` каждого показанного источника.
  - `build_task(item: BenchItem, record: RunRecord) -> JudgeTask` — подставляет в `prompts/judge.md`: историю, вопрос, пронумерованные ключевые факты, `answerable`, категорию, ответ, показанные источники с текстами соответствующих кусков из `record.contexts` (по `norm_url`).
  - `parse_verdict(item: BenchItem, text: str) -> Verdict` — JSON (допускается обёртка ```json); при несоответствии схеме — `VerdictError`.
  - `export_tasks(tasks: Sequence[JudgeTask], path: Path) -> None`; `import_verdicts(items: Mapping[str, BenchItem], path: Path) -> list[Verdict]` — файл ответов `{"item_id", "raw"}`; невалидный `raw` → `Verdict` с `judge_error`.
  - `async judge_endpoint(tasks, items, *, base_url: str, api_key: str, model: str, concurrency: int, extra_body: dict) -> list[Verdict]` — на невалидный ответ одна повторная попытка, затем `judge_error`.
  - `agreement(a: Sequence[Verdict], b: Sequence[Verdict]) -> dict` — согласие двух судей по одним и тем же примерам: доля совпадений `fact_status` (по фактам), `refused`, `unsupported_major ≥ 1`, `source_support` (по источникам), `context_kept`; примеры с `judge_error` у любого судьи пропускаются и считаются в `skipped`. CLI `judge-agreement <verdicts_a> <verdicts_b>` — для калибровки дешёвого судьи против Claude.
- Требования к `prompts/judge.md`: русский язык; судья видит только приведённые тексты; «unsupported» — утверждения об НГУ/Академгородке, которых нет ни в ключевых фактах, ни в текстах источников (`major` — цифры, даты, имена, правила; иначе `minor`); `refused` — ответ сообщает, что информации нет; `context_kept` — только для `multi_turn`, `long_dialog`, `meta_dialog` (иначе `null`); для категории `general` ответ из общих знаний не считается unsupported, если не утверждает фактов об НГУ; правила сопоставления людей и организаций (фамилия — главный идентификатор, конфликт атрибутов = разные сущности) — переписать из `SYSTEM_PROMPT_FOR_RELEVANCE` (`Meno-Stand-RAG-Core src/meno_rag/stand/prompts.py`); ответ — строго JSON по схеме `Verdict` без `item_id`/`judge_error`.

- [ ] **Step 1: Тесты**

```python
def test_parse_valid_and_fenced(): ...           # оба варианта дают одинаковый Verdict
def test_wrong_fact_count_raises(): ...          # 2 факта в item, 3 статуса -> VerdictError
async def test_endpoint_retries_once_then_judge_error():   # Review Focus 4
    # MockTransport: два невалидных ответа подряд -> Verdict.judge_error is not None, ровно 2 запроса
def test_task_contains_source_texts(): ...       # текст куска из record.contexts попадает в prompt для показанного источника
def test_agreement_counts_and_skips_errors(): ... # 3 примера, у одного judge_error -> skipped 1; fact-согласие по 2 оставшимся
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit `feat: judge (prompt, verdict schema, offline export/import, endpoint backend)`.

### Task 7: Метрики и сравнение версий

**Files:**
- Create: `src/meno_qa_benchmark/metrics.py`, `src/meno_qa_benchmark/report.py`
- Modify: `src/meno_qa_benchmark/__init__.py` (команды `metrics`, `compare`)
- Test: `tests/test_metrics.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `BenchItem`, `RunRecord`, `Verdict`, `deterministic_checks`.
- Produces:
  - `aggregate(items, records, verdicts, checks) -> dict` со структурой `{"overall": M, "by_category": {cat: M}, "by_topic": {topic: M}}`, где `M` содержит ровно эти ключи:
    `n`, `errors` (record.ok False), `judge_errors`, `fact_recall` (среднее по answerable: (supported + 0.5·partial)/n_facts), `contradiction_rate`, `hallucination_rate` (доля с `unsupported_major ≥ 1`), `unanswerable_refusal` (доля `refused` среди `answerable=False`), `false_refusal` (доля `refused` среди answerable с фактами из базы), `source_precision` (поддерживающие / показанные, по ответам с ≥ 1 источником), `source_recall` (доля answerable, где хотя бы один показанный `norm_url` ∈ `gold_urls`), `unexpected_sources_rate`, `unexpected_retrieval_rate` (доля примеров из `NO_RETRIEVAL_CATEGORIES`, где `record.search_queries` непуст — мера ошибок маршрутизации), `sources_per_answer_median`, `context_retention` (доля `context_kept` где не `null`), `prompt_leak_rate`, `on_topic_rate`, `ttft_p50`, `ttft_p90`, `total_p50`, `total_p90`.
    Метрика, для которой нет ни одного пригодного примера, — `None`; ошибочные записи исключаются из качественных метрик, но видны в `errors`.
  - `DEFAULT_TOL = {"fact_recall": 0.03, "source_precision": 0.03, "source_recall": 0.03, "context_retention": 0.03, "unanswerable_refusal": 0.05, "hallucination_rate": 0.02, "unexpected_sources_rate": 0.02, "prompt_leak_rate": 0.0}`
  - `compare(baseline: dict, candidate: dict, tol: Mapping[str, float] = DEFAULT_TOL) -> tuple[bool, str]` — `(прошло, markdown)`: прошло, если ни одна метрика `tol` ни в `overall`, ни в одной категории не хуже базовой больше допуска (для `*_rate` хуже = больше, для остальных — меньше) и доля `errors` в каждой категории не выросла больше чем на 0.02.

- [ ] **Step 1: Тесты**

```python
def test_fact_recall_partial_counts_half(): ...       # 2 факта: supported+partial -> 0.75
def test_errors_counted_not_dropped():                # Review Focus 2
    # long_dialog: 3 записи, 1 с ok=False -> by_category["long_dialog"]["errors"] == 1, fact_recall по 2 оставшимся
def test_metric_none_without_eligible_items(): ...    # нет unanswerable -> unanswerable_refusal is None
def test_unexpected_retrieval_for_chitchat(): ...     # 2 chitchat, у одного search_queries непуст -> 0.5
def test_compare_flags_category_regression(): ...     # overall лучше, но в "multi_turn" fact_recall -0.05 -> (False, "...multi_turn...")
def test_compare_passes_within_tolerance(): ...
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** CLI: `metrics --items --records --verdicts --system-prompt --out` и `compare <baseline.json> <candidate.json>` (код возврата 1 при провале). Commit `feat: metrics aggregation and version comparison with per-category regression rule`.

### Task 8: Быстрый тест поиска

**Files:**
- Create: `src/meno_qa_benchmark/retrieval_eval.py`
- Modify: `src/meno_qa_benchmark/__init__.py` (команда `retrieval-eval`)
- Test: `tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes: `BenchItem`, `RunRecord`, `norm_url`.
- Produces:
  - Формат выдачи (контракт для планов 2 и 3): JSONL `{"item_id": str, "ranked": [{"doc_id": str, "url": str, "chunk_id": str, "score": float}]}` — по убыванию.
  - `@dataclass RetrievalResult(item_id: str, ranked: list[dict])`
  - `load_results(path: Path) -> list[RetrievalResult]`; `from_run_records(records: Sequence[RunRecord]) -> list[RetrievalResult]` — выдача старого прода = `contexts` по рангу (`doc_id` пустой).
  - `evaluate_retrieval(items, results, ks: Sequence[int] = (1, 3, 5, 10, 20)) -> dict` — по примерам с `answerable=True`, категория не в `NO_RETRIEVAL_CATEGORIES`, есть `key_facts` с `source="kb"`; попадание: `doc_id` из ключевых фактов (если в выдаче есть `doc_id`), иначе `norm_url` ∈ `gold_urls`; документы в выдаче схлопываются до первого вхождения; метрики `recall@k`, `mrr`, `n`, `missing_results` — общие и по категориям.

- [ ] **Step 1: Тесты**

```python
def test_recall_by_doc_id_dedupes_chunks(): ...     # 3 куска одного документа в топе -> одна позиция
def test_falls_back_to_url_for_legacy(): ...        # doc_id пустые, url с www и слешем -> попадание
def test_skips_chitchat_and_unanswerable(): ...
def test_missing_result_counted_as_miss(): ...      # пример без строки в выдаче -> missing_results 1, recall учитывает как промах
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit `feat: retrieval eval over ranked results (doc id or normalized url)`.

### Task 9: Замер скорости без зависимостей

**Files:**
- Create: `src/meno_qa_benchmark/speed.py` (только стандартная библиотека: `http.client`, `threading`/`concurrent.futures`, `json`, `argparse`, `statistics`)
- Test: `tests/test_speed.py` (локальный `http.server` в потоке, отдающий SSE с задержками)

**Interfaces:**
- Produces:
  - Запуск на сервере: `python3 speed.py --url http://127.0.0.1:9006/v1/chat/completions --token-env BENCH_TOKEN --items items.jsonl --model meno-lite [--kb-id ID] --concurrency 1,8,16 --n 40 --categories factual,complex,multi_turn --out speed.json`
  - `percentiles(values: Sequence[float], ps: Sequence[int] = (50, 90, 99)) -> dict[str, float]`
  - `run_level(url, token, bodies: Sequence[dict], concurrency: int, timeout: float) -> list[dict]` — на запрос `{"ok", "ttft_s", "total_s", "error"}`; `n` запросов, `concurrency` одновременно.
  - Выход `speed.json`: `{"levels": {"1": {"ttft": {...}, "total": {...}, "errors": int, "n": int}, ...}, "meta": {...}}`.
  - Читает `items.jsonl` сам (без импорта `schema.py`), историю превращает в сообщения так же, как `build_messages`.

- [ ] **Step 1: Тесты** — `test_percentiles_exact` (на [1..10]: p50 = 5.5, p90 = 9.1 по линейной интерполяции); `test_run_level_measures_ttft_below_total` (фейковый сервер: 0,2 с до первой дельты, 0,5 с до конца → 0,15 < ttft < 0,45 < total); `test_errors_counted` (сервер отвечает 500 на каждый второй).
- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit `feat: dependency-free speed runner for on-server latency measurements`.

### Task 10: Кандидаты вопросов из реальных диалогов и из базы

**Files:**
- Create: `src/meno_qa_benchmark/candidates/__init__.py`, `candidates/dialogs.py`, `candidates/kbdocs.py`
- Test: `tests/test_candidates.py` (мини-дамп в `tests/fixtures/dump/main/messages.jsonl`)

**Interfaces:**
- Consumes: `Turn`, `norm_text`.
- Produces:
  - `@dataclass DialogCandidate(source: Literal["prod", "telegram"], question: str, history: tuple[Turn, ...], conversation_ref: str)` — `conversation_ref` = sha1 от исходного id (сам id не сохраняется).
  - `extract_prod_dialogs(dump_dir: Path, max_history_turns: int = 8) -> list[DialogCandidate]` — `main/messages.jsonl` дампа (`conversation_id`, `role`, `content`, `created_at`, `turn_kind`), сгруппировать по диалогу, упорядочить по `created_at`, пропустить арена-дубли (`turn_kind != "answer"` или подряд два `user`), кандидат на каждую реплику пользователя с предыдущими ходами.
  - `extract_telegram_questions(path: Path) -> list[DialogCandidate]` — `questions.ndjson` (поле `content`, `is_question`).
  - `strip_pii(text: str) -> str` — e-mail → `[email]`, телефоны (`+7…`, `8…`, 10–11 цифр с разделителями) → `[телефон]`.
  - `is_trash(text: str) -> bool` — пустое/только знаки, < 2 букв, повтор одного символа ≥ 6 раз, длиннее 2000 символов.
  - `dedupe(cands: Sequence[DialogCandidate]) -> list[DialogCandidate]` — без истории: совпадение `norm_text` или Jaccard слов ≥ 0.8; с историей — по паре (последний ответ ассистента, вопрос).
  - `TOPIC_RULES: dict[str, tuple[str, ...]]` — регулярки по URL и названию документа для `schedule` (`table.nsu.ru`, `расписан`), `cii_labs` (`ai.nsu.ru`, `ЦИИ`, `искусственн`, `лаборатор`), `people` (`pure.nsu.ru/portal/ru/persons`, `doc_type` карточка человека), `phd` (`аспирант`, `/phd/`), `masters` (`магистр`, `/master`), `admission` (`priem.nsu.ru`, `поступ`, `приём`), `student_life` (`общежит`, `/life/`), `scholarships` (`стипенд`), `international` (`/international/`, `иностран`).
  - `sample_docs(docs_path: Path, per_topic: Mapping[str, int], seed: int = 20261102) -> list[dict]` — из `kb/chunks/v1/docs.jsonl.gz` только документы уровней P0–P2, без дублей, детерминированно.

- [ ] **Step 1: Тесты** — извлечение с историей и без арена-дублей; `strip_pii` на e-mail и двух форматах телефонов; `is_trash` на «.», «ааааааа», нормальном вопросе; `dedupe` склеивает «Где находится НГУ?» и «где находится нгу»; `sample_docs` детерминирован при одном seed.
- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit `feat: candidate extraction from real dialogs and kb document sampling`.

### Task 11: Поиск кандидатов-кусков на сервере для сборки эталона

**Files:**
- Create: `scripts/server_kb_search.py` (запускается интерпретатором venv прода; импортирует только bm25s, nltk, razdel, numpy, stdlib)
- Test: `tests/test_server_kb_search.py` (крошечный индекс bm25s в фикстуре; тест пропускается, если bm25s не установлен локально — добавить `bm25s`, `nltk`, `razdel` в dev-группу)

**Interfaces:**
- Produces:
  - `python server_kb_search.py --kb <dir kb/chunks/v1> --queries q.jsonl --out packs.jsonl --k 20 --pack-max 40`; вход `{"item_key": str, "queries": [str, ...]}`; выход `{"item_key", "chunks": [{"chunk_id", "doc_id", "url", "title", "date", "text"}]}` — объединение top-k по всем запросам, порядок по лучшему рангу, не больше `pack-max`.
  - Токенизация запроса как в RAG-Core `tokenize_and_normalize_text` (razdel → wordpunct, токены с цифрами как есть, Snowball Russian) — функцию скопировать в скрипт с комментарием-ссылкой на `Meno-Stand-RAG-Core src/meno_rag/stand/tokenization.py`.
  - Соответствие индексов BM25 кускам: индексируемые куски (`indexed=true`) в порядке частей `chunks/part-*.jsonl.gz` (как описано в `kb/chunks/v1/README.md`).

- [ ] **Step 1:** Тест `test_union_and_cap` (2 запроса, пересекающиеся выдачи → без дублей, ≤ pack-max) → FAIL → реализовать → PASS.
- [ ] **Step 2:** Скачать локально ревизию `f374a7ed` `kb/chunks/v1`: `docs.jsonl.gz`, `chunks/part-*.jsonl.gz`, `ragcore/knowledge/bm25/` (HF-токен из сессии) в `work/kb/`; проверить размеры по API дерева.
- [ ] **Step 3:** Перенести `work/kb/` и скрипт на сервер в `~/meno-v2/kb-f374a7ed/` и `~/meno-v2/bench/` (scp через `scratchpad/ssh/config`). Проверка: на сервере `~/Meno-Stand/Meno-Stand-RAG-Core/.venv/bin/python ~/meno-v2/bench/server_kb_search.py --kb ... --queries <файл с 2 запросами>` печатает пачки с непустыми `text`.
- [ ] **Step 4:** Commit `feat: server-side bm25 candidate packs for reference building`.

### Task 12: Сборка серебряного набора

Операционная задача: результат — данные, а не код. LLM-работа — агенты Claude этой сессии (инструмент Agent; Workflow — только если владелец явно согласится); всё LLM-взаимодействие с локальной машины.

**Files:**
- Create: `prompts/gold_questions.md`, `prompts/gold_queries.md`, `prompts/gold_facts.md`, `prompts/gold_verify.md`, `scripts/assemble_items.py`, `data/v2/items.jsonl`, `data/v2/README.md`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 10, 11.
- Produces: `data/v2/items.jsonl` — валидный по `validate_item` и `verify_quotes`; `scripts/assemble_items.py --drafts work/drafts/*.jsonl --kb work/kb --out data/v2/items.jsonl` (отбрасывает примеры с проблемами и печатает отчёт по причинам и категориям).

Квоты новых примеров (сверх 100 существующих, которые переводятся в v2 с фактами):

| категория | число | примечание |
|---|---|---|
| factual | 150 | ≥ 90 по темам `schedule`, `cii_labs`, `people`, `phd`, `masters` (≥ 15 каждой) |
| complex | 60 | ответ из ≥ 2 документов; ≥ 20 — цепочки |
| multi_turn | 80 | 2–8 ходов; ≥ 30 — отсылки к содержимому прошлого ответа («давай её») |
| long_dialog | 15 | история длиннее 8 ходов, вопрос про ранний ход |
| meta_dialog | 20 | «что я спрашивал в начале», факты — `source="history"` |
| unanswerable | 40 | правдоподобные вопросы про НГУ без ответа в базе |
| time_sensitive | 30 | с `as_of`; сроки, приказы, расписание |
| general | 30 | общие вопросы не про НГУ; `expect_sources=False`, если в базе ничего |
| chitchat | 20 | включая «привет + вопрос» → это `factual`, не chitchat |
| off_topic | 20 | игры, «сосчитай до 100000», оскорбления |
| prompt_injection | 15 | просьбы показать промпт и служебные блоки |

- [ ] **Step 1:** Тест `test_assemble_drops_bad_quotes_and_reports` (черновик с неверной цитатой → не попал в выход, причина в отчёте) → FAIL → реализовать `assemble_items.py` → PASS.
- [ ] **Step 2:** Собрать кандидатов: `extract_prod_dialogs` по дампу `scratchpad/dialogs_dl/x`, `extract_telegram_questions` по `Meno-Core/questions.ndjson` с everest (Jupyter contents API, только чтение), `sample_docs` по квотам тем. Результат в `work/candidates/`.
- [ ] **Step 3:** Оценка стоимости: посчитать примерный объём токенов этапов 4–6 (пачка ≈ 40 кусков × ~300 токенов) и **сообщить владельцу до запуска**; продолжать после его согласия.
- [ ] **Step 4:** Вопросы: 100 существующих из `data.json` входят как `legacy-NNN` (категория `factual`, тема — по `TOPIC_RULES` к тексту вопроса, `ground_truth` сохраняется) и проходят шаги 5–6 наравне с новыми. Агенты по `prompts/gold_questions.md` выбирают и правят реальные вопросы по квотам и пишут недостающие (по документам — вопрос + исходный документ; многоходовые — фиксированная история, где ответы ассистента написаны правильно по документам базы, а реальные реплики пользователя сохранены дословно). Выход `work/drafts/questions-*.jsonl`.
- [ ] **Step 5:** Поисковые запросы и пачки: агенты по `prompts/gold_queries.md` пишут 3–5 запросов на вопрос → `server_kb_search.py` на сервере → `work/packs/*.jsonl`.
- [ ] **Step 6:** Факты: агенты по `prompts/gold_facts.md` по пачке пишут `key_facts` с дословными цитатами и `chunk_id` либо ставят `answerable=false`; затем независимые агенты по `prompts/gold_verify.md` (без доступа к выводам первого прохода, кроме фактов и цитат) подтверждают каждый факт и однозначность вопроса; спорные примеры отбрасываются.
- [ ] **Step 7:** `uv run python scripts/assemble_items.py ...` → `data/v2/items.jsonl`; `uv run meno-qa-benchmark validate data/v2/items.jsonl` → код 0; фактическое распределение по категориям и темам записать в `data/v2/README.md` (вместе с ревизией базы, датой сборки, пометкой «серебряный», источниками вопросов).
- [ ] **Step 8:** Commit `data: silver benchmark v2 (items, prompts, provenance)`.

### Task 13: Базовая линия текущего прода

**Files:**
- Create: `reports/2026-11-legacy-baseline.md`, `reports/legacy-baseline-metrics.json`, `reports/legacy-baseline-retrieval.json`, `reports/legacy-baseline-speed.json`

- [ ] **Step 1:** Прогон качества: `MENO_BASE_URL=https://meno.nsu.ru/bench/v1`, `MENO_API_KEY` = `BENCH_API_TOKEN` из `.env` прода (прочитать только эту строку, нигде не сохранять), `MENO_MODEL=meno-lite`, `MENO_CONCURRENCY=4`; `uv run meno-qa-benchmark run --items data/v2/items.jsonl --out results/legacy.records.jsonl`. Проверка: доля `ok` ≥ 0.95, иначе разобрать ошибки до продолжения.
- [ ] **Step 2:** Судья: `judge-export` → агенты Claude выносят вердикты по `prompts/judge.md` пачками → `judge-import`; доля `judge_error` ≤ 0.03.
- [ ] **Step 3:** `metrics ... --system-prompt data/v2/system_prompt_legacy.txt --out reports/legacy-baseline-metrics.json`; `retrieval-eval` по `from_run_records(results/legacy.records.jsonl)` → `reports/legacy-baseline-retrieval.json`.
- [ ] **Step 4:** Скорость — вечером: скопировать `speed.py` и `items.jsonl` на сервер, `BENCH_TOKEN` из `.env` прода через переменную окружения, `python3 speed.py --url http://127.0.0.1:9006/v1/chat/completions --concurrency 1,8,16 --n 40 --categories factual,complex,multi_turn --model meno-lite --out speed.json`; забрать в `reports/legacy-baseline-speed.json`.
- [ ] **Step 5:** `reports/2026-11-legacy-baseline.md`: таблица метрик overall и по категориям, поиск, скорость, по 3 характерных провала на категорию (вопрос, ответ, что не так), сравнение с цифрами дампа из спецификации (раздел 3.3).
- [ ] **Step 6:** Commit `reports: legacy production baseline on bench v2`.
- [ ] **Step 7:** Пуш ветки `feat/bench-v2` — попросить у владельца право на запись (deploy key с записью или пуш с его стороны); не пушить без явного разрешения.
