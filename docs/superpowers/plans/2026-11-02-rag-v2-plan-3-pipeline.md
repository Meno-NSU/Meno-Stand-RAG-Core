# Менон RAG v2 — План 3: пайплайн ответа v2 рядом со старым

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Второй пайплайн ответа (база `nsu-kb-2026-11`, «База НГУ (ноябрь 2026) · новый поиск») в том же бэкенде Meno-Stand-RAG-Core: анализ реплики одним вызовом LLM по JSON-схеме, гибридный поиск по релизу `f374a7ed` с кросс-энкодером и порогом, многокруговой путь для сложных вопросов, трёхслойная память диалога, сборка контекста с датами, генерация, источники после ответа; старый пайплайн не меняется.

**Architecture:** Новые модули в `src/meno_rag/v2/` поверх ядра поиска Плана 2 (`kb`, `lexical`, `embed`, `dense`, `fusion`, `rerank`). `V2Pipeline` повторяет интерфейс `StandRagPipeline` (`prepare` → `PipelineOutcome`, `stream_text`, `generate_text`) и добавляет `finalize_sources(outcome, answer)`. Вся работа на GPU (эмбеддинг запроса, dense, rerank) идёт в одном выделенном потоке, чтобы параллельные запросы не дрались за карту. `api/main.py` выбирает пайплайн по `knowledge_base_id`; для v2 событие `sources` уходит после текста ответа.

**Tech Stack:** Python 3.12, FastAPI, pydantic-settings, httpx, torch 2.11, transformers 5.8, bm25s, razdel (как в venv прода); pytest (+ `fastapi.testclient`).

**Spec:** `docs/superpowers/specs/2026-11-02-rag-v2-design.md` (разделы 5, 6, 8, 11). Ядро поиска — План 2; инструмент оценки — План 1 (`/home/devuser/Meno-QA-Benchmark`).

## Global Constraints

- Репозиторий `/home/devuser/Meno-RAG-Core`, ветка `feat/rag-v2`. Код `src/meno_rag/stand/` не меняется; поведение старого пайплайна (`nsu-stand-faiss-bm25`) и его SSE-порядок (источники до генерации) не меняются.
- Id и подписи баз (спек 11): `nsu-stand-faiss-bm25` — «База НГУ (июль 2026) · прежний поиск»; `nsu-kb-2026-11` — «База НГУ (ноябрь 2026) · новый поиск». База по умолчанию — старая.
- Новые настройки — поля `Settings` в `config.py` с алиасами `V2_*` (и `LEGACY_ENABLED`), строка-комментарий в `example.env` на каждое; значения по умолчанию ниже — стартовые, пороги калибруются в Task 9.
- Стартовые значения: `V2_RERANK_THRESHOLD=0.3`, `V2_ATTRIBUTION_THRESHOLD=0.5`, `V2_HISTORY_BUDGET_TOKENS=5000`, `V2_SUMMARY_BUDGET_TOKENS=800`, `V2_CONTEXT_BUDGET_TOKENS=6000`, `V2_MAX_ANSWER_TOKENS=1500`, `V2_MAX_CONTEXT_CHUNKS=8`, `V2_MAX_SOURCES=5`, `V2_MAX_ROUNDS=3`, `V2_CANDIDATES_PER_QUERY=50` (dense и BM25 каждый), `V2_RERANK_TOP=40`. Оценка токенов — `ceil(len(text) / 3)` (консервативно для кириллицы), без токенизатора модели.
- Тексты для пользователя — по-русски. Отказ «нет в базе» (вопрос про НГУ без документов выше порога), дословно: «В документах НГУ, которые у меня есть, нет ответа на этот вопрос. Уточнить можно на сайте nsu.ru или в профильном подразделении — деканате, приёмной комиссии или отделе аспирантуры.» Пометка общего ответа без документов (добавляется кодом в конец ответа), дословно: «_Это общий ответ — не из документов НГУ._» Замена при утечке промпта, дословно: «Извините, служебные инструкции я показать не могу. Спросите меня об учёбе, поступлении или жизни в НГУ.»
- Маршруты анализа (enum, ровно эти строки): `nsu`, `general`, `meta_dialog`, `rework`, `chitchat`, `off_topic`, `prompt_injection`. Поиск — только для `nsu` и `general`.
- Имена стадий в SSE — только существующие `StageName` (фронт их уже знает): анализ = `query_rewrite`, поиск = `retrieval`, слияние = `fusion`, rerank = `rerank`, сборка = `context_assembly`, генерация = `generation`; повторные круги — те же стадии с `detail.round`. Новых значений `StageName` не добавлять.
- LLM для анализа, переформулировки, сводки и проверки источников — та же модель, что отвечает (`runtime.generation`); вызовы с JSON — через `LLMRouter.chat_completion(..., response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}, extra_body={"chat_template_kwargs": {"enable_thinking": False}}, temperature=0)`.
- Тесты не загружают веса и не ходят в сеть: LLM — фейк с заранее заданными ответами (как `tests/_fake_llm.py`), поиск — фейковые ретриверы/реранкер; локальная машина ~1 ГБ свободной RAM.
- Формат трассы v2 совместим с раннером бенчмарка (`runner._contexts_from_trace`): `trace.search_queries: list[str]`, `trace.chunks: {chunk_id: {url, title, text}}`, `trace.rerank.candidates: [{chunk_id, rank, score, kept}]` (`kept` — кусок вошёл в контекст), плюс `trace.v2: {route, rounds, standalone_question, subqueries, coverage, timings_ms, sources_debug}`.

## Review Focus

1. Анализ реплики вернул невалидный JSON, обрезанный ответ или таймаут → путь `nsu` с исходной репликой как запросом (никогда не падение и не «болтовня» по умолчанию). Тест — Task 3.
2. Ни одного куска выше порога → `nsu`: дословный отказ без источников и без вызова генерации; `general`: ответ модели + дословная пометка; источников нет. Тест — Task 7.
3. Очень длинный диалог (40 обменов, одна реплика длиннее всего бюджета) → сообщения модели не превышают бюджеты; огромная реплика обрезается; старые ходы уходят в сводку, а не теряются. Тест — Task 4.
4. 16 одновременных запросов → GPU-работа выполняется по одной, без взаимоблокировки; исключение в одном запросе не ломает поток для следующих. Тест — Task 2.
5. Ответ модели дословно повторяет кусок системного промпта (8 слов подряд) → поток обрывается и заменяется дословной фразой. Тест — Task 5.

---

### Task 1: Настройки v2 и реестр баз

**Files:** Modify `src/meno_rag/config.py`, `example.env`; Create `src/meno_rag/api/knowledge_bases.py`; Test `tests/v2/test_settings_registry.py`.

**Interfaces:**
- Produces: поля `Settings`: `legacy_enabled: bool = True` (`LEGACY_ENABLED`), `v2_enabled: bool = False` (`V2_ENABLED`), `v2_kb_dir: str` (`V2_KB_DIR`, каталог `kb/chunks/v1`), `v2_embedder: str = "frida"`, `v2_embedder_path: str`, `v2_matrix_dir: str`, `v2_reranker: str = "bge"`, `v2_reranker_path: str`, `v2_device: str = "cuda"`, `v2_gpu_mem_fraction: float = 0.3`, `v2_embed_dtype: str = "bf16"`, `v2_sources_llm_check: bool = False` и все числа из Global Constraints (алиасы `V2_<ИМЯ>` в верхнем регистре).
- `api/knowledge_bases.py`: `LEGACY_KB_ID = "nsu-stand-faiss-bm25"`, `V2_KB_ID = "nsu-kb-2026-11"`; `@dataclass(frozen=True) KnowledgeBase(id, name, description, rag_engine_id)`; `knowledge_bases(settings) -> list[KnowledgeBase]` (старая — если `legacy_enabled`, новая — если `v2_enabled`; порядок: старая, новая); `default_kb_id(settings) -> str` (старая, если включена, иначе новая).
- `v2/embed.py`: `load_embedder(name, path, device, dtype: str | None = None)` — `"bf16"`/`"fp16"` на CUDA (Ruling P2-R4), fp32 на CPU.

- [ ] Тесты: значения по умолчанию и чтение из env (`V2_RERANK_THRESHOLD=0.42` → 0.42); реестр при четырёх комбинациях флагов; подписи дословно из Global Constraints; `dtype="bf16"` даёт bf16-модель на фейковом пути. → FAIL → реализовать → PASS → commit `feat(v2): settings and knowledge-base registry`.

### Task 2: Сервис поиска

**Files:** Create `src/meno_rag/v2/search.py`; Test `tests/v2/test_search.py`.

**Interfaces:**
- Consumes: План 2 (`KbRelease`, `Bm25Retriever`, `DenseRetriever`, `rrf`, `Reranker`).
- Produces: `@dataclass Candidate(pos: int, chunk: Chunk, score: float)` (оценка кросс-энкодера ∈ [0,1]); `@dataclass SearchResult(candidates: list[Candidate]` — объединение по всем запросам, по убыванию лучшей оценки, без повторов куска; `coverage: dict[str, float]` — лучшая оценка по каждому подзапросу; `timings_ms: dict[str, float])`; `class SearchService(release, bm25, dense, reranker, settings)`: `async search(queries: list[str], rerank_queries: list[str]) -> SearchResult`. Для каждого запроса — dense и BM25 по `V2_CANDIDATES_PER_QUERY`, RRF; объединение топ-`V2_RERANK_TOP` каждого; rerank одним батчем пар (rerank-запрос, `chunk.index_text`) — каждый кандидат оценивается против `rerank_queries[i]` того запроса, который его нашёл (если нашли несколько — берётся максимум); `coverage[q]` = максимум по кандидатам запроса `q`. Пустые/пробельные запросы пропускаются. Вся работа на GPU и BM25 — в `ThreadPoolExecutor(max_workers=1)` через `loop.run_in_executor`.
- `load_search_service(settings) -> SearchService` — загрузка релиза, BM25, матрицы (`expected_n=len(release.chunks)`), эмбеддера и реранкера с `set_per_process_memory_fraction`.

- [ ] Тесты (фейки): объединение без повторов и порядок по оценке; `coverage` по каждому подзапросу; пустой запрос пропущен; 16 параллельных `search` → все завершились, фейковый GPU-счётчик ни разу не видел двух одновременных вызовов (Review Focus 4); исключение в одном вызове не мешает следующему. → FAIL → реализовать → PASS → commit `feat(v2): hybrid search service with serialized GPU work`.

### Task 3: Анализ реплики

**Files:** Create `src/meno_rag/v2/analysis.py`; Test `tests/v2/test_analysis.py`.

**Interfaces:**
- Produces: `@dataclass TurnAnalysis(route: str, standalone_question: str, subqueries: list[str], chain: bool, focus: str, fallback: bool)`; `ANALYSIS_SCHEMA: dict` (JSON Schema: `route` enum из Global Constraints, `standalone_question` string, `subqueries` array of string maxItems 4, `chain` boolean, `focus` string; все обязательны); `async analyze(llm_router, runtime, *, question: str, recent: list[dict], summary: str, today: date, timeout_s: float) -> TurnAnalysis`.
- Промпт (система, по-русски): роль «анализатор реплик ассистента НГУ»; определения маршрутов с примерами из спека 5.2 («Привет! Как поступить на ФИТ?» → `nsu`; «а ещё?» после темы → продолжение темы, не болтовня); правило «при сомнении — `nsu`»; самодостаточный вопрос раскрывает местоимения и сокращения по истории и фокусу; подзапросы — только для сравнения, нескольких аспектов, перечисления по условию, цепочки; `chain` — если второй шаг зависит от первого; `focus` — факультет/программа/группа/человек из диалога; сегодняшняя дата. В сообщение пользователя — сводка (если есть), последние 3 обмена (ответы ассистента обрезаны до 600 символов) и текущая реплика.
- Пост-обработка (детерминированная): `standalone_question` пустой → исходная реплика; `subqueries` без пустых и дублей; `route ∈ {chitchat, off_topic}` принимается, только если реплика ≤ 12 слов и не содержит основ из списка `NSU_STEMS = ("нгу", "университет", "факультет", "институт", "кафедр", "лаборатор", "общежит", "стипенд", "поступ", "экзамен", "сесси", "расписан", "аспирант", "магистр", "бакалавр", "академгород", "деканат", "ректор", "преподава", "студент", "курс", "ции")` — иначе `nsu`.
- Любая ошибка (исключение клиента, таймаут, невалидный JSON, не тот enum) → `TurnAnalysis(route="nsu", standalone_question=question, subqueries=[], chain=False, focus="", fallback=True)`.

- [ ] Тесты (фейковый LLM): разбор валидного ответа; `response_format` со схемой и `enable_thinking: False` реально переданы; невалидный JSON / таймаут / неизвестный route → fallback (Review Focus 1); «Привет! Как поступить на ФИТ?» с ответом модели `chitchat` → `nsu` по стоп-основам; короткое «спасибо!» с `chitchat` → `chitchat`. → FAIL → реализовать → PASS → commit `feat(v2): turn analysis with guided JSON and routing guards`.

### Task 4: Память диалога

**Files:** Create `src/meno_rag/v2/memory.py`; Test `tests/v2/test_memory.py`.

**Interfaces:**
- Produces: `estimate_tokens(text) -> int`; `@dataclass DialogueMemory(recent: list[dict]` — последние обмены целиком в хронологическом порядке, `older: list[dict]` — всё, что старше, `summary: str`, `recalled: list[dict]` — дословно поднятые старые обмены`)`; `split_history(history: list[dict], budget_tokens: int) -> tuple[list[dict], list[dict]]` — с конца берутся целые обмены (user+assistant), пока помещаются; самый свежий обмен берётся всегда, но каждая его реплика обрезается до `budget_tokens // 2` с пометкой «[…]»; `class SummaryStore` — LRU на 2048 записей, ключ `sha256` от `older` (JSON); `async summarize(llm_router, runtime, older, previous_summary, budget_tokens) -> str` — промпт: кто пользователь (если сказал), что спрашивал, какие факты и ссылки уже даны, что не решено, фокус; ответ не длиннее бюджета (иначе обрезка по предложениям); `recall_old_turns(older, query, k=2) -> list[dict]` — обмены из `older` с наибольшим пересечением основ слов (`stand.tokenization.tokenize_and_normalize_text` + `SnowballStemmer("russian")`), не меньше 2 общих основ; `async build_memory(history, query, llm_router, runtime, store, settings) -> DialogueMemory` — сводка берётся из `store`, при промахе считается сразу; `async precompute_next(history_with_answer, ...)` — фоновой задачей считает сводку, которая понадобится следующему ходу (ошибки только в лог).

- [ ] Тесты: 40 обменов → `recent` в бюджете, `older` непуст, сводка запрошена один раз, второй вызов с тем же `older` — из кэша; одна реплика длиннее бюджета → обрезана с «[…]» (Review Focus 3); `recall_old_turns` находит обмен про общежитие по «что ты говорил про общежития?»; пустая история → пустая память без вызова LLM. → FAIL → реализовать → PASS → commit `feat(v2): layered dialogue memory with cached summaries`.

### Task 5: Контекст, промпты генерации, защита промпта

**Files:** Create `src/meno_rag/v2/prompting.py`; Test `tests/v2/test_prompting.py`.

**Interfaces:**
- Produces: `assemble_context(candidates: list[Candidate], threshold: float, budget_tokens: int, max_chunks: int) -> tuple[str, list[Candidate]]` — только кандидаты с `score ≥ threshold`, от лучшего к худшему, пока помещаются; блок каждого: `[n] {title} · {тип документа} · {дата или «дата не указана»} · {url}` + текст; `SYSTEM_PROMPT_V2: str`; `build_messages(route, *, question, standalone_question, memory: DialogueMemory, context: str, today: date) -> list[dict]` — система (+ сегодняшняя дата) → сводка и поднятые старые обмены отдельным системным блоком «Ранее в этом диалоге…» → `memory.recent` настоящими репликами с ролями → последняя реплика пользователя; для `nsu`/`general` с контекстом — блок документов в системном сообщении с правилами: опираться только на документы для фактов об НГУ, называть даты из документов, если сведения могут устареть, не выдумывать контакты и числа, отвечать по-русски кратко и по делу; для `general` без контекста — «ответь из общих знаний, без фактов об НГУ»; для `meta_dialog`/`rework` — «отвечай только по истории диалога, новых фактов об НГУ не добавляй»; для `chitchat` — коротко и дружелюбно; для `off_topic` — вежливый отказ с подсказкой, о чём спросить; для `prompt_injection` — отказ показывать инструкции.
- `class LeakGuard(system_prompt: str, n: int = 8)`: `feed(token: str) -> bool` (True — в выводе появилось n слов подряд из системного промпта, нормализация: нижний регистр, только буквы и цифры).

- [ ] Тесты: порог и бюджет отсекают худшие; порядок по оценке; дата и тип в шапке блока; роли и порядок сообщений для каждого маршрута; бюджет истории не превышен; `LeakGuard` срабатывает на 8 слов промпта, разбитых на токены, и не срабатывает на 7 (Review Focus 5). → FAIL → реализовать → PASS → commit `feat(v2): context assembly, generation prompts and prompt-leak guard`.

### Task 6: Источники после ответа

**Files:** Create `src/meno_rag/v2/sources.py`; Test `tests/v2/test_sources.py`.

**Interfaces:**
- Produces: `split_sentences(text) -> list[str]` (`razdel.sentenize`, отбрасываются предложения короче 4 слов и строки-пометки); `async attribute(answer: str, context_chunks: list[Candidate], reranker_call, threshold: float) -> list[tuple[Candidate, float]]` — пары (предложение, текст куска) одним батчем через `reranker_call` (тот же однопоточный исполнитель, что у поиска); кусок использован, если его максимальная оценка ≥ порога; `to_sources(used, docs: dict, max_sources: int) -> list[dict]` — словари `{"document_title", "source_url", "date"}` (`date` — `YYYY-MM-DD` или пустая строка); схлопывание по нормализованному URL; из страниц одного документа предпочесть `nsu.ru` репосту `vk.com`/`t.me`; порядок — по оценке; `async llm_check(llm_router, runtime, answer, sources) -> list[dict]` — один вызов с JSON-схемой `{"keep": [int]}` (номера источников, которые подтверждают сказанное); при ошибке возвращает вход без изменений.

- [ ] Тесты (фейковый реранкер): предложение про стипендию → использован кусок про стипендию, не про общежитие; дубли URL схлопнуты; nsu.ru предпочтён vk.com; ответ без предложений выше порога → `[]`; `llm_check` оставляет указанные номера, на ошибке — вход. → FAIL → реализовать → PASS → commit `feat(v2): post-hoc source attribution with dates`.

### Task 7: `V2Pipeline`

**Files:** Create `src/meno_rag/v2/pipeline.py`; Test `tests/v2/test_pipeline.py`.

**Interfaces:**
- Consumes: Tasks 2–6; `meno_rag.schemas.PipelineOutcome`, `ChatMessage`; `stand.pipeline.PipelineRuntime`, `StageSink`, `extract_question_and_history`; `api.events.StageEvent`, `StageName`, `StageStatus`.
- Produces: `class V2Outcome(PipelineOutcome)` с полями `route: str`, `context_chunks: list[Any]`, `general_note: bool`, `refusal: bool`; `class V2Pipeline(search, llm_router, settings, summary_store, docs)`:
  - `async prepare(*, messages, runtime, stage_sink=None, capture_trace=False) -> V2Outcome`: память → анализ (стадия `query_rewrite`) → для `nsu`/`general`: круг поиска (стадии `retrieval`, `rerank`; запросы = `[standalone] + subqueries`, rerank-запросы — те же) → проверка достаточности: подзапрос покрыт, если `coverage ≥ V2_RERANK_THRESHOLD`; для простого вопроса — лучший кусок ≥ порога → непокрытые подзапросы (или `chain`) и круг < `V2_MAX_ROUNDS` → короткий вызов LLM со схемой `{"queries": [str]}` (≤ 3, по вопросу, непокрытым подзапросам и заголовкам найденного) → ещё круг (`detail.round`) → `assemble_context` (стадия `context_assembly`) → `build_messages`. `nsu` без контекста → `refusal=True` (генерация не вызывается); `general` без контекста → `general_note=True`. Трасса — формат из Global Constraints.
  - `async stream_text(*, outcome, runtime, max_tokens=None, temperature=None)`: `refusal` → отдаёт дословный отказ одним куском; иначе поток модели (`max_tokens` ≤ `V2_MAX_ANSWER_TOKENS`) через `LeakGuard` (срабатывание → поток прерывается, отдаётся дословная замена); в конце при `general_note` — `"\n\n" +` дословная пометка; после завершения — `precompute_next` фоновой задачей.
  - `async generate_text(...) -> str` — то же без потока.
  - `async finalize_sources(outcome, answer: str) -> list[dict]` — `[]` для маршрутов без поиска, при `refusal`, `general_note` и если сработал `LeakGuard`; иначе `attribute` → `to_sources` (+ `llm_check`, если `V2_SOURCES_LLM_CHECK`); пишет `outcome.sources` и `trace.v2.sources_debug`.

- [ ] Тесты (фейковые поиск и LLM, по одному на путь): `chitchat` — без поиска и без источников; `nsu` с покрытием с первого круга — один круг; непокрытый подзапрос → второй круг с новыми запросами, `detail.round == 2`; ничего выше порога для `nsu` → дословный отказ, генерация не вызвана, источников нет (Review Focus 2); `general` без контекста → дословная пометка в конце, источников нет; трасса читается `runner._contexts_from_trace` (скопировать функцию в тест); события стадий только из разрешённых имён. → FAIL → реализовать → PASS → commit `feat(v2): answer pipeline with multi-round search`.

### Task 8: Подключение к API

**Files:** Modify `src/meno_rag/api/main.py`; Test `tests/v2/test_api_v2.py`.

**Interfaces:**
- Consumes: Task 1 (`knowledge_bases`, `default_kb_id`, id), Task 7 (`V2Pipeline`, `load` через `load_search_service`).
- Produces:
  - lifespan: старые ресурсы грузятся только при `LEGACY_ENABLED`; при `V2_ENABLED` — `load_search_service` в `asyncio.to_thread`, затем `app.state.v2_pipeline`; ошибка загрузки → лог `v2_resources_load_failed` и `None` (как у старого).
  - `GET /v1/knowledge-bases`: записи из `knowledge_bases(settings)` с `available` по своему пайплайну; `default_selection.knowledge_base_id = default_kb_id(settings)`; `/healthz` — список id и доступность.
  - `POST /v1/chat/completions`: `knowledge_base_id` (пусто → `default_kb_id`) выбирает пайплайн; неизвестный id → прежняя 400; выбранный пайплайн `None` → 503 (проверка `app.state.pipeline` в начале маршрута заменяется проверкой выбранного). В `_stream_response` / `_non_stream_response` вместо `request.app.state.pipeline` — выбранный объект; если у него есть `finalize_sources`: события `sources` до генерации нет, после последнего токена — `sources = await pipeline.finalize_sources(outcome, answer)` и `event: sources` (если не пусто) до чанка `finish_reason`; в JSON-ответе — `sources` после генерации. Для старого пайплайна порядок событий не меняется (тест-снимок).
  - Сохранение (`_persist_success`, `_persist_failure`, pipeline_run): `knowledge_base_id` — выбранный id, не `KB_ID`.
- [ ] Тесты (`TestClient`, фейковые пайплайны в `app.state`): список баз при флагах; выбор v2 по id и по умолчанию; 503 при недоступной выбранной базе, когда другая доступна; порядок SSE у v2: стадии → роль → токены → `sources` → finish → summary → `[DONE]`; у старого — прежний порядок; bench-режим отдаёт `pipeline_trace` v2; в БД (`with_db`) записан id v2. Весь `tests/` зелёный. → FAIL → реализовать → PASS → commit `feat(api): serve the v2 pipeline as a separate knowledge base`.

### Task 9: Развёртывание и сравнение на сервере (операционная, контролёр)

- [ ] Отдельный экземпляр бэкенда в `~/meno-v2/rag-core` (код `git archive`), свой `.env` (порт 9106 на 127.0.0.1, `LEGACY_ENABLED=false`, `V2_ENABLED=true`, sqlite-БД в `~/meno-v2/var`, те же `VLLM_ENDPOINTS` и `BENCH_API_TOKEN`), запуск `nohup` из `~/meno-v2/venv`; прод не трогать.
- [ ] Калибровка порогов на выдаче CLI Плана 2 (`hybrid:<emb>+rerank:<rr>`) по серебряному набору: `V2_RERANK_THRESHOLD` — максимум F1 «есть золотой кусок выше порога» для отвечаемых против «ничего выше порога» для `answerable=false`; `V2_ATTRIBUTION_THRESHOLD` — по точности источников на 50 ответах.
- [ ] Прогоны раннера бенчмарка через SSH-туннель к 9106: v2 × {`meno-lite`, `gemma-4-12B-it`}; старый прод × `gemma-4-12B-it` через `/bench/v1` (meno-lite уже есть); `speed.py` на сервере для v2 при 1/8/16; судья — Claude (export/import), метрики, сравнение `compare`; источники — точность/полнота/лишние; отчёт `reports/2026-11-v2-vs-legacy.md` в бенчмарке.
