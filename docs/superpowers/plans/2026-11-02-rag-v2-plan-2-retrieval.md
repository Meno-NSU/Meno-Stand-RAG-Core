# Менон RAG v2 — План 2: ядро поиска v2 и сравнение компонентов

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Модули поиска v2 в Meno-Stand-RAG-Core (загрузка релиза базы, BM25, эмбеддеры, плотный поиск, RRF, кросс-энкодеры) и CLI, выдающий ранжированную выдачу в формате бенчмарка, чтобы сравнить эмбеддеры и реранкеры на быстром тесте поиска и замерить их задержку на A6000.

**Architecture:** Новый пакет `src/meno_rag/v2/` рядом с замороженным `stand/`. Всё работает по релизу `kb/chunks/v1` (ревизия `f374a7ed`), загруженному в память один раз. BM25 — готовый индекс bm25s из релиза; плотный поиск — точное скалярное произведение по fp16-матрице на GPU (для 477 тыс. векторов этого достаточно; движок Qdrant/zvec сравнивается отдельно в Плане 3, когда понадобятся фильтры и обновления). Кандидаты dense и BM25 сливаются RRF и переранжируются кросс-энкодером. CLI `meno-rag-v2-retrieve` читает `items.jsonl` бенчмарка и пишет выдачу в формате `{"item_id","ranked":[{"doc_id","url","chunk_id","score"}]}` (контракт Плана 1, Task 8).

**Tech Stack:** Python 3.12, torch 2.11, transformers 5.8, bm25s, numpy, httpx (как в venv прода); pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-rag-v2-design.md` (разделы 5.2, 7, 9). Инструмент оценки — План 1 (`/home/devuser/Meno-QA-Benchmark`).

## Global Constraints

- Репозиторий `/home/devuser/Meno-RAG-Core`, ветка `feat/rag-v2` от `main` (`a4977e9`). Код `src/meno_rag/stand/` не меняется (замороженный старый пайплайн).
- Сервер: все сетевые запросы и LLM — с локальной машины; на сервере только запуск. Отдельный venv `~/meno-v2/venv` (копия venv прода, venv прода не трогать), код запускается с `PYTHONPATH=src`. GPU — только `CUDA_VISIBLE_DEVICES=0` и не больше ~16 ГБ (`torch.cuda.set_per_process_memory_fraction`), прод (`meno-lite`, 9020) и сервисы `rsolikhov` не останавливать.
- Релиз базы: `~/meno-v2/kb-f374a7ed/kb/chunks/v1` (сервер), `/home/devuser/meno-work/kb-f374a7ed/kb/chunks/v1` (локально): 104 767 документов, 520 414 кусков, 476 869 индексируемых; порядок BM25 = индексируемые куски в порядке частей.
- Модели на сервере: FRIDA — кэш HF `~/.cache/huggingface/hub/models--ai-forever--FRIDA`; `~/meno-v2/models/{Qwen3-Embedding-0.6B,Qwen3-Reranker-0.6B,bge-reranker-v2-m3}`.
- Индексируемый текст куска для эмбеддингов — `index_text` (шапка + текст), как для BM25.
- FRIDA: `"search_query: "` / `"search_document: "`, CLS-пулинг, L2-норма, 512 токенов. Qwen3-Embedding: запрос `"Instruct: Given a question about Novosibirsk State University, retrieve passages that answer it\nQuery: " + q`, документ без префикса, last-token пулинг, L2-норма, 512 токенов. bge-reranker-v2-m3: `AutoModelForSequenceClassification`, пара (query, text), логит → sigmoid. Qwen3-Reranker: официальный формат (system «Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".»), оценка = softmax(logit_yes, logit_no)[yes].

## Review Focus

1. Индекс BM25 и порядок кусков разошлись (другой релиз) — загрузка падает с понятной ошибкой, а не тихо возвращает чужие куски.
2. Пустой запрос / запрос из одних стоп-символов — пустая выдача без исключений.
3. Кандидат без текста или очень длинный (> 512 токенов) — реранкер обрезает и не падает.
4. Нехватка памяти GPU на большом батче — батч уменьшается вдвое и повторяется, а не валит прогон.
5. Повтор одного куска в dense и BM25 — RRF считает его один раз с суммой вкладов.

---

### Task 1: Загрузка релиза базы

**Files:** Create `src/meno_rag/v2/__init__.py`, `src/meno_rag/v2/kb.py`; Test `tests/v2/test_kb.py` (крошечный релиз в `tmp_path`).

**Interfaces:**
- Produces: `@dataclass(frozen=True) Chunk(chunk_id: str, doc_id: str, url: str, title: str, date: str | None, doc_type: str | None, tier: str | None, text: str, index_text: str)`; `class KbRelease` с `chunks: list[Chunk]` (только индексируемые, в порядке BM25), `by_id: dict[str, int]`, `docs: dict[str, dict]` (из `docs.jsonl.gz`: title, urls, tier, doc_type); `load_release(root: Path) -> KbRelease` — сверяет число индексируемых кусков с `ragcore/knowledge/bm25/params.index.json:num_docs`, иначе `ValueError` с обоими числами.

- [ ] Тест: релиз из 3 кусков (один `indexed=false`) → `len(chunks)==2`, порядок по частям; несовпадение `num_docs` → `ValueError`. → FAIL → реализовать (потоковое чтение gzip) → PASS → commit `feat(v2): load pinned kb release`.

### Task 2: BM25

**Files:** Create `src/meno_rag/v2/lexical.py`; Test `tests/v2/test_lexical.py`.

**Interfaces:**
- Consumes: `KbRelease`; `meno_rag.stand.tokenization.tokenize_and_normalize_text` + `SnowballStemmer("russian")`.
- Produces: `class Bm25Retriever(release: KbRelease, index_dir: Path)`; `.search(query: str, k: int) -> list[tuple[int, float]]` (позиция куска в `release.chunks`, оценка > 0), пустой запрос → `[]`.

- [ ] Тест на крошечном индексе bm25s (построить в фикстуре той же нормализацией): правильный первый кусок; пустой запрос → `[]`. → FAIL → реализовать → PASS → commit.

### Task 3: Эмбеддеры

**Files:** Create `src/meno_rag/v2/embed.py`; Test `tests/v2/test_embed.py` (модели подменяются крошечным случайным трансформером из `transformers` конфигом, без загрузки весов).

**Interfaces:**
- Produces: `class Embedder(Protocol)`: `name: str`, `dim: int`, `encode_queries(texts: list[str]) -> torch.Tensor` и `encode_docs(texts: list[str], batch_size: int = 128) -> torch.Tensor` (fp16, L2-нормированные, на устройстве модели); реализации `FridaEmbedder(path, device)`, `Qwen3Embedder(path, device)` с префиксами и пулингом из Global Constraints; `load_embedder(name: str, path: str, device: str) -> Embedder` (`name ∈ {"frida","qwen3"}`); при `torch.cuda.OutOfMemoryError` батч делится пополам (Review Focus 4).

- [ ] Тесты: размерность и норма = 1; префиксы реально подставляются (проверка через перехват токенизатора); OOM-деление батча (подмена forward, кидающего OOM при batch > 2). → FAIL → реализовать → PASS → commit.

### Task 4: Плотный индекс

**Files:** Create `src/meno_rag/v2/dense.py`; Test `tests/v2/test_dense.py`.

**Interfaces:**
- Produces: `build_matrix(embedder, release, batch_size) -> torch.Tensor` (N×dim fp16); `save_matrix(m, path)` / `load_matrix(path, device) -> torch.Tensor` (`.npy` fp16 + `meta.json` с именем модели, ревизией релиза, N, dim; несовпадение N с релизом → `ValueError`); `class DenseRetriever(matrix, embedder)`, `.search(query: str, k: int) -> list[tuple[int, float]]` — точный top-k скалярного произведения.

- [ ] Тесты на случайной матрице: top-1 = сам вектор; сохранение/загрузка; несовпадение N → ошибка. → FAIL → реализовать → PASS → commit.

### Task 5: RRF и реранкеры

**Files:** Create `src/meno_rag/v2/fusion.py`, `src/meno_rag/v2/rerank.py`; Test `tests/v2/test_fusion.py`, `tests/v2/test_rerank.py`.

**Interfaces:**
- Produces: `rrf(lists: Sequence[Sequence[tuple[int, float]]], k: int = 60, limit: int | None = None) -> list[tuple[int, float]]` (вклад 1/(k+rank) суммируется, повтор считается один раз в каждом списке); `class Reranker(Protocol)`: `score(query: str, texts: list[str], batch_size: int = 32, max_length: int = 512) -> list[float]` ∈ [0,1]; `BgeReranker(path, device)`, `Qwen3Reranker(path, device)`; `load_reranker(name, path, device)` (`name ∈ {"bge","qwen3"}`).

- [ ] Тесты: RRF на двух списках с общим элементом (сумма вкладов, порядок); длинный текст обрезается без ошибки (крошечная модель); оценки в [0,1]. → FAIL → реализовать → PASS → commit.

### Task 6: CLI выдачи и замер задержки компонентов

**Files:** Create `src/meno_rag/v2/experiments/__init__.py`, `src/meno_rag/v2/experiments/retrieve.py`; Modify `pyproject.toml` (`[project.scripts] meno-rag-v2-retrieve = "meno_rag.v2.experiments.retrieve:main"`); Test `tests/v2/test_retrieve_cli.py` (подмена ретриверов).

**Interfaces:**
- Consumes: Tasks 1–5; формат `items.jsonl` бенчмарка (поля `id`, `question`, `history`, `category`, `answerable`); файл запросов бенчмарка `work/queries/*.jsonl` (`{"item_key","queries"}`) как источник самодостаточного запроса.
- Produces: `meno-rag-v2-retrieve --kb <release> --items <items.jsonl> --queries <queries.jsonl> --config <name> --out <results.jsonl> --timings <timings.json>`; конфиги (строкой): `bm25`, `dense:<emb>`, `hybrid:<emb>` (RRF dense+BM25, по 50), `hybrid:<emb>+rerank:<rr>` (rerank top-40 после RRF); запрос = первый запрос из файла запросов для элемента (самодостаточный; если его нет — сам вопрос); выдача топ-20 в контрактном формате; `timings.json` — p50/p90 по стадиям (embed, dense, bm25, rrf, rerank) и пиковая память GPU.

- [ ] Тест: две фиктивные позиции → строки результата с полями `doc_id`, `url`, `chunk_id`, `score`, порядок по убыванию; `timings.json` с ключами стадий. → FAIL → реализовать → PASS → commit.

### Task 7: Сравнение на сервере (операционная, контролёр)

- [ ] Перенести код (`git archive` ветки) в `~/meno-v2/rag-core`; создать `~/meno-v2/venv` копией venv прода.
- [ ] Построить матрицы FRIDA и Qwen3-Embedding-0.6B по всем 476 869 кускам (записать время и память).
- [ ] Прогнать конфиги `bm25`, `dense:frida`, `dense:qwen3`, `hybrid:frida`, `hybrid:qwen3`, `hybrid:<лучший>+rerank:bge`, `hybrid:<лучший>+rerank:qwen3` по серебряному набору; посчитать `meno-qa-benchmark retrieval-eval` для каждого и для старого прода (из `RunRecord`).
- [ ] Итоговая таблица (recall@1/5/10/20, MRR, по категориям) + задержки компонентов → `reports/2026-11-retrieval-bakeoff.md` в бенчмарке; выбор эмбеддера и реранкера — по recall@10 после rerank, при равенстве (±0,01) — по задержке.
