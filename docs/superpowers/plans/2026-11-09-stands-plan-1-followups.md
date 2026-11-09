# Stands plan 1: follow-ups

Decisions taken during execution and findings deliberately left for later. Line references are to the commit where each was found.

## Rulings

- Ruling 1: drop StandRegistry.knows_stand from T4 — T6's gate covers every unresolved id with "@", nothing calls it, YAGNI — if wrong, one small method added later.
- Ruling 2: iter_blocks must also cope with CR and LF landing in different network reads (hold a trailing "\r" until the next piece) and T2 adds that to test_iter_blocks_survives_real_servers — this is Review Focus 1 and the plan's test misses that split — if wrong, a few extra lines.
- Ruling 3: str(StandStreamError) starts with the code ("<code>: <message>") — plan tests match on the code; the private text still reaches only ClassifiedError.message — if wrong, a test wording change.
- Ruling 4: tests/stands/_helpers.stand_handler defaults to manifest() (full stand, lab@v1); manifest=None answers 404 (plain) — T6 and T7 tests read as a full stand — if wrong, test helper edit.
- Ruling 5: T5 extracts the failure sequence (stage failed, event error, chunk with error, [DONE]) into one helper used by both _stream_response and _stand_stream_response instead of a third copy — the rubric treats verbatim duplication as a defect and the behaviour stays identical — if wrong, a small refactor is undone.
- Ruling 6: Relay.stage_details keeps, per stage, the last {"lines": [...]} seen — the plan names the field, the spec's stage_details column takes a dict; only used when the trace has no stage_details — if wrong, analysis rows show less detail.
- Ruling 7: implementer template's "run the full suite before committing" is replaced by "run the test files named in the brief, one at a time, check free memory first" — 3.9 GB machine, whole suite once crashed it (Global Constraints) — if wrong, CI or the server run catches what focused runs missed.
- Ruling 8: implementers and reviewers on sonnet, T5 and T6 reviewers on opus because of stream and slot handling, final whole-branch review on fable — turn count beats token price, those two tasks carry the concurrency risk — if wrong, cost moves a little either way.
- Ruling 9: block and answer limits (MAX_BLOCK_BYTES, MAX_TRACE_BLOCK_BYTES, MAX_ANSWER_CHARS) live in meno_contract.models, used by check.py now and relay.py in Task 5 — the spec wants checker and gateway to agree, so one definition — if wrong, moving three constants.
- Ruling 10: on the stand path the client payload carries no stand text: `detail` is omitted, the text stays in the log and _persist_failure; _stream_response unchanged — plan's interface line said "лог и detail" but spec 7.6 (binding) says the log only — if wrong, less detail in devtools for stand errors.
- Ruling 11: classify_stand_error normalizes the code to a known one or stand_error, so a stand cannot create meno_errors series; the raw code goes to the log — spec 8.6 wants bounded series — if wrong, less specific error metrics for stands.
- Ruling 12: the "RAG resources are not initialized" 503 moves below the "@" gate, guarded by "no stand target" — Review Focus 3 wants a clear 400 on a stand-only gateway, the plan's placement note did not foresee it — if wrong, one check reordered back.
- Ruling 13: the 403 for arena:true on a stand without a manifest gets a test in this fix round, though the reviewer filed it as minor — Review Focus 5 names that case explicitly — if wrong, one extra test.
- Ruling 14: participant_for treats the part after "@" as a stand only if it matches the stand id pattern, otherwise the whole id falls to the no-"@" rows; VoteRequest.model_a/b and ArenaSide.model capped at 256 — keeps <stand>@<version> within String(128) on Postgres; manifest versions are already capped at 64 by the contract, stand ids at 32 — if wrong, an odd id is rated under menon-1.0 instead of being refused.
- Ruling 15: Task 10's ruff acceptance is limited to src/meno_rag/api, packages, tests/stands, tests/contract — 91 of 93 findings predate this branch, sit in v2 and tests/v2, and the plan forbids touching src/meno_rag/v2; the owner is told CI lint was already red there — if wrong, CI lint stays red until a separate cleanup.
- Ruling 16: every 4xx from a stand except 429 maps to non-retryable invalid_upstream_request — a wrong api_key must not read as a transient outage — if wrong, a retryable 4xx stops being retried.
- Ruling 17: Llm.id pattern [A-Za-z0-9._:/+@-]{1,128}, at most 32 LLMs per stand, manifest text fields capped, source titles clipped to 500 and urls over 2000 dropped — stands are not trusted to be correct and the shared models close it in one place — if wrong, a stand with an unusual LLM id must rename it or we widen the pattern.
- Ruling 18: checker measures the first token against 120 s, uses 300 s between chunks and 900 s per stream, from shared constants — must agree with the gateway per spec 7.10 — if wrong, the checker waits longer on a hung stand.
- Ruling 19: deferred to a follow-up, not fixed in this wave: the StreamingResponse hardening for slot release on disconnect (both shared and stand slots, same mechanism as before this branch); thinking total cap; RecursionError in _json; role and finish_reason whitelists; stale meno_stand_up after removal; manifest 4xx wiping the last list; refresh outside the lock; non-stream finish_reason — the final reviewer rated them follow-ups and each is contained — if wrong, an operational annoyance, the worst being a slot not freed until garbage collection after a disconnect.
- Ruling 20 (исправление от 2026-11-09): встроенные пайплайны сохраняют свои id баз знаний и строки рейтинга, пока фронтенд не научится скрывать строки с `live: false`; план, который переведёт их на `<стенд>@<версия>`, должен сделать это вместе с решением о старых строках — the production frontend renders every row, so moving them now lists each model twice from the first vote — if wrong, ratings of our pipelines keep old names a little longer.

## Deferred findings

- Task 1: minor (deferred): models.py:141 Manifest stage key check uses match with $, accepts a trailing newline; use fullmatch
- Task 1: minor (deferred): non-string item in detail.lines rejects the whole StageEvent, which Relay would drop as bad_event
- Task 1: minor (deferred): defaults of ErrorEvent, Llm, LlmStatus, Context have no direct tests
- Task 1: minor (deferred): clean_sources does not cap title or url length
- Task 2: minor (deferred): example_stand non-JSON body gives 500, list-valued content is repr'd; sample is copied by authors
- Task 2: minor (deferred): example stand test checks substrings only; Task 3 check-backend now covers the shape
- Task 2: minor (deferred): sse.py buffer split is quadratic on a large unfinished block, bounded by max_bytes
- Task 3: minor (deferred): check.py:338-345 chunk with choices: [] (OpenAI usage chunk) raises IndexError and FAILs a normal stand; Task 5 relay must also tolerate it
- Task 3: minor (deferred): check.py Content-Type text/event-stream not checked
- Task 3: minor (deferred): check.py no overall deadline per stream, a stand sending only : ping forever hangs the checker
- Task 3: minor (deferred): check.py contexts before the text passes, order not tracked
- Task 3: minor (deferred): check.py 401/403 on manifest gives no --key hint; httpx.InvalidURL from a malformed URL crashes with a traceback
- Task 3: minor (deferred): test_contract_check.py no tests for cut stream, slow first token, non-200 chat, error event, blocks after [DONE], garbage JSON, manifest 500, plain-endpoint failure paths; assertion does not check which item failed
- Task 3: minor (deferred): meno-backend-v1.md:197 "название в меню мы назначаем сами" contradicts required manifest name; lines 30-36 wording on which endpoints are required conflicts with the table
- Task 3: minor (deferred): an oversize sources block also reports "no sources" as a second failure; test does not cover contexts or answer length
- Task 4: minor (deferred): registry.py:904 401/403/429 on the manifest wipes the last manifest and removes the stand from the menu, while 5xx keeps it unreachable
- Task 4: minor (deferred): registry.py:921 meno_stand_up series of a stand removed from the file stays at its last value; a false alert
- Task 4: minor (deferred): registry.py repeats arena rule and version-or-plain expression; small helper
- Task 4: minor (deferred): plain endpoint display_name from /v1/models no longer read
- Task 4: minor (deferred): test_stand_registry.py tempfile.mkstemp leaks an fd per _registry call; no test for a stand removed from the file
- Task 4: note: ruff reports two pre-existing issues in main.py (import order, unused pipeline in healthz); Task 10 step 6 must resolve them
- Task 5: minor (deferred): main.py:1429 relay_stream not wrapped in aclosing; carried into Task 6 dispatch
- Task 5: minor (deferred): relay.py:79 _json misses RecursionError on deeply nested JSON, kills the answer instead of dropping the event
- Task 5: minor (deferred): models.py:102 duration_ms 1e999 passes ge=0, Infinity then goes out as invalid JSON; allow_inf_nan=False
- Task 5: minor (deferred): relay.py:184 stand error text has no length limit before log and pipeline_runs.error
- Task 5: minor (deferred): relay.py:150,162 delta.role and finish_reason forwarded as free strings; allow known values only
- Task 5: minor (deferred): main.py:1436 request-time get_settings(); handlers use app.state.settings
- Task 5: minor (deferred): if persisting fails after [DONE], the request counts both ok and error and a second [DONE] follows (same as _stream_response)
- Task 5: note: non-stream stand path at main.py:1608 still returns detail; Task 6 rewrites it and must omit detail (Ruling 10)
- Task 6: minor (deferred): slot release rests on Starlette scheduling and cycle GC when a generator is parked at yield during cancel; a lost slot never self-heals; hardening is a StreamingResponse subclass with a shielded aclose and once-only release (shared-slot path has the same gap)
- Task 6: minor (deferred): disconnect test covers aclose at yield only; cancel while awaiting the stand works per probe but is untested
- Task 6: minor (deferred): ASK has no knowledge_base_id so that assertion is vacuous; lab@plain has no chat-level test after the port
- Task 6: minor (deferred): nothing shows gemma@ok answering while gemma@bad still hangs
- Task 6: minor (deferred): main.py:1625 finish_reason always stop on the non-stream path; a client cannot detect a truncated answer
- Task 6: minor (deferred): main.py:1507-1546 vs 1628-1657 near-identical failure handling in both stand paths
- Task 6: note for plan 4: gemma@menon-1.1 after a stand named menon-1.1 leaves the file runs in-process v2 (virtual ids exempt from the gate); spec 8.8 removes that path
- Task 7: minor (deferred): _stand_outcome runs after [DONE] outside _persist_success's try; a raise there lands in the error path instead of saving history; wrap with a trace-less fallback
- Task 7: minor (deferred): trace stage_details stored unsanitised, up to 16 MB of JSON in pipeline_stage_runs.detail, only with consent
- Task 7: minor (deferred): relay keeps contexts and trace even when not asked; consent granted mid-answer would store an unsolicited trace
- Task 7: minor (deferred): no tests for a failed consent read, wrong-typed trace keys, arena with consent, stand_build NULL for in-process runs
- Task 8: minor (deferred): arena.py:68 loop variable key shadows the lock key
- Task 8: minor (deferred): any x@y id with a valid stand pattern creates a y@unknown rating row with an invented stand name; live:false hides it, spec needs the row for removed stands
- Task 8: minor (deferred): no test for a stored side in the old form, nor for turn_index None through the API
- Task 9: minor (deferred): metrics.py:450 too_many_versions counted per call, not per distinct version
- Task 9: minor (deferred): llm label on the in-process path takes the client's id when the vLLM list is empty (resolve_model returns it), so that label is unbounded only while vLLM is down; checked llm/registry.py:119-121
- Task 9: minor (deferred): test_metrics.py fallback masks a wrong version expectation; fixture rebinding is brittle
- Task 10: minor (deferred): a meno-rag-api started outside run_backend.sh has no meno_contract unless installed; README does not say so
- Task 10: minor (deferred): CI torch assert would not catch a lazy import; add find_spec("torch") is None
- Task 10: minor (deferred): README gaps: max_concurrent default, api_key semantics, name, where STANDS_FILE is set
- Task 10: minor (deferred): src/meno_rag/api/knowledge_bases.py not formatted (untouched by the branch); whole-tree ruff 91 errors and 38 unformatted files, mostly v2 and tests/v2, predate the branch
- Final: minor (deferred): relay.py _CODE_RE.match accepts a trailing newline; classify_stand_error normalizes the code afterwards, so nothing reaches metrics
- Final review of bb02d8f..5928dcf: minor (fixed in 1c579e1): arena.py:178 stores knowledge_base_id None for a side whose stand is not in the arena, the frontend restores it as kb: null, and VoteRequest.kb_a: str (schemas.py:112) answered 422 before the vote's 403 stand_not_in_arena; kb_a and kb_b now take null as empty
- Final review of bb02d8f..5928dcf: minor (fixed in 8f84cac): main.py:645 POST /v1/models/refresh could run next to a background refresh, and a stand the second reload dropped made _discover raise KeyError (registry.py:162); refreshes no longer overlap (a second caller waits for the running one) and _discover returns for a stand no longer known
- Final review of bb02d8f..5928dcf: minor (fixed in 5e7bd23): main.py:364 the lifespan closed the stand client while a background discovery could still run; StandRegistry.aclose() cancels it and the lifespan calls it first
- Final review of bb02d8f..5928dcf: minor (fixed in ab3c3a4): metrics.py:184 bounded_llm gave menon-1.0 and menon-1.1 the 64 names of a stand, so the free OpenRouter list could push our own models into other with a too_many_llms violation; the two built-in stands keep 512 names and record no violation, stands keep 64
