# OpenRouter free models — operator cheatsheet

When most free OpenRouter models silently fail in our pipeline, the cause is
usually one of: rate-limit ceiling hit, request body too large for the model's
context window, the model rejecting `seed`/`response_format` parameters, or the
upstream returning an empty completion that downstream code can't recover from.

## How to fill this table

1. Start the backend with `OPENROUTER_API_KEY` set.
2. Hit `GET /v1/diagnostics/openrouter` — it pings every discovered free model
   with a tiny prompt and reports `ok`, `latency_ms`, `finish_reason`,
   `content_preview`, `error_code`, `error_message`.
3. For models that succeed on the probe but still fail on real RAG requests,
   check `logs/meno-rag-api.log` for the structured events listed below.

## Key log events to grep

| Event | Meaning |
|---|---|
| `or_request_4xx` | Body returned by OpenRouter on a 4xx (now logged). Most common cause: `context_length_exceeded`. |
| `or_empty_completion` | Model returned 200 but with empty `content` — often happens when only `<think>` was generated. |
| `llm_thinking_detected` | Response contained `<think>...</think>`. Look at `visible_chars` to see if anything useful was left after stripping. |
| `or_request_rate_limited` | Per-model rate limit hit; `retry_after_sec` indicates how long the model stays unusable. |
| `qa_prompt_oversized` | Our QA prompt > 30k chars — likely too large for the free-tier context window of many models. |

## Known issues (fill in after running diagnostics)

| Model | Context window | Common failure | Workaround |
|---|---|---|---|
| _example: qwen/qwen-2.5-72b-instruct:free_ | _8k_ | _context_length_exceeded on 12-doc RAG context_ | _Lower `RERANK_TOP_K`._ |
| _example: deepseek/deepseek-chat:free_ | _32k_ | _Frequent 5xx during peak hours_ | _Retry with backoff; the structured error from `/v1/chat/completions` includes `retryable: true`._ |

## Related code

- [src/meno_rag/llm/openrouter_client.py](../src/meno_rag/llm/openrouter_client.py): 4xx body logging, usage metrics, empty-completion warning.
- [src/meno_rag/api/main.py](../src/meno_rag/api/main.py) — `diagnostics_openrouter`: the probe endpoint.
- [src/meno_rag/llm/think_detector.py](../src/meno_rag/llm/think_detector.py): used to detect `<think>` blocks emitted by Qwen3.
