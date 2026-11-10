"""check-backend: verify over the network that a stand follows the meno-backend/1 contract.

Usage:  check-backend http://host:port [--key KEY]

Only httpx and pydantic are needed. Messages are in Russian because they are read by stand authors.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from meno_contract.models import (
    MAX_ANSWER_CHARS,
    MAX_ANSWER_S,
    MAX_BLOCK_BYTES,
    MAX_CHUNK_PAUSE_S,
    MAX_STAGE_NAMES,
    MAX_STREAM_BYTES,
    MAX_TRACE_BLOCK_BYTES,
    PROTOCOL,
    STANDARD_STAGES,
    Context,
    Manifest,
    StageEvent,
    clean_sources,
)
from meno_contract.sse import SseBlockTooLarge, iter_blocks

QUESTION = "Привет! Расскажи коротко, что такое НГУ."
UNKNOWN_MODEL = "meno-check-no-such-model"


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str = ""
    warning: bool = False


@dataclass
class Report:
    items: list[CheckItem] = field(default_factory=list)
    plain: bool = False

    @property
    def ok(self) -> bool:
        return all(item.ok or item.warning for item in self.items)


def format_report(report: Report) -> str:
    lines: list[str] = []
    for item in report.items:
        mark = "OK" if item.ok else ("ПРЕДУПРЕЖДЕНИЕ" if item.warning else "FAIL")
        lines.append(f"[{mark}] {item.name}")
        if item.detail and not item.ok:
            lines.extend(f"    {line}" for line in item.detail.splitlines())
    failed = sum(1 for i in report.items if not i.ok and not i.warning)
    warned = sum(1 for i in report.items if not i.ok and i.warning)
    if report.ok:
        verdict = "Итог: стенд соответствует контракту."
    else:
        verdict = f"Итог: не пройдено пунктов: {failed}. Исправьте их и запустите проверку снова."
    if warned:
        verdict += f" Предупреждений: {warned} (они не мешают)."
    lines.append(verdict)
    return "\n".join(lines)


@dataclass
class _Stream:
    """What was seen in one answer stream."""

    status: int = 200
    problems: list[str] = field(default_factory=list)
    stage_events: list[StageEvent] = field(default_factory=list)
    content: str = ""
    done: bool = False
    after_done: int = 0
    sources_events: int = 0
    contexts_events: int = 0
    errors: list[str] = field(default_factory=list)
    first_token_s: float | None = None
    too_long: bool = False
    failure: str = ""  # connection-level failure; the other fields are then incomplete


def _problem_text(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'значение'}: {e['msg']}" for e in exc.errors()[:3])


async def _read_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    model: str,
    first_token_limit_s: float,
    *,
    pause_limit_s: float = MAX_CHUNK_PAUSE_S,
    answer_limit_s: float = MAX_ANSWER_S,
) -> _Stream:
    """Read one answer the way the gateway does: a pause between chunks up to `pause_limit_s`, the whole answer up
    to `answer_limit_s`. The first token is timed here and judged against `first_token_limit_s` afterwards."""
    seen = _Stream()
    body = {"model": model, "messages": [{"role": "user", "content": QUESTION}], "stream": True}
    timeout = httpx.Timeout(30.0, read=pause_limit_s)
    started = time.monotonic()
    try:
        await asyncio.wait_for(_consume(client, url, headers, body, timeout, seen, started), answer_limit_s)
    except TimeoutError:
        seen.failure = f"Ответ шёл дольше {answer_limit_s:g} с: шлюз оборвёт его. Отвечайте быстрее."
    except SseBlockTooLarge:
        seen.problems.append(f"Блок SSE больше {MAX_TRACE_BLOCK_BYTES} байт. Шлите ответ частями, а не одним блоком.")
    except httpx.TimeoutException:
        if seen.first_token_s is None:
            seen.failure = f"Стенд не выдал первый токен: молчал дольше {pause_limit_s:g} с."
        else:
            seen.failure = f"Пауза между кусками потока дольше {pause_limit_s:g} с: шлюз оборвёт ответ."
    except httpx.HTTPError as exc:
        seen.failure = f"Нет связи со стендом или поток оборван: {type(exc).__name__}: {exc}"
    return seen


async def _consume(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: httpx.Timeout,
    seen: _Stream,
    started: float,
) -> None:
    async with client.stream("POST", url, json=body, headers=headers, timeout=timeout) as response:
        seen.status = response.status_code
        if response.status_code != 200:
            await response.aread()
            seen.failure = f"Стенд ответил HTTP {response.status_code}: {response.text[:200]!r}"
            return
        async for block in iter_blocks(_as_the_gateway_reads(response, seen), max_bytes=MAX_TRACE_BLOCK_BYTES):
            if seen.done:
                seen.after_done += 1
                continue
            if block.event not in ("contexts", "pipeline_trace") and block.raw_size > MAX_BLOCK_BYTES:
                seen.problems.append(
                    f"Блок {block.event!r} занимает {block.raw_size} байт, шлюз принимает не больше "
                    f"{MAX_BLOCK_BYTES} (для contexts и pipeline_trace до {MAX_TRACE_BLOCK_BYTES}). "
                    "Шлите ответ частями."
                )
                continue
            _take_block(seen, block.event, block.data, started)


async def _as_the_gateway_reads(response: httpx.Response, seen: _Stream) -> AsyncIterator[str]:
    """The stream as the gateway reads it: UTF-8 whatever charset Content-Type names, a byte that is not UTF-8 turned
    into U+FFFD, cut after MAX_STREAM_BYTES bytes (comments and thinking included). Bytes that are not UTF-8 are a
    problem anywhere in the stream, even where the gateway skips them (a comment, anything after [DONE]): the
    contract asks for UTF-8 throughout."""
    shown = codecs.getincrementaldecoder("utf-8")(errors="replace")
    strict = codecs.getincrementaldecoder("utf-8")()
    utf8 = True
    read = 0
    async for data in response.aiter_bytes():
        read += len(data)
        if read > MAX_STREAM_BYTES:
            seen.problems.append(f"Поток длиннее {MAX_STREAM_BYTES} байт: шлюз оборвёт ответ. Сократите поток.")
            return
        if utf8 and not _decodes(strict, data):
            utf8 = False
            seen.problems.append(_not_utf8(response))
        if text := shown.decode(data):
            yield text
    if utf8 and not _decodes(strict, b"", final=True):
        seen.problems.append(_not_utf8(response))
    if tail := shown.decode(b"", final=True):
        yield tail


def _decodes(decoder: codecs.IncrementalDecoder, data: bytes, *, final: bool = False) -> bool:
    try:
        decoder.decode(data, final)
    except UnicodeDecodeError:
        return False
    return True


def _not_utf8(response: httpx.Response) -> str:
    named = f" (Content-Type называет charset={response.charset_encoding})" if response.charset_encoding else ""
    return (
        f"Поток не в UTF-8{named}. Контракт требует UTF-8 во всём потоке. Шлюз читает поток как UTF-8, какую бы "
        "кодировку ни называл Content-Type, и заменяет такие байты на «�»: если они попадут в текст ответа, "
        "источники или этапы, читатель увидит «�» вместо букв. Шлите поток в UTF-8."
    )


def _take_block(seen: _Stream, event: str, data: str, started: float) -> None:
    if event == "message":
        if data.strip() == "[DONE]":
            seen.done = True
            return
        try:
            payload = json.loads(data)
            if isinstance(payload, dict) and payload.get("choices") == []:
                return  # the usage chunk many servers send last; the gateway accepts it
            delta = payload["choices"][0]["delta"]
        except (ValueError, KeyError, IndexError, TypeError):
            seen.problems.append(
                f"Блок data не похож на OpenAI chat.completion.chunk (нужно choices[0].delta): {data[:120]!r}"
            )
            return
        text = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(text, str) and text:
            if seen.first_token_s is None:
                seen.first_token_s = time.monotonic() - started
            seen.content += text
            if len(seen.content) > MAX_ANSWER_CHARS and not seen.too_long:
                seen.too_long = True
                seen.problems.append(f"Ответ длиннее {MAX_ANSWER_CHARS} символов: шлюз оборвёт его. Сократите ответ.")
        return
    if event not in ("stage", "sources", "contexts", "error"):
        return  # thinking, pipeline_trace and unknown events are allowed
    try:
        payload = json.loads(data)
    except ValueError:
        seen.problems.append(f"Событие {event}: данные не являются JSON: {data[:120]!r}")
        return
    try:
        if event == "stage":
            seen.stage_events.append(StageEvent.model_validate(payload))
        elif event == "sources":
            if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
                raise ValueError('нужен объект {"sources": [...]}')
            _rows, dropped = clean_sources(payload["sources"])
            if dropped:
                seen.problems.append(
                    f"В sources {dropped} записей отброшены: нужны объекты с document_title и source_url "
                    "(http:// или https://), не больше 20 штук."
                )
            seen.sources_events += 1
        elif event == "contexts":
            if not isinstance(payload, dict) or not isinstance(payload.get("contexts"), list):
                raise ValueError('нужен объект {"contexts": [...]}')
            for item in payload["contexts"]:
                Context.model_validate(item)
            seen.contexts_events += 1
        else:
            seen.errors.append(str(payload.get("message") or payload.get("code") or payload)[:200])
    except (ValidationError, ValueError, AttributeError) as exc:
        reason = _problem_text(exc) if isinstance(exc, ValidationError) else str(exc)
        seen.problems.append(f"Событие {event} имеет неверный формат ({reason}).")


def _stream_items(
    prefix: str, seen: _Stream, *, labels: set[str] | None, trace: bool, full: bool, limit_s: float
) -> list[CheckItem]:
    """Judge one stream. `labels` None means a stand without a manifest: stages are not required."""
    if seen.failure:
        return [CheckItem(f"{prefix}: запрос", False, seen.failure)]
    items = [CheckItem(f"{prefix}: запрос принят, идёт поток", True)]
    if seen.problems:
        items.append(CheckItem(f"{prefix}: формат блоков", False, "\n".join(seen.problems)))
    if seen.errors:
        items.append(CheckItem(f"{prefix}: события error", False, "Стенд прислал error: " + "; ".join(seen.errors)))

    if not seen.content:
        items.append(
            CheckItem(
                f"{prefix}: текст ответа",
                False,
                "В потоке нет текста ответа. Шлите блоки data: в формате OpenAI chat.completion.chunk "
                'с непустым choices[0].delta.content (стенд должен отвечать при "stream": true).',
            )
        )
    else:
        items.append(CheckItem(f"{prefix}: текст ответа", True))
        if "<think>" in seen.content:
            items.append(
                CheckItem(
                    f"{prefix}: <think> в тексте ответа",
                    False,
                    "В ответе есть <think>. Рассуждения шлите событием event: thinking, а не в тексте ответа. "
                    "Пока фронтенд вырежет их сам, но так делать не нужно.",
                    warning=True,
                )
            )
        if seen.first_token_s is not None and seen.first_token_s > limit_s:
            items.append(
                CheckItem(
                    f"{prefix}: первый токен",
                    False,
                    f"Первый токен пришёл через {seen.first_token_s:.0f} с при лимите {limit_s:g} с.",
                )
            )

    if full:
        items += _stage_items(prefix, seen, labels)
        if seen.sources_events:
            items.append(CheckItem(f"{prefix}: событие sources", True))
        else:
            items.append(
                CheckItem(
                    f"{prefix}: событие sources",
                    False,
                    'Нет event: sources. Пришлите его хотя бы раз до [DONE] в виде {"sources": [...]}; '
                    "список может быть пустым.",
                )
            )
        if trace:
            if seen.contexts_events:
                items.append(CheckItem(f"{prefix}: событие contexts", True))
            else:
                items.append(
                    CheckItem(
                        f"{prefix}: событие contexts",
                        False,
                        "При заголовке X-Meno-Trace: 1 нужно прислать event: contexts после текста ответа и "
                        'до [DONE]: {"contexts": [{"id": "...", "text": "...", "rank": 1}]}.',
                    )
                )
        elif seen.contexts_events:
            items.append(
                CheckItem(
                    f"{prefix}: событие contexts",
                    False,
                    "Без заголовка X-Meno-Trace: 1 событие contexts слать нельзя. Отправляйте contexts только "
                    "когда заголовок есть.",
                )
            )

    if not seen.done:
        items.append(
            CheckItem(
                f"{prefix}: завершение [DONE]",
                False,
                "Поток закончился без data: [DONE]. Последним блоком отправьте строку data: [DONE].",
            )
        )
    elif seen.after_done:
        items.append(
            CheckItem(
                f"{prefix}: завершение [DONE]",
                False,
                f"После data: [DONE] пришло ещё блоков: {seen.after_done}. [DONE] должен быть последним, "
                "а sources и contexts нужно слать до него.",
            )
        )
    else:
        items.append(CheckItem(f"{prefix}: завершение [DONE]", True))
    return items


def _stage_items(prefix: str, seen: _Stream, labels: set[str] | None) -> list[CheckItem]:
    events = seen.stage_events
    if not events:
        return [
            CheckItem(
                f"{prefix}: этапы",
                False,
                "Нет ни одного этапа. Пришлите хотя бы один event: stage, например "
                '{"stage": "generation", "status": "started"} и затем "completed".',
            )
        ]
    items = [CheckItem(f"{prefix}: этапы", True)]
    open_count: dict[str, int] = {}
    unpaired: list[str] = []
    for event in events:
        if event.status == "started":
            open_count[event.stage] = open_count.get(event.stage, 0) + 1
        elif open_count.get(event.stage, 0) > 0:
            open_count[event.stage] -= 1
        else:
            unpaired.append(f"{event.stage} ({event.status} без started)")
    unpaired += [f"{name} (started без завершения)" for name, n in open_count.items() if n > 0]
    if unpaired:
        items.append(
            CheckItem(
                f"{prefix}: парность этапов",
                False,
                "Нет парного завершения у этапов: "
                + ", ".join(unpaired)
                + ". У каждого started должен быть completed, failed или skipped.",
            )
        )
    else:
        items.append(CheckItem(f"{prefix}: парность этапов", True))
    if len({e.stage for e in events}) > MAX_STAGE_NAMES:
        items.append(
            CheckItem(
                f"{prefix}: число этапов",
                False,
                f"Больше {MAX_STAGE_NAMES} разных имён этапов в одном ответе: остальные шлюз пропустит.",
            )
        )
    if labels is not None:
        names = {e.stage for e in events}
        missing = sorted(n for n in names if n not in STANDARD_STAGES and n not in labels)
        if missing:
            items.append(
                CheckItem(
                    f"{prefix}: подписи этапов",
                    False,
                    "Нет подписи в манифесте у этапов: "
                    + ", ".join(missing)
                    + '. Добавьте их в "stages": {"имя": {"ru": "...", "en": "..."}}.',
                )
            )
        else:
            items.append(CheckItem(f"{prefix}: подписи этапов", True))
    return items


async def _check_manifest(client: httpx.AsyncClient, base: str, headers: dict[str, str]) -> tuple[Manifest | None, str]:
    """Returns (manifest, note). manifest None with an empty note means: no manifest (404)."""
    response = await client.get(f"{base}/meno/v1/manifest", headers=headers, timeout=30.0)
    if response.status_code == 404:
        return None, ""
    if response.status_code != 200:
        return None, f"Манифест ответил HTTP {response.status_code}. Нужен 200 и JSON по пути /meno/v1/manifest."
    try:
        raw = response.json()
    except ValueError:
        return None, "Манифест не является JSON."
    if not isinstance(raw, dict):
        return None, "Манифест должен быть JSON-объектом."
    if raw.get("protocol") != PROTOCOL:
        return None, f'Неверный протокол в манифесте: {raw.get("protocol")!r}, нужно "{PROTOCOL}".'
    try:
        return Manifest.model_validate(raw), ""
    except ValidationError as exc:
        return None, "Манифест не прошёл проверку: " + _problem_text(exc)


async def run_checks(
    client: httpx.AsyncClient, base_url: str, *, api_key: str = "", first_token_limit_s: float = 120.0
) -> Report:
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    report = Report()
    try:
        manifest, note = await _check_manifest(client, base, headers)
        if manifest is None and not note:
            report.plain = True
            await _check_plain(client, base, headers, report, first_token_limit_s)
            return report
        if manifest is None:
            report.items.append(CheckItem("Манифест", False, note))
            return report
        report.items.append(CheckItem("Манифест", True))
        await _check_chat(client, base, headers, manifest, report, first_token_limit_s)
    except httpx.HTTPError as exc:
        report.items.append(
            CheckItem(
                "Связь со стендом",
                False,
                f"Не удалось связаться со стендом ({type(exc).__name__}: {exc}). "
                "Проверьте адрес, порт и что стенд запущен.",
            )
        )
    return report


async def _check_chat(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    manifest: Manifest,
    report: Report,
    limit_s: float,
) -> None:
    url = f"{base}/v1/chat/completions"
    labels = set(manifest.stages)
    request_headers = {**headers, "X-Meno-Request-Id": f"check-{uuid.uuid4().hex[:12]}"}
    plain_run = await _read_stream(client, url, request_headers, manifest.default_llm, limit_s)
    report.items += _stream_items("Обычный вопрос", plain_run, labels=labels, trace=False, full=True, limit_s=limit_s)
    trace_run = await _read_stream(client, url, {**request_headers, "X-Meno-Trace": "1"}, manifest.default_llm, limit_s)
    report.items += _stream_items(
        "Вопрос с X-Meno-Trace", trace_run, labels=labels, trace=True, full=True, limit_s=limit_s
    )
    report.items.append(await _check_unknown_model(client, url, request_headers))


async def _check_unknown_model(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> CheckItem:
    name = "Неизвестная модель"
    body = {"model": UNKNOWN_MODEL, "messages": [{"role": "user", "content": "ping"}], "stream": True}
    async with client.stream("POST", url, json=body, headers=headers, timeout=30.0) as response:
        status = response.status_code
        if not 400 <= status < 500:
            return CheckItem(
                name,
                False,
                f"На неизвестную модель стенд ответил HTTP {status}. Нужен 400 и тело "
                '{"error": {"message": "...", "code": "model_not_found"}}, до начала потока.',
            )
        await response.aread()
    try:
        error = response.json()["error"]
        if not isinstance(error, dict):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        return CheckItem(
            name,
            False,
            'Статус верный, но тело ответа не в формате {"error": {"message": "...", "code": "..."}}.',
            warning=True,
        )
    return CheckItem(name, True)


async def _check_plain(
    client: httpx.AsyncClient, base: str, headers: dict[str, str], report: Report, limit_s: float
) -> None:
    report.items.append(
        CheckItem(
            "Участие в арене",
            False,
            "Манифеста нет (404 на /meno/v1/manifest): такой эндпоинт можно подключить в меню как обычный "
            "OpenAI-совместимый, но он не попадёт в арену и лидерборд. Чтобы участвовать в арене, "
            "реализуйте манифест, этапы и источники по описанию контракта.",
        )
    )
    response = await client.get(f"{base}/v1/models", headers=headers, timeout=30.0)
    models: list[Any] = []
    if response.status_code == 200:
        try:
            models = [m["id"] for m in response.json()["data"] if isinstance(m, dict) and m.get("id")]
        except (ValueError, KeyError, TypeError):
            models = []
    if not models:
        report.items.append(
            CheckItem(
                "Список моделей /v1/models",
                False,
                'Нужен GET /v1/models с ответом {"data": [{"id": "..."}]}, иначе в меню нечего показать.',
            )
        )
        return
    report.items.append(CheckItem("Список моделей /v1/models", True))
    seen = await _read_stream(client, f"{base}/v1/chat/completions", headers, str(models[0]), limit_s)
    report.items += _stream_items("Обычный вопрос", seen, labels=None, trace=False, full=False, limit_s=limit_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-backend", description="Проверка стенда на контракт meno-backend/1")
    parser.add_argument("url", help="адрес стенда, например http://host:9100")
    parser.add_argument("--key", default="", help="ключ API стенда (Authorization: Bearer ...)")
    args = parser.parse_args(argv)

    async def run() -> Report:
        async with httpx.AsyncClient() as client:
            return await run_checks(client, args.url, api_key=args.key)

    report = asyncio.run(run())
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
