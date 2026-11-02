import asyncio
import json
from datetime import date

from meno_rag.v2.analysis import ANALYSIS_SCHEMA, analyze

TODAY = date(2026, 11, 2)


class FakeRouter:
    def __init__(self, reply="", exc=None, delay=0.0):
        self.reply, self.exc, self.delay = reply, exc, delay
        self.calls = []

    async def chat_completion_text(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.reply


def reply(route="nsu", standalone="Q?", subqueries=(), chain=False, focus=""):
    return json.dumps(
        {"route": route, "standalone_question": standalone,
         "subqueries": list(subqueries), "chain": chain, "focus": focus},
        ensure_ascii=False,
    )


async def run(router, question="Как поступить на ФИТ?", recent=(), summary="", timeout_s=5.0):
    return await analyze(router, object(), question=question, recent=list(recent),
                         summary=summary, today=TODAY, timeout_s=timeout_s)


async def test_valid_parse():
    r = FakeRouter(reply("nsu", "Сколько баллов нужно на ФИТ?", ["a", "b"], True, "ФИТ"))
    a = await run(r, "а баллы?")
    assert (a.route, a.standalone_question, a.subqueries, a.chain, a.focus, a.fallback) == (
        "nsu", "Сколько баллов нужно на ФИТ?", ["a", "b"], True, "ФИТ", False)


async def test_call_kwargs():
    r = FakeRouter(reply())
    await run(r)
    k = r.calls[0]
    assert k["stage"] == "query_rewrite"
    assert k["temperature"] == 0
    assert k["max_tokens"] == 512
    assert k["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert k["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "turn_analysis", "schema": ANALYSIS_SCHEMA, "strict": True},
    }
    assert k["timeout"] == 5.0
    assert k["messages"][0]["role"] == "system" and "2026-11-02" in k["messages"][0]["content"]


async def test_invalid_json_fallback():
    a = await run(FakeRouter("{not json"), "вопрос")
    assert (a.route, a.standalone_question, a.subqueries, a.chain, a.focus, a.fallback) == (
        "nsu", "вопрос", [], False, "", True)


async def test_truncated_json_fallback():
    assert (await run(FakeRouter('{"route": "nsu", "standalone_qu'))).fallback


async def test_exception_fallback():
    a = await run(FakeRouter(exc=RuntimeError("boom")), "вопрос")
    assert a.fallback and a.route == "nsu" and a.standalone_question == "вопрос"


async def test_timeout_fallback():
    a = await run(FakeRouter(reply(), delay=1.0), "вопрос", timeout_s=0.05)
    assert a.fallback and a.route == "nsu"


async def test_unknown_route_fallback():
    assert (await run(FakeRouter(reply(route="weather")))).fallback


async def test_missing_fields_fallback():
    assert (await run(FakeRouter('{"route": "nsu"}'))).fallback


async def test_nsu_stem_guard():
    a = await run(FakeRouter(reply("chitchat")), "Привет! Как поступить на ФИТ?")
    assert a.route == "nsu" and not a.fallback


async def test_off_topic_long_question_guard():
    q = " ".join(["слово"] * 13)
    assert (await run(FakeRouter(reply("off_topic")), q)).route == "nsu"


async def test_thanks_stays_chitchat():
    a = await run(FakeRouter(reply("chitchat")), "спасибо!")
    assert a.route == "chitchat"


async def test_short_off_topic_kept():
    assert (await run(FakeRouter(reply("off_topic")), "сосчитай до 100000")).route == "off_topic"


async def test_fenced_json():
    a = await run(FakeRouter("```json\n" + reply("general", "Что такое производная?") + "\n```"),
                  "объясни, что такое производная")
    assert a.route == "general" and not a.fallback


async def test_text_around_json():
    a = await run(FakeRouter("Вот ответ: " + reply("meta_dialog") + " Готово."), "что я спрашивал?")
    assert a.route == "meta_dialog" and not a.fallback


async def test_blank_standalone_uses_question():
    a = await run(FakeRouter(reply(standalone="   ")), "исходная реплика")
    assert a.standalone_question == "исходная реплика"


async def test_subqueries_dedup_and_blank():
    a = await run(FakeRouter(reply(subqueries=["a", " ", "a", "b ", "", "b"])))
    assert a.subqueries == ["a", "b"]


async def test_recent_truncation_and_layout():
    long = "я" * 900
    recent = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": long}]
    r = FakeRouter(reply())
    await run(r, "а на магистратуре?", recent=recent, summary="Речь о ФИТ")
    msgs = r.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    u = msgs[1]["content"]
    assert u.index("Сводка диалога: Речь о ФИТ") < u.index("Последние ходы:") < u.index(
        "Текущая реплика: а на магистратуре?")
    assert "я" * 600 + "…" in u and "я" * 601 not in u


async def test_only_last_three_exchanges():
    recent = []
    for i in range(5):
        recent += [{"role": "user", "content": f"вопрос{i}"}, {"role": "assistant", "content": f"ответ{i}"}]
    r = FakeRouter(reply())
    await run(r, "x", recent=recent)
    u = r.calls[0]["messages"][1]["content"]
    assert "вопрос1" not in u and "вопрос2" in u and "ответ4" in u


async def test_no_summary_no_recent():
    r = FakeRouter(reply())
    await run(r, "привет")
    u = r.calls[0]["messages"][1]["content"]
    assert "Сводка" not in u and "Последние ходы" not in u and "Текущая реплика: привет" in u
