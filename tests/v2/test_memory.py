import asyncio
import json

import pytest

from meno_rag.v2.memory import (
    DialogueMemory,
    SummaryStore,
    build_memory,
    estimate_tokens,
    precompute_next,
    recall_old_turns,
    split_history,
    summarize,
)


class FakeRouter:
    def __init__(self, reply="Сводка: пользователь спрашивал о приёме.", exc=None):
        self.reply, self.exc = reply, exc
        self.calls = []

    async def chat_completion_text(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.reply


class S:
    v2_history_budget_tokens = 600
    v2_summary_budget_tokens = 1000


def dialogue(n, size=100):
    h = []
    for i in range(n):
        h.append({"role": "user", "content": f"вопрос номер {i} " + "а" * size})
        h.append({"role": "assistant", "content": f"ответ номер {i} " + "б" * size})
    return h


def toks(msgs):
    return sum(estimate_tokens(m["content"]) for m in msgs)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 2


def test_split_keeps_whole_exchanges_in_order():
    h = dialogue(40)
    recent, older = split_history(h, 600)
    assert recent and older
    assert older + recent == h
    assert len(recent) % 2 == 0 and recent[0]["role"] == "user"
    assert toks(recent) <= 600


def test_split_short_history_all_recent():
    h = dialogue(2, 10)
    assert split_history(h, 5000) == (h, [])
    assert split_history([], 5000) == ([], [])


def test_split_truncates_huge_latest_turn():
    h = dialogue(3)
    h[-2] = {"role": "user", "content": "я" * 100000}
    h[-1] = {"role": "assistant", "content": "ю" * 100000}
    recent, older = split_history(h, 600)
    assert len(recent) == 2 and older == h[:4]
    assert all(m["content"].endswith("[…]") for m in recent)
    assert all(estimate_tokens(m["content"]) <= 300 for m in recent)
    assert toks(recent) <= 600


def test_summary_store_lru_and_key():
    st = SummaryStore(max_entries=2)
    a, b, c = [{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}], [{"role": "user", "content": "c"}]
    st.put(a, "A"); st.put(b, "B")
    assert st.get(a) == "A"
    st.put(c, "C")
    assert st.get(b) is None and st.get(a) == "A" and st.get(c) == "C"
    assert SummaryStore.key(a) == SummaryStore.key(list(a))


async def test_summarize_call_kwargs_and_truncation():
    r = FakeRouter("Первое предложение. " + "Длинное второе предложение. " * 50)
    older = dialogue(2, 10)
    out, ok = await summarize(r, object(), older, "прежняя сводка", 100)
    assert ok
    k = r.calls[0]
    assert k["stage"] == "query_rewrite" and k["temperature"] == 0
    assert k["max_tokens"] == 100 and k["timeout"] == 30
    assert k["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "прежняя сводка" in json.dumps(k["messages"], ensure_ascii=False)
    assert estimate_tokens(out) <= 100 and out.startswith("Первое предложение.")
    assert out.endswith(".")


async def test_summarize_failure_falls_back_to_user_questions():
    r = FakeRouter(exc=RuntimeError("down"))
    older = dialogue(3, 10)
    older[0]["content"] = "Расскажи про стипендию " + "x" * 400
    out, ok = await summarize(r, object(), older, "", 200)
    assert not ok
    assert "стипендию" in out or "вопрос номер 1" in out
    assert out.index("вопрос номер 1") < out.index("вопрос номер 2") if "вопрос номер 1" in out else True
    assert estimate_tokens(out) <= 200
    assert "ответ номер" not in out


async def test_summarize_empty_reply_falls_back():
    out, ok = await summarize(FakeRouter("  "), object(), dialogue(2, 5), "", 200)
    assert not ok
    assert "вопрос номер 0" in out


async def test_build_memory_40_exchanges_summary_once_then_cached():
    h = dialogue(40)
    r, store = FakeRouter(), SummaryStore()
    m = await build_memory(h, "что дальше?", r, object(), store, S())
    assert isinstance(m, DialogueMemory)
    assert toks(m.recent) <= 600 and m.older and m.older + m.recent == h
    assert m.summary and len(r.calls) == 1
    m2 = await build_memory(h, "что дальше?", r, object(), store, S())
    assert m2.summary == m.summary and len(r.calls) == 1


async def test_build_memory_empty_history():
    r = FakeRouter()
    m = await build_memory([], "привет", r, object(), SummaryStore(), S())
    assert (m.recent, m.older, m.summary, m.recalled) == ([], [], "", [])
    assert r.calls == []


async def test_build_memory_short_history_no_llm():
    r = FakeRouter()
    m = await build_memory(dialogue(2, 5), "q", r, object(), SummaryStore(), S())
    assert m.older == [] and m.summary == "" and r.calls == []


async def test_precompute_next_makes_next_build_free():
    h = dialogue(40)
    r, store = FakeRouter(), SummaryStore()
    await precompute_next(h, r, object(), store, S())
    assert len(r.calls) == 1
    await build_memory(h, "q", r, object(), store, S())
    assert len(r.calls) == 1


async def test_precompute_next_swallows_errors():
    class Boom:
        def get(self, *a):
            raise ValueError("x")

    await precompute_next(dialogue(40), FakeRouter(), object(), Boom(), S())


def test_recall_dorm_without_literal_word():
    older = [
        {"role": "user", "content": "Как получить место в общежитии НГУ?"},
        {"role": "assistant", "content": "Места дают первокурсникам."},
        {"role": "user", "content": "Сколько стоит обучение на ФИТ?"},
        {"role": "assistant", "content": "Стоимость зависит от программы."},
    ]
    assert recall_old_turns(older, "что ты говорил про общежития?") == older[:2]


def test_recall_stipend_and_unrelated():
    older = [
        {"role": "user", "content": "Какая стипендия у аспирантов?"},
        {"role": "assistant", "content": "Размер стипендии зависит от кафедры."},
        {"role": "user", "content": "Где находится библиотека?"},
        {"role": "assistant", "content": "Рядом с главным корпусом."},
    ]
    assert recall_old_turns(older, "напомни про стипендию из начала") == older[:2]
    assert recall_old_turns(older, "какая погода завтра") == []
    assert recall_old_turns([], "стипендия") == []


def test_recall_ranks_and_orders_chronologically():
    older = [
        {"role": "user", "content": "стипендия аспирантам размер"}, {"role": "assistant", "content": "ок"},
        {"role": "user", "content": "общежитие"}, {"role": "assistant", "content": "ок"},
        {"role": "user", "content": "стипендия"}, {"role": "assistant", "content": "ок"},
    ]
    got = recall_old_turns(older, "размер стипендии аспирантам и общежитие", k=2)
    assert got[:2] == older[:2]


def prompt_chars(call):
    return sum(len(m["content"]) for m in call["messages"])


async def test_incremental_one_call_with_delta_only():
    h = dialogue(41)
    r, store = FakeRouter(), SummaryStore()
    await precompute_next(h[:80], r, object(), store, S())
    n0 = len(r.calls)
    await build_memory(h, "q", r, object(), store, S())
    assert len(r.calls) == n0 + 1
    prompt = r.calls[-1]["messages"][1]["content"]
    assert "Прежняя сводка" in prompt and "Сводка: пользователь спрашивал" in prompt
    assert "вопрос номер 0 " not in prompt


async def test_incremental_when_two_exchanges_move_at_once():
    h = dialogue(42)
    r, store = FakeRouter(), SummaryStore()
    await precompute_next(h[:80], r, object(), store, S())
    n0 = len(r.calls)
    m = await build_memory(h, "q", r, object(), store, S())
    assert len(r.calls) == n0 + 1
    assert "Прежняя сводка" in r.calls[-1]["messages"][1]["content"]
    assert m.older + m.recent == h


async def test_huge_history_prompts_stay_bounded():
    h = dialogue(300, 600)
    r, store = FakeRouter(), SummaryStore()
    m = await build_memory(h, "q", r, object(), store, S())
    assert m.summary and len(r.calls) > 1
    bound = (4 * 1000 + 1000) * 3 + 1500
    assert max(prompt_chars(c) for c in r.calls) <= bound


async def test_fallback_not_cached():
    r, store = FakeRouter(exc=RuntimeError("x")), SummaryStore()
    h = dialogue(40)
    m = await build_memory(h, "q", r, object(), store, S())
    assert "вопрос номер" in m.summary
    r.exc = None
    await build_memory(h, "q", r, object(), store, S())
    assert len(r.calls) == 2


async def test_concurrent_builds_coalesce():
    class Slow(FakeRouter):
        async def chat_completion_text(self, **kw):
            await asyncio.sleep(0.05)
            return await super().chat_completion_text(**kw)

    h = dialogue(40)
    r, store = Slow(), SummaryStore()
    a, b = await asyncio.gather(
        precompute_next(h, r, object(), store, S()),
        build_memory(h, "q", r, object(), store, S()),
    )
    assert len(r.calls) == 1
