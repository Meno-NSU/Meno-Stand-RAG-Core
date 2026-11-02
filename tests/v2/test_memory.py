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
    v2_summary_budget_tokens = 100


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
    out = await summarize(r, object(), older, "прежняя сводка", 100)
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
    out = await summarize(r, object(), older, "", 200)
    assert "стипендию" in out or "вопрос номер 1" in out
    assert out.index("вопрос номер 1") < out.index("вопрос номер 2") if "вопрос номер 1" in out else True
    assert estimate_tokens(out) <= 200
    assert "ответ номер" not in out


async def test_summarize_empty_reply_falls_back():
    out = await summarize(FakeRouter("  "), object(), dialogue(2, 5), "", 200)
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


def test_recall_finds_dorm_exchange():
    older = [
        {"role": "user", "content": "Как получить место в общежитии НГУ?"},
        {"role": "assistant", "content": "Про общежития я говорил: места дают первокурсникам."},
        {"role": "user", "content": "Сколько стоит обучение на ФИТ?"},
        {"role": "assistant", "content": "Стоимость зависит от программы."},
    ]
    got = recall_old_turns(older, "что ты говорил про общежития?")
    assert got == older[:2]


def test_recall_needs_two_common_stems_and_orders_chronologically():
    older = dialogue(3, 5)
    assert recall_old_turns(older, "совсем другое") == []
    assert recall_old_turns([], "q") == []
    older = [
        {"role": "user", "content": "стипендия аспирантам размер"},
        {"role": "assistant", "content": "ок"},
        {"role": "user", "content": "стипендия аспирантам когда выплата"},
        {"role": "assistant", "content": "ок"},
    ]
    got = recall_old_turns(older, "размер стипендии аспирантам", k=2)
    assert got == older
