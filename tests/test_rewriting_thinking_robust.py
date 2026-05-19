"""Regression tests for `parse_rewritten_queries` against thinking-model
outputs and chat-template artefacts.

The reference research code split the rewrite LLM's response on newlines
verbatim, which works fine for plain instruction-tuned models. With Qwen3,
DeepSeek-R1, and similar reasoning models, the response routinely opens
with `<think>...</think>`, and chat-template-aware models occasionally
trail a sentinel like `<|im_end|>` on its own line — both used to leak
straight into FAISS/BM25 as search queries."""

from meno_rag.stand.rewriting import parse_rewritten_queries


def test_plain_response_passes_through():
    out = parse_rewritten_queries("query one\nquery two\nquery three")
    assert out == ["query one", "query two", "query three"]


def test_strips_closed_think_block():
    raw = "<think>Let me decompose this. The user wants A and B.</think>\nwhat is A\nwhat is B"
    assert parse_rewritten_queries(raw) == ["what is A", "what is B"]


def test_truncated_think_yields_empty():
    """Model opened <think> and ran out of tokens before closing — no
    visible queries were ever produced. Anything between `<think>` and EOF
    must NOT become a search query."""
    raw = "<think>still reasoning, nearly out of tokens"
    assert parse_rewritten_queries(raw) == []


def test_multiple_think_blocks_collapse_to_visible():
    raw = "<think>first thought</think>\nquery 1\n<think>second thought</think>\nquery 2"
    assert parse_rewritten_queries(raw) == ["query 1", "query 2"]


def test_chat_template_sentinel_is_dropped():
    raw = "what is the deadline\nwhere to apply\n<|im_end|>"
    assert parse_rewritten_queries(raw) == [
        "what is the deadline",
        "where to apply",
    ]


def test_lone_tag_lines_are_dropped():
    """Defence-in-depth: if for some reason `extract_thinking` misses a tag
    variant, a line that is JUST a tag must still be dropped."""
    raw = "<think>\nwhat is A\n</think>\n<assistant>\nwhat is B"
    out = parse_rewritten_queries(raw)
    assert "<assistant>" not in out
    assert "<think>" not in out
    assert "</think>" not in out
    assert "what is B" in out


def test_short_lines_and_bare_bullets_dropped():
    raw = "a\n1.\n- \n\nwhat is the schedule"
    assert parse_rewritten_queries(raw) == ["what is the schedule"]


def test_russian_queries_pass_through():
    raw = "<think>Декомпозирую вопрос на части.</think>\nКакие факультеты есть в НГУ\nКогда дни открытых дверей в НГУ"
    assert parse_rewritten_queries(raw) == [
        "Какие факультеты есть в НГУ",
        "Когда дни открытых дверей в НГУ",
    ]


def test_empty_input_yields_empty():
    assert parse_rewritten_queries("") == []
    assert parse_rewritten_queries("   \n\n  ") == []
