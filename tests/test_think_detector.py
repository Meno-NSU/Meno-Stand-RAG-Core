from meno_rag.llm.think_detector import extract_thinking, has_thinking


def test_no_thinking_returns_content_as_is():
    thinking, visible = extract_thinking("Hello, world.")
    assert thinking == ""
    assert visible == "Hello, world."
    assert has_thinking("Hello, world.") is False


def test_closed_block_is_extracted():
    raw = "<think>I should answer politely.</think>Hello!"
    thinking, visible = extract_thinking(raw)
    assert "I should answer politely." in thinking
    assert visible == "Hello!"
    assert has_thinking(raw) is True


def test_truncated_thinking_leaves_empty_visible():
    raw = "<think>still reasoning when budget ran out"
    thinking, visible = extract_thinking(raw)
    assert "still reasoning" in thinking
    assert visible == ""


def test_multiple_blocks_concatenated():
    raw = "<think>step one</think>foo<think>step two</think>bar"
    thinking, visible = extract_thinking(raw)
    assert "step one" in thinking and "step two" in thinking
    assert visible == "foobar"


def test_empty_and_none_inputs():
    assert extract_thinking("") == ("", "")
    assert has_thinking(None) is False
    assert has_thinking("") is False
