from meno_rag.stand.context import normalize_urls


def test_normalize_urls_none_and_empty():
    assert normalize_urls(None) == []
    assert normalize_urls("") == []
    assert normalize_urls("   ") == []


def test_normalize_urls_single_string_is_stripped():
    assert normalize_urls("  http://a  ") == ["http://a"]


def test_normalize_urls_list_strips_dedupes_preserves_order():
    assert normalize_urls(["http://b", " http://a ", "http://b"]) == ["http://b", "http://a"]


def test_normalize_urls_tuple_and_non_str_elements():
    assert normalize_urls(("http://a", 123)) == ["http://a", "123"]
