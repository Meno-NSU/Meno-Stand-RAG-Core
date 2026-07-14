from meno_rag.stand.context import flatten_sources, normalize_urls, prepare_context


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


_DOCS = [
    {
        "doc_full_text": "Первый документ про НГУ.",
        "chunks": [{"start_char": 0, "end_char": 24}],
        "doc_title": "Документ А",
        "doc_annotation": "Аннотация",
        "url": "https://a.test",
    },
    {
        "doc_full_text": "Сводный документ про НГУ.",
        "chunks": [{"start_char": 0, "end_char": 25}],
        "doc_title": "Сводка",
        "doc_annotation": "Аннотация",
        "url": ["https://x.test", "https://y.test"],
    },
]
_MAPPING = {
    "0": {"doc_index": 0, "local_chunk_index": 0},
    "1": {"doc_index": 1, "local_chunk_index": 0},
}


def test_prepare_context_returns_structured_sources():
    descriptions, sources = prepare_context([0], [0.9], _DOCS, _MAPPING, min_document_quality=0.0)
    assert len(descriptions) == 1
    assert sources == [{"document_title": "Документ А", "source_urls": ["https://a.test"]}]


def test_prepare_context_handles_list_url():
    _descriptions, sources = prepare_context([1], [0.9], _DOCS, _MAPPING, min_document_quality=0.0)
    assert sources == [{"document_title": "Сводка", "source_urls": ["https://x.test", "https://y.test"]}]


def test_flatten_sources_expands_one_row_per_url():
    per_doc = [
        {"document_title": "Документ А", "source_urls": ["https://a.test"]},
        {"document_title": "Сводка", "source_urls": ["https://x.test", "https://y.test"]},
    ]
    assert flatten_sources(per_doc) == [
        {"document_title": "Документ А", "source_url": "https://a.test"},
        {"document_title": "Сводка", "source_url": "https://x.test"},
        {"document_title": "Сводка", "source_url": "https://y.test"},
    ]


def test_flatten_sources_empty_urls_yields_one_empty_row():
    assert flatten_sources([{"document_title": "T", "source_urls": []}]) == [{"document_title": "T", "source_url": ""}]
