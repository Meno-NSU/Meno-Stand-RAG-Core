import json

import pytest

from meno_rag.api.stands.config import StandEntry, parse_stands, read_stands_file


def test_parse_reads_an_entry():
    (lab,) = parse_stands(json.dumps([{"id": "lab", "url": "http://lab:9006/", "api_key": "k", "arena": True}]))
    assert lab == StandEntry(id="lab", url="http://lab:9006", api_key="k", arena=True)
    assert lab.headers() == {"Authorization": "Bearer k"} and lab.max_concurrent == 32


def test_a_dot_is_allowed_in_the_id_unless_reserved():
    assert parse_stands('[{"id": "menon-1.1", "url": "http://a"}]')[0].id == "menon-1.1"
    with pytest.raises(ValueError, match=r"menon-1\.1"):
        parse_stands('[{"id": "menon-1.1", "url": "http://a"}]', reserved_ids=frozenset({"menon-1.1"}))


@pytest.mark.parametrize(
    "raw",
    [
        "[{",
        '{"id": "lab"}',
        '[{"id": "lab"}]',
        '[{"id": "Lab X", "url": "http://a"}]',
        '[{"id": "lab", "url": "ftp://a"}]',
        '[{"id": "lab", "url": "http://a", "max_concurrent": 0}]',
        '[{"id": "lab", "url": "http://a"}, {"id": "lab", "url": "http://b"}]',
    ],
)
def test_parse_rejects_a_bad_file(raw):
    with pytest.raises(ValueError):
        parse_stands(raw)


def test_no_path_means_no_stands_and_a_missing_file_is_an_error(tmp_path):
    assert read_stands_file("") == []
    with pytest.raises(ValueError):
        read_stands_file(str(tmp_path / "absent.json"))


@pytest.mark.parametrize("key", ["my secret", "secret\n", "sec\tret", "secret\x00", "secret\x7f", "sécret", " secret"])
def test_an_api_key_that_cannot_be_a_header_is_refused_without_echoing_it(key):
    # h11 puts a bad header value into its exception text, and that text is logged
    raw = json.dumps([{"id": "lab", "url": "http://a", "api_key": key}])
    with pytest.raises(ValueError) as e:
        parse_stands(raw)
    assert "api_key" in str(e.value) and "secret" not in str(e.value) and "ret" not in str(e.value)


def test_an_ordinary_api_key_is_kept():
    key = "sk-AZaz09._~+/="
    assert parse_stands(json.dumps([{"id": "lab", "url": "http://a", "api_key": key}]))[0].api_key == key
