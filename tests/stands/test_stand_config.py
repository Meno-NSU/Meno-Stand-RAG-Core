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
