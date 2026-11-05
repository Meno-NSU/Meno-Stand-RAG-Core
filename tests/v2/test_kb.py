import gzip
import json
from pathlib import Path

import pytest

from meno_rag.v2.kb import load_release


def _gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _chunk(cid: str, doc: str, indexed: bool) -> dict:
    return {
        "chunk_id": cid, "doc_id": doc, "url": f"https://x/{doc}", "title": "T " + doc,
        "date": None, "doc_type": "news", "tier": "A", "text": "текст " + cid,
        "index_text": "idx " + cid, "indexed": indexed, "extra": "ignored",
    }


def _make(root: Path, num_docs: int) -> None:
    _gz(root / "chunks/part-00000.jsonl.gz", [_chunk("a", "d1", True), _chunk("b", "d1", False)])
    _gz(root / "chunks/part-00001.jsonl.gz", [_chunk("c", "d2", True)])
    _gz(root / "docs.jsonl.gz", [
        {"doc_id": "d1", "title": "T d1", "urls": ["u1"], "tier": "A", "doc_type": "news"},
        {"doc_id": "d2", "title": "T d2", "urls": ["u2"], "tier": "B", "doc_type": "page"},
    ])
    p = root / "ragcore/knowledge/bm25/params.index.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"num_docs": num_docs}))


def test_load_release(tmp_path):
    _make(tmp_path, 2)
    rel = load_release(tmp_path)
    assert [c.chunk_id for c in rel.chunks] == ["a", "c"]
    assert rel.by_id == {"a": 0, "c": 1}
    assert rel.chunks[0].text == "текст a" and rel.chunks[0].index_text == "idx a"
    assert rel.docs["d2"]["urls"] == ["u2"]


def test_num_docs_mismatch(tmp_path):
    _make(tmp_path, 3)
    with pytest.raises(ValueError, match="2.*3|3.*2"):
        load_release(tmp_path)


def test_quality_score_loaded(tmp_path):
    _make(tmp_path, 2)
    rows = [dict(_chunk("a", "d1", True), quality_score=0.6), _chunk("c", "d2", True)]
    _gz(tmp_path / "chunks/part-00000.jsonl.gz", rows[:1])
    _gz(tmp_path / "chunks/part-00001.jsonl.gz", rows[1:])
    rel = load_release(tmp_path)
    assert rel.chunks[0].quality == 0.6 and rel.chunks[1].quality is None
