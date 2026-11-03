import bm25s
import pytest
from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.tokenization import tokenize_and_normalize_text
from meno_rag.v2.kb import Chunk, KbRelease
from meno_rag.v2.lexical import Bm25Retriever

TEXTS = [
    "Стипендия назначается студентам по результатам сессии",
    "Расписание занятий публикуется на сайте университета",
    "Общежитие предоставляется иногородним студентам",
    "Заявление на академический отпуск подается в деканат",
]


@pytest.fixture
def retriever(tmp_path):
    stem = SnowballStemmer("russian")
    toks = [tokenize_and_normalize_text(t, stem).split() for t in TEXTS]
    idx = bm25s.BM25()
    idx.index(toks)
    idx.save(str(tmp_path / "bm25"))
    chunks = [
        Chunk(f"c{i}", f"d{i}", "", "", None, None, None, t, t)
        for i, t in enumerate(TEXTS)
    ]
    rel = KbRelease(chunks=chunks, by_id={c.chunk_id: i for i, c in enumerate(chunks)})
    return Bm25Retriever(rel, tmp_path / "bm25")


def test_top_hit(retriever):
    res = retriever.search("где взять общежитие", 3)
    assert res and res[0][0] == 2 and res[0][1] > 0
    assert all(s > 0 for _, s in res)


def test_empty_query(retriever):
    assert retriever.search("", 5) == []
    assert retriever.search("?!...", 5) == []


def test_idf_from_the_index(retriever):
    stem = SnowballStemmer("russian")
    common, rare = stem.stem("студентам"), stem.stem("деканат")
    assert retriever.idf(rare) > retriever.idf(common) > 1.0  # "студентам" is in two of four texts
    assert retriever.idf("несуществующийтокен") >= retriever.idf(rare)
