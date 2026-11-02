from meno_rag.v2.fusion import rrf


def test_two_lists_shared_item_sums():
    out = rrf([[(1, 9.0), (2, 8.0), (3, 7.0)], [(3, 5.0), (4, 4.0)]], k=60)
    d = dict(out)
    assert d[3] == 1 / 63 + 1 / 61
    assert d[1] == 1 / 61 and d[4] == 1 / 62
    assert [p for p, _ in out] == [3, 1, 2, 4]


def test_duplicate_in_one_list_counted_once_best_rank():
    out = dict(rrf([[(1, 3.0), (2, 2.0), (1, 1.0)]], k=60))
    assert out[1] == 1 / 61
    assert out[2] == 1 / 62


def test_limit_and_deterministic_ties():
    out = rrf([[(5, 1.0)], [(2, 1.0)]], k=60)
    assert [p for p, _ in out] == [2, 5]  # tie: same best rank -> lower position
    assert rrf([[(1, 1.0), (2, 1.0), (3, 1.0)]], limit=2) == out_first(2)


def out_first(n):
    return rrf([[(1, 1.0), (2, 1.0), (3, 1.0)]])[:n]


def test_empty():
    assert rrf([]) == [] and rrf([[], []]) == []
