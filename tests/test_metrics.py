import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodrec import metrics  # noqa: E402
from foodrec.geo import haversine_km  # noqa: E402


def test_precision_recall_hit():
    recs, rel = ["a", "b", "c", "d"], {"b", "x"}
    assert metrics.precision_at_k(recs, rel, 4) == 0.25
    assert metrics.recall_at_k(recs, rel, 4) == 0.5
    assert metrics.hit_rate_at_k(recs, rel, 4) == 1.0


def test_ndcg_perfect_and_empty():
    assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["c"], {"a"}, 1) == 0.0


def test_coverage_and_diversity():
    assert metrics.catalogue_coverage([["a", "b"], ["b", "c"]], 6) == 0.5
    assert metrics.intra_list_diversity(["a", "b"], {"a": "thai", "b": "thai"}) == 0.5


def test_haversine_one_degree_latitude():
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.1)
