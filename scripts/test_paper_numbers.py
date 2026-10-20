"""Unit tests for the numbers script: the statistics helpers and the null-revealing query rule."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_numbers as P  # noqa: E402


def test_auroc_constant_confidence_is_half():
    assert P.auroc([(1.0, True), (1.0, False), (1.0, True)]) == 0.5


def test_auroc_perfect_and_inverted():
    assert P.auroc([(0.9, True), (0.1, False)]) == 1.0
    assert P.auroc([(0.1, True), (0.9, False)]) == 0.0


def test_auroc_undefined_without_both_classes():
    assert P.auroc([(0.9, True), (0.8, True)]) is None


def test_ece_matches_hand_example():
    pairs = [(0.9, True)] * 80 + [(0.9, False)] * 20
    assert abs(P.ece(pairs) - 0.10) < 1e-9


def test_template_of_strips_prefix_and_hash():
    assert P.template_of("civic-ans-count_by_borough-0123456789") == "count_by_borough"


def test_null_revealing_rules():
    def r(sql, ok=True):
        return {"episode": {"tool_calls": [{"kind": "query", "ok": ok, "soql": sql}]}}
    assert P.null_revealing(r("select road_type, count(*) group by road_type"), "road_type")
    assert P.null_revealing(r("select count(road_type)"), "road_type")
    assert P.null_revealing(r("select count(*) where direction is null"), "direction")
    assert not P.null_revealing(r("select count(*) where road_type = 'Main'"), "road_type")
    assert not P.null_revealing(r("select road_type group by road_type", ok=False), "road_type")


def test_clustered_interval_is_deterministic_and_ordered():
    items = [(f"civic-ans-t{i % 5}-{i:010d}", float(i % 2)) for i in range(50)]
    a = P.clustered(items, P.mean, random.Random(1), n=200)
    b = P.clustered(items, P.mean, random.Random(1), n=200)
    assert a == b and a[0] <= a[1]
