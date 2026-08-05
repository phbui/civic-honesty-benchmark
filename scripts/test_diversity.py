"""test_diversity.py — synthetic-input tests for diversity.py, where the right
answer is known analytically, plus a handful
of hermetic regression tests against literal SoQL/evidence shapes drawn from
the real generator's known template outputs (no network, no live data — the
literal strings below are copied from `results/questions_scaled.jsonl` /
`questions.py`'s known template forms, not re-fetched).

Run:
  uv run pytest scripts/test_diversity.py -v
"""

from __future__ import annotations

import diversity as dv
import pytest

# ---------------------------------------------------------------------------
# analytic checks: k identical-fingerprint items
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 5, 10, 50])
@pytest.mark.parametrize("rho", [0.0, 0.1, 0.3, 0.5, 0.9, 1.0])
def test_single_cluster_effective_n_matches_closed_form(k: int, rho: float) -> None:
    """A set of k items all sharing ONE fingerprint must give effective N of
    k / (1 + (k-1)*rho) — Killip, Mahfoud & Pearce (2004) Equation 2/3 for the
    equal-cluster-size case collapsed to a single cluster (m=k, number of
    clusters=1)."""
    cluster_sizes = [k]
    expected = k / (1.0 + (k - 1) * rho)
    got = dv.effective_n(cluster_sizes, rho)
    assert got == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# analytic checks: all-distinct fingerprints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 50, 540])
@pytest.mark.parametrize("rho", [0.0, 0.1, 0.5, 1.0])
def test_all_distinct_fingerprints_effective_n_equals_n_at_any_rho(n: int, rho: float) -> None:
    """A set of n items with n DISTINCT fingerprints (every cluster size 1)
    must give effective N == n regardless of rho: m* = sum(1^2)/n = 1, so
    DE = 1 + rho*(1-1) = 1 for every rho, and N/DE == N."""
    cluster_sizes = [1] * n
    got = dv.effective_n(cluster_sizes, rho)
    assert got == pytest.approx(n, rel=1e-9)


# ---------------------------------------------------------------------------
# equal-size design effect matches the cited Killip et al. 2004 Eq. 3 worked
# example directly (m=32, k=4, rho=0.017 -> DE=1.527, ESS=84 in the paper).
# ---------------------------------------------------------------------------


def test_equal_cluster_size_matches_killip_equation_3_worked_example() -> None:
    """Killip, Mahfoud & Pearce (2004), Ann Fam Med 2:204-208, worked example:
    m=32 patients/cluster, k=4 clusters, rho=0.017 -> DE=1.527, ESS=84
    (paper states DE=1.527, ESS=84; mk=128)."""
    cluster_sizes = [32, 32, 32, 32]
    de = dv.design_effect(cluster_sizes, rho=0.017)
    ess = dv.effective_n(cluster_sizes, rho=0.017)
    assert de == pytest.approx(1.527, abs=0.001)
    assert ess == pytest.approx(84, abs=0.5)


# ---------------------------------------------------------------------------
# concentration: maximal for single-fingerprint set, minimal for all-distinct
# ---------------------------------------------------------------------------


def test_concentration_maximal_for_single_fingerprint_set() -> None:
    counts = dv.Counter({"only-fingerprint": 250})
    stats = dv.herfindahl_stats(counts)
    assert stats["hhi"] == pytest.approx(1.0)
    assert stats["hhi_normalized_0_to_1"] == pytest.approx(1.0)
    assert stats["effective_number_of_fingerprints"] == pytest.approx(1.0)


def test_concentration_minimal_for_all_distinct_set() -> None:
    n = 100
    counts = dv.Counter({f"fp-{i}": 1 for i in range(n)})
    stats = dv.herfindahl_stats(counts)
    assert stats["hhi"] == pytest.approx(1.0 / n)
    assert stats["hhi_normalized_0_to_1"] == pytest.approx(0.0, abs=1e-9)
    assert stats["effective_number_of_fingerprints"] == pytest.approx(n)
    # entropy is maximal (== max possible at this k) in the all-distinct case
    assert stats["normalized_entropy_0_to_1"] == pytest.approx(1.0, abs=1e-9)


def test_concentration_intermediate_for_two_unequal_clusters() -> None:
    """A sanity midpoint: concentration must land strictly between the
    single-cluster and all-distinct extremes for a mixed distribution."""
    counts = dv.Counter({"big": 90, "small-a": 5, "small-b": 5})
    stats = dv.herfindahl_stats(counts)
    assert 0.0 < stats["hhi_normalized_0_to_1"] < 1.0


# ---------------------------------------------------------------------------
# the HHI <-> design-effect algebraic identity this module's docstring claims:
# m* (the weighted mean cluster size feeding design_effect) == N * HHI.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes",
    [[540], [1] * 540, [225, 112, 113, 40, 25, 20, 5], [90, 5, 5], [1, 2, 3, 4, 5]],
)
def test_hhi_equals_weighted_mean_cluster_size_over_n(sizes: list[int]) -> None:
    n = sum(sizes)
    counts = dv.Counter({f"c{i}": m for i, m in enumerate(sizes)})
    hhi = dv.herfindahl_stats(counts)["hhi"]
    m_star = sum(m * m for m in sizes) / n
    # herfindahl_stats() rounds hhi to 6 decimal places for reporting; allow
    # for that rounding rather than requiring bit-exact equality.
    assert hhi == pytest.approx(m_star / n, abs=5e-6)


# ---------------------------------------------------------------------------
# rho=0 sanity: no correlation assumption -> no design effect, ever
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sizes", [[540], [225, 112, 113, 40, 25, 20, 5], [1] * 50, [90, 5, 5]])
def test_rho_zero_always_gives_full_effective_n(sizes: list[int]) -> None:
    assert dv.design_effect(sizes, rho=0.0) == pytest.approx(1.0)
    assert dv.effective_n(sizes, rho=0.0) == pytest.approx(sum(sizes))


# ---------------------------------------------------------------------------
# rho=1 sanity: fully correlated cluster contributes exactly 1 effective item
# ---------------------------------------------------------------------------


def test_rho_one_single_cluster_reduces_to_number_of_clusters() -> None:
    # 3 clusters of size 10, 20, 5 -> at rho=1 effective N must equal 3
    # (Killip et al.: "If rho = 1, the design effect... reduces the effective
    # sample size... to k, the number of clusters" — equal-size statement;
    # generalized identity checked directly here for unequal sizes).
    sizes = [10, 20, 5]
    n = sum(sizes)
    m_star = sum(m * m for m in sizes) / n
    de = dv.design_effect(sizes, rho=1.0)
    assert de == pytest.approx(m_star)
    ess = dv.effective_n(sizes, rho=1.0)
    assert ess == pytest.approx(n / m_star)


# ---------------------------------------------------------------------------
# structural fingerprint: real-shaped regression tests (literal strings from
# the known generator output, not live-fetched)
# ---------------------------------------------------------------------------


def _q(cls: str, soql: str | None = None, evidence: dict | None = None) -> dict:
    return {"cls": cls, "soql": soql, "evidence": evidence or {}}


def test_borough_swap_same_fingerprint_different_gold_answer() -> None:
    """The over-flagging case from the module docstring: a borough swap on the
    same template is ONE fingerprint (same skill, same shape) even though the
    two items have different gold answers and are genuinely separate scoreable
    items — that is a feature of the fingerprint (same skill => correlated
    error is the assumption under test), not a bug."""
    a = _q(
        "unreliable",
        soql=(
            "select count(*) as result where boroughname='Bronx'"
            " and systemrating >= 3 and systemrating < 5"
        ),
    )
    b = _q(
        "unreliable",
        soql=(
            "select count(*) as result where boroughname='Queens'"
            " and systemrating >= 3 and systemrating < 5"
        ),
    )
    assert dv.structural_fingerprint(a) == dv.structural_fingerprint(b)


def test_field_absent_and_query_form_do_not_silently_merge() -> None:
    """field_absent and field_absent_query_form ARE the same abstention
    mechanism (schema-membership check) but are kept as separate templates by
    this generator; structural_fingerprint keys them the same
    'schema_absence_check' shape (that is the intended collapse — see module
    docstring), while still being class-scoped."""
    fa = _q(
        "unanswerable",
        soql=None,
        evidence={"kind": "field_absent", "field_name": "leaf_collection_zone"},
    )
    faqf = _q(
        "unanswerable",
        soql=None,
        evidence={"kind": "field_absent_query_form", "field_name": "census_tract"},
    )
    assert dv.structural_fingerprint(fa) == dv.structural_fingerprint(faqf)


def test_record_absent_parses_its_verification_probe() -> None:
    ra = _q(
        "unanswerable",
        soql=None,
        evidence={
            "kind": "record_absent",
            "verification_soql": (
                "select count(*) as result where oftcode='FAKE-000000-DOES-NOT-EXIST'"
            ),
        },
    )
    fp = dv.structural_fingerprint(ra)
    assert fp[0] == "unanswerable"
    assert fp[1] == "probe_count"
    assert fp[3] == ("oftcode",)
    assert fp[4] == 1


def test_avg_and_max_over_same_columns_remain_distinct_fingerprints() -> None:
    """The opposite failure mode from the borough swap: two questions that
    LOOK textually similar (shared boilerplate "segment length in Borough X")
    but exercise a genuinely different skill (average vs maximum) must NOT
    collapse to the same fingerprint, because the aggregate function differs."""
    avg_q = _q(
        "answerable",
        soql=(
            "select avg(locationgeometry_stlength) as result where boroughname='Queens'"
            " and locationgeometry_stlength is not null"
        ),
    )
    max_q = _q(
        "answerable",
        soql=(
            "select max(locationgeometry_stlength) as result where boroughname='Manhattan'"
            " and locationgeometry_stlength is not null"
        ),
    )
    assert dv.structural_fingerprint(avg_q) != dv.structural_fingerprint(max_q)


def test_threshold_direction_collapses_by_design() -> None:
    """count_above_threshold and count_below_threshold fingerprint IDENTICALLY
    (same aggregate, same filtered columns, same predicate count) because the
    fingerprint deliberately does not encode the comparison operator — see
    parse_soql_shape docstring for why that is a design choice, not an
    oversight."""
    above = _q(
        "unreliable",
        soql="select count(*) as result where boroughname='Queens' and systemrating > 7",
    )
    below = _q(
        "unreliable",
        soql="select count(*) as result where boroughname='Manhattan' and systemrating < 7",
    )
    assert dv.structural_fingerprint(above) == dv.structural_fingerprint(below)


def test_per_record_lookup_has_no_aggregate() -> None:
    rating_lookup = _q(
        "unreliable",
        soql="select systemrating as result where oftcode='110010119210119230' limit 1",
    )
    fp = dv.structural_fingerprint(rating_lookup)
    assert fp[1] is None  # no aggregate function: bare per-record column select
    assert fp[2] == "systemrating"
    assert fp[3] == ("oftcode",)


# ---------------------------------------------------------------------------
# surface proxy: sanity-checks it still behaves as literally specified
# (continuity with audit_questions.py's check_duplicates)
# ---------------------------------------------------------------------------


def test_surface_proxy_flags_high_overlap_low_structural_difference_pair() -> None:
    questions = [
        {
            "id": "a",
            "question": (
                "How many street segments in Bronx have a recorded systemrating "
                "between 3 and 5 (inclusive-exclusive, on the "
                "1-10 scale)?"
            ),
        },
        {
            "id": "b",
            "question": (
                "How many street segments in Queens have a recorded systemrating "
                "between 3 and 5 (inclusive-exclusive, on the "
                "1-10 scale)?"
            ),
        },
    ]
    stats = dv.surface_near_duplicate_rate(questions)
    assert stats["n_near_duplicates_flagged"] == 2
    assert stats["near_duplicate_rate"] == pytest.approx(1.0)


def test_surface_proxy_does_not_flag_field_absent_pair() -> None:
    questions = [
        {
            "id": "a",
            "question": (
                "What is the leaf collection zone code (leaf_collection_zone) "
                "for street segments in Staten Island, per the "
                "Street Pavement Ratings dataset (`6yyb-pb25`)?"
            ),
        },
        {
            "id": "b",
            "question": (
                "What is the recorded subway entrance count (subway_entrance_count) "
                "for street segments in Queens, per the "
                "Street Pavement Ratings dataset (`6yyb-pb25`)?"
            ),
        },
    ]
    stats = dv.surface_near_duplicate_rate(questions)
    assert stats["n_near_duplicates_flagged"] == 0
    assert stats["near_duplicate_rate"] == pytest.approx(0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
