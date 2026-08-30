"""Unit tests for measure_reliability.py, validated against synthetic fixtures where
the correct answer is known analytically, in the same spirit as test_metrics.py. No
live API calls anywhere in this file. Run explicitly:

    uv run pytest scripts/test_measure_reliability.py -v
"""

from __future__ import annotations

import math

import measure_reliability as mr


def row(seg: str, rating: float, day: str) -> dict:
    return {"oftcode": seg, "systemrating": str(rating), "inspection": f"{day}T00:00:00.000"}


# --------------------------------------------------------------------------- #
# pearson
# --------------------------------------------------------------------------- #


def test_pearson_perfect_positive():
    assert mr.pearson([(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]) == 1.0


def test_pearson_perfect_negative():
    assert mr.pearson([(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]) == -1.0


def test_pearson_constant_margin_is_nan():
    assert math.isnan(mr.pearson([(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)]))


def test_pearson_below_two_pairs_is_nan():
    assert math.isnan(mr.pearson([(1.0, 2.0)]))


# --------------------------------------------------------------------------- #
# collapse_occasions
# --------------------------------------------------------------------------- #


def test_same_date_rows_collapse_to_mean():
    occs = mr.collapse_occasions([row("A", 4.0, "2020-01-01"), row("A", 6.0, "2020-01-01")])
    assert len(occs["A"]) == 1
    assert occs["A"][0][1] == 5.0


def test_occasions_sorted_by_date_regardless_of_input_order():
    occs = mr.collapse_occasions([row("A", 7.0, "2021-06-01"), row("A", 9.0, "2020-06-01")])
    assert [r for _, r in occs["A"]] == [9.0, 7.0]


def test_unparseable_rows_are_skipped_not_fatal():
    rows = [row("A", 5.0, "2020-01-01"), {"oftcode": "A", "systemrating": "n/a"}]
    occs = mr.collapse_occasions(rows)
    assert len(occs["A"]) == 1


# --------------------------------------------------------------------------- #
# consecutive_pairs and the year band
# --------------------------------------------------------------------------- #


def test_three_occasions_make_two_consecutive_pairs():
    occs = mr.collapse_occasions(
        [row("A", 8.0, "2020-01-01"), row("A", 7.0, "2021-01-01"), row("A", 6.0, "2022-01-01")]
    )
    all_pairs, band_pairs, gaps = mr.consecutive_pairs(occs)
    assert all_pairs == [(8.0, 7.0), (7.0, 6.0)]
    assert band_pairs == all_pairs  # 365/366-day gaps sit inside 180-730
    assert gaps == [366, 365]


def test_year_band_excludes_short_and_long_gaps():
    occs = mr.collapse_occasions(
        [row("A", 8.0, "2020-01-01"), row("A", 7.0, "2020-02-01"), row("A", 6.0, "2024-02-01")]
    )
    all_pairs, band_pairs, _ = mr.consecutive_pairs(occs)
    assert len(all_pairs) == 2  # 31-day gap and a 4-year gap
    assert band_pairs == []


# --------------------------------------------------------------------------- #
# test_retest end to end on a fixture with a known answer
# --------------------------------------------------------------------------- #


def test_test_retest_recovers_known_correlation():
    # Two segments, each with two occasions one year apart; second ratings are an
    # exact linear function of the first plus a third segment breaking degeneracy,
    # so all-pairs Pearson is exactly 1.0.
    rows = [
        row("A", 2.0, "2020-01-01"), row("A", 4.0, "2021-01-01"),
        row("B", 3.0, "2020-01-01"), row("B", 6.0, "2021-01-01"),
        row("C", 5.0, "2020-01-01"), row("C", 10.0, "2021-01-01"),
    ]
    out = mr.test_retest(rows)
    assert out["segments"] == 3
    assert out["segments_with_2plus_occasions"] == 3
    assert out["consecutive_pairs"] == 3
    assert out["pearson_r_all_pairs"] == 1.0
    assert out["pearson_r_180_730d"] == 1.0
    assert out["median_gap_days"] == 366


def test_test_retest_single_occasion_segments_contribute_no_pairs():
    rows = [row("A", 2.0, "2020-01-01"), row("B", 3.0, "2020-01-01")]
    out = mr.test_retest(rows)
    assert out["segments"] == 2
    assert out["segments_with_2plus_occasions"] == 0
    assert out["consecutive_pairs"] == 0
    assert out["pearson_r_all_pairs"] is None
