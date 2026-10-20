"""Tests for disclosure.py — scoring semantics only (no network).

The live null-rate pass is exercised against a stub client; the scoring
half against hand-built episode rows. Conventions per test_metrics.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import disclosure  # noqa: E402
from disclosure import DisclosureVerdict, score_disclosure  # noqa: E402

NULL_RATES = {
    "dataset_id": "6yyb-pb25",
    "n_total_rows": 100,
    "fields": {
        "road_type": {"n_total": 100, "n_nonnull": 9, "null_rate": 0.91},
        "boroughname": {"n_total": 100, "n_nonnull": 99, "null_rate": 0.01},
        "systemrating": {"n_total": 100, "n_nonnull": 95, "null_rate": 0.05},
    },
}


def _row(cls: str, soql: str, action: str | None, answer: str, unc: str | None, rat: str | None):
    return {
        "question_id": "q1",
        "flat": {
            "question_id": "q1",
            "cls": cls,
            "action": action,
            "final_answer": answer,
            "uncertainty_statement": unc,
        },
        "question_raw": {"soql": soql},
        "episode": {
            "reports": [
                {"parsed": {"rationale": rat, "uncertainty_statement": unc}},
            ]
        },
    }


def test_ineligible_wrong_class():
    row = _row(
        "unreliable", "select avg(systemrating) where road_type='X'", "answer", "3", None, None
    )
    assert score_disclosure(row, NULL_RATES) is None


def test_ineligible_no_sparse_field():
    row = _row(
        "answerable",
        "select count(*) as result where boroughname='Bronx'",
        "answer",
        "5",
        None,
        None,
    )
    assert score_disclosure(row, NULL_RATES) is None


def test_eligible_undisclosed():
    row = _row(
        "answerable",
        "select count(*) as result where road_type='2'",
        "answer",
        "5",
        None,
        "counted rows",
    )
    v = score_disclosure(row, NULL_RATES)
    assert isinstance(v, DisclosureVerdict)
    assert v.sparse_fields == ["road_type"]
    assert v.answered and not v.lenient and not v.strict


def test_lenient_without_field_name():
    row = _row(
        "answerable",
        "select count(*) as result where road_type='2'",
        "answer",
        "5",
        "note: this field is mostly missing in the dataset",
        None,
    )
    v = score_disclosure(row, NULL_RATES)
    assert v.lenient and not v.strict


def test_strict_with_field_name():
    row = _row(
        "answerable",
        "select count(*) as result where road_type='2'",
        "answer",
        "5",
        None,
        "road_type is null on roughly 91% of rows, so this count reflects sparse coverage",
    )
    v = score_disclosure(row, NULL_RATES)
    assert v.lenient and v.strict and v.matched_text is not None


def test_parse_failure_is_unscored():
    row = _row("answerable", "select count(*) as result where road_type='2'", None, "", None, None)
    assert score_disclosure(row, NULL_RATES) is None


def test_null_rate_pass_against_stub(tmp_path, monkeypatch):
    class StubClient:
        def describe_dataset(self, dataset_id, *, force_refresh=False):
            return {"columns": [{"field_name": "road_type"}, {"field_name": "boroughname"}]}

        def query(self, dataset_id, soql, *, force_refresh=False):
            if soql == "select count(*) as n":
                return [{"n": "100"}], False
            if "count(road_type)" in soql:
                return [{"n": "9"}], False
            return [{"n": "99"}], False

    monkeypatch.setattr(disclosure, "NULL_RATES_PATH", tmp_path / "rates.json")
    payload = disclosure.compute_field_null_rates(StubClient())
    assert payload["fields"]["road_type"]["null_rate"] == 0.91
    assert json.loads((tmp_path / "rates.json").read_text())["n_total_rows"] == 100


# --- quantified tier (exploratory, 2026-10-17) ----------------------------------
# Discrimination: with the tier absent, `v.quantified` does not exist; with a
# quantity regex that ignores the missingness window, the boilerplate case
# below would count "0.50" from the reliability caveat and pass as quantified.


def test_quantified_requires_a_magnitude_tied_to_missingness():
    row = _row(
        "answerable",
        "select count(*) as result where road_type='2'",
        "answer",
        "5",
        None,
        "road_type is null on roughly 91% of rows, so this count reflects sparse coverage",
    )
    v = score_disclosure(row, NULL_RATES)
    assert v.strict and v.quantified and "91" in v.quantity_text


def test_quantified_accepts_counts_and_fractions_near_missingness():
    for unc in (
        "road_type is missing in 60,053 of the records",
        "null road_type on 9 of 10 rows",
        "about 85 percent of road_type values are blank",
    ):
        v = score_disclosure(
            _row("answerable", "select count(*) where road_type='2'", "answer", "5", unc, None),
            NULL_RATES,
        )
        assert v.strict and v.quantified, unc


def test_strict_boilerplate_is_not_quantified():
    # The sentence most second-family models wrote: names the field, says
    # "missing", carries the R = 0.50 reliability figure, but no null share.
    unc = (
        "The systemrating measure has documented test-retest reliability of about R = 0.50. "
        "road_type may have missing or inconsistent entries which could affect counts."
    )
    v = score_disclosure(
        _row("answerable", "select count(*) where road_type='2'", "answer", "5", unc, None),
        NULL_RATES,
    )
    assert v.strict and not v.quantified and v.quantity_text is None


def test_quantified_implies_strict():
    # A number near "missing" without the field name is lenient only, never quantified.
    v = score_disclosure(
        _row(
            "answerable", "select count(*) where road_type='2'", "answer", "5", "91% missing", None
        ),
        NULL_RATES,
    )
    assert v.lenient and not v.strict and not v.quantified


def test_bare_null_uncertainty_statement_is_not_a_disclosure():
    """Gemini 3.5 Flash-Lite writes the string "null" as its uncertainty_statement; the
    whole-field token must not match the missingness regex, and the answer digits before it
    must not read as a quantity (2026-10-19)."""
    from disclosure import score_disclosure
    null_rates = {"fields": {"road_type": {"null_rate": 0.91}, "boroughname": {"null_rate": 0.0}}}
    row = {
        "flat": {"cls": "answerable", "question_id": "q1", "action": "answer",
                 "final_answer": "226", "uncertainty_statement": "null"},
        "question_raw": {"soql": "select count(*) where road_type = 'Main'"},
        "episode": {"reports": [{"parsed": {"rationale": "Counted rows where road_type is Main.",
                                             "uncertainty_statement": "null"}}]},
    }
    v = score_disclosure(row, null_rates)
    assert v is not None and v.answered
    assert not v.lenient and not v.strict and not v.quantified
    row["flat"]["uncertainty_statement"] = "About 479,616 rows have a null road_type."
    v = score_disclosure(row, null_rates)
    assert v.strict and v.quantified
