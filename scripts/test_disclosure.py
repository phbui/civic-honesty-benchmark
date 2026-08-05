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
