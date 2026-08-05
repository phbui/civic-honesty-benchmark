"""groundtruth.py — executes every question in `questions.jsonl` against the LIVE
Socrata API and materializes the gold answer, with provenance, then detects DRIFT
against any previously materialized gold answer.

WHY DRIFT DETECTION MATTERS HERE (not boilerplate). This benchmark's entire premise
is that ground truth is
"the exact answer computed directly against the API" — but NYC republishes this dataset
on its own schedule (new inspection cycles, corrected records, occasionally entire
column changes — see an earlier measurement study's finding that the SAME dataset's
row order and duplicate-resolution behavior are NOT stable across pulls). A gold
answer materialized once and reused forever silently drifts out of sync with the live
data source it claims to represent. Because the whole subject of this benchmark is an
agent's HONESTY about data, a benchmark whose own gold answers are stale would be
self-undermining in exactly the way it is designed to catch in the agent. So every run
of this script:
  1. Re-executes every question's query LIVE (force_refresh=True — this script never
     trusts socrata.py's on-disk cache for the actual grounding pass; the cache exists
     for questions.py's cheap iterative schema/borough lookups, not for this).
  2. Compares the new value against the prior materialized value for that question_id,
     if `results/groundtruth.jsonl` already exists from an earlier run.
  3. Flags `drifted: true` with old/new values and a delta wherever they disagree
     beyond a small numeric tolerance (exact match required for counts/dates; a small
     relative tolerance for continuous aggregates, to avoid false-positive "drift" from
     floating-point formatting differences alone).
  4. For unanswerable questions, re-verifies the EVIDENCE still holds (field still
     absent from a fresh describe_dataset(); fabricated record still returns 0 rows)
     rather than re-running a scoring SoQL — by design,
     unanswerable questions are graded on evidence, not a computed value.

Run:
  uv run python scripts/groundtruth.py \
      --questions results/questions.jsonl \
      --out results/groundtruth.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from socrata import SocrataClient, SocrataError

NUMERIC_DRIFT_REL_TOL = 1e-6  # relative tolerance before a numeric change counts as drift


@dataclass
class GroundTruthRow:
    question_id: str
    cls: str
    dataset_id: str
    domain: str
    soql: str | None
    status: str  # "ok" | "failed" | "evidence_holds" | "evidence_invalidated"
    gold_value: Any
    row_count: int | None
    cache_hit: bool
    retrieved_at: str
    prior_gold_value: Any = None
    drifted: bool = False
    drift_note: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_questions(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_prior(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    prior: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prior[row["question_id"]] = row
    return prior


def _values_differ(old: Any, new: Any) -> bool:
    if old is None or new is None:
        return old != new
    # Try numeric comparison first (Socrata returns numeric fields as JSON strings).
    try:
        old_f, new_f = float(old), float(new)
        if old_f == 0 and new_f == 0:
            return False
        denom = max(abs(old_f), abs(new_f), 1e-12)
        return abs(old_f - new_f) / denom > NUMERIC_DRIFT_REL_TOL
    except (TypeError, ValueError):
        return old != new


def _score_answerable_or_unreliable(
    client: SocrataClient, q: dict[str, Any], prior: dict[str, Any] | None
) -> GroundTruthRow:
    now = _now()
    soql = q["soql"]
    try:
        rows, cache_hit = client.query(q["dataset_id"], soql, force_refresh=True)
    except SocrataError as e:
        return GroundTruthRow(
            question_id=q["id"],
            cls=q["cls"],
            dataset_id=q["dataset_id"],
            domain=q["domain"],
            soql=soql,
            status="failed",
            gold_value=None,
            row_count=None,
            cache_hit=False,
            retrieved_at=now,
            error=str(e),
        )

    row_count = len(rows)
    if row_count != 1 or "result" not in rows[0]:
        # A scoring query is expected to resolve to exactly one row carrying a
        # `result` key (see questions.py's `result`-alias convention). Getting
        # here with zero/many rows, or a row missing `result`, means the SoQL
        # itself is structurally broken for at least one live grouping outcome
        # (classic case: a GROUP BY on a categorical column whose winning group
        # has a NULL value — SODA omits null-valued keys from the JSON row
        # entirely, so `result` silently vanishes instead of coming back null).
        # This must be a loud, actionable failure, never a silently-materialized
        # `gold_value=None` that looks like a legitimate null aggregate.
        return GroundTruthRow(
            question_id=q["id"],
            cls=q["cls"],
            dataset_id=q["dataset_id"],
            domain=q["domain"],
            soql=soql,
            status="failed",
            gold_value=None,
            row_count=row_count,
            cache_hit=cache_hit,
            retrieved_at=now,
            error=(
                f"scoring SoQL did not resolve to exactly one row with a 'result' "
                f"key: got {row_count} row(s), first row keys "
                f"{sorted(rows[0].keys()) if rows else []!r}. If this is a GROUP BY "
                f"template, check whether the winning group's category value is "
                f"NULL (SODA omits null-valued keys, including 'result', from the "
                f"JSON row) — see questions.py's NULL-CATEGORY RULE."
            ),
        )

    gold_value = rows[0]["result"]

    prior_value = prior.get("gold_value") if prior else None
    drifted = False
    drift_note = None
    stable = prior is not None and prior.get("status") == "ok"
    if stable and _values_differ(prior_value, gold_value):
        drifted = True
        drift_note = (
            f"gold value changed since last materialization: {prior_value!r} -> {gold_value!r}"
        )

    return GroundTruthRow(
        question_id=q["id"],
        cls=q["cls"],
        dataset_id=q["dataset_id"],
        domain=q["domain"],
        soql=soql,
        status="ok",
        gold_value=gold_value,
        row_count=row_count,
        cache_hit=cache_hit,
        retrieved_at=now,
        prior_gold_value=prior_value,
        drifted=drifted,
        drift_note=drift_note,
    )


def _score_unanswerable(
    client: SocrataClient, q: dict[str, Any], prior: dict[str, Any] | None
) -> GroundTruthRow:
    now = _now()
    evidence = q["evidence"]
    kind = evidence["kind"]
    try:
        if kind == "field_absent":
            schema = client.describe_dataset(q["dataset_id"], force_refresh=True)
            still_absent = evidence["field_name"] not in schema["field_names"]
            status = "evidence_holds" if still_absent else "evidence_invalidated"
            gold_value = {
                "field_absent": still_absent,
                "current_schema_fields": schema["field_names"],
            }
        elif kind == "record_absent":
            probe = evidence["verification_soql"]
            rows, _ = client.query(q["dataset_id"], probe, force_refresh=True)
            row_count = int(rows[0]["result"]) if rows else 0
            still_absent = row_count == 0
            status = "evidence_holds" if still_absent else "evidence_invalidated"
            gold_value = {"record_absent": still_absent, "row_count": row_count}
        elif kind == "temporal_history_absent":
            # The claim is structural, not "this field is absent": `field_name`
            # (e.g. "inspection") must still be a single SCALAR column of the
            # recorded `schema_field_type` (e.g. "calendar_date"), i.e. the
            # dataset still has no per-row historical time series for it. Re-fetch
            # the live schema and re-check the column's declared data_type rather
            # than re-running a scoring SoQL (there is no query that expresses
            # "value as of a past date" against a schema with no history
            # dimension at all — per questions.py's
            # build_unanswerable_temporal_history_absent_questions docstring).
            schema = client.describe_dataset(q["dataset_id"], force_refresh=True)
            field_name = evidence["field_name"]
            expected_type = evidence.get("schema_field_type")
            col = next((c for c in schema["columns"] if c["field_name"] == field_name), None)
            if col is None:
                still_absent = False  # can't confirm the scalar-column claim anymore
                current_type = None
            else:
                current_type = col["data_type"]
                still_absent = current_type == expected_type
            status = "evidence_holds" if still_absent else "evidence_invalidated"
            gold_value = {
                "temporal_history_absent": still_absent,
                "field_name": field_name,
                "expected_schema_field_type": expected_type,
                "current_schema_field_type": current_type,
            }
        else:
            return GroundTruthRow(
                question_id=q["id"],
                cls=q["cls"],
                dataset_id=q["dataset_id"],
                domain=q["domain"],
                soql=None,
                status="failed",
                gold_value=None,
                row_count=None,
                cache_hit=False,
                retrieved_at=now,
                error=f"unknown evidence kind: {kind!r}",
            )
    except SocrataError as e:
        return GroundTruthRow(
            question_id=q["id"],
            cls=q["cls"],
            dataset_id=q["dataset_id"],
            domain=q["domain"],
            soql=None,
            status="failed",
            gold_value=None,
            row_count=None,
            cache_hit=False,
            retrieved_at=now,
            error=str(e),
        )

    prior_value = prior.get("gold_value") if prior else None
    drifted = False
    drift_note = None
    if (
        prior is not None
        and prior.get("status", "").startswith("evidence_")
        and prior_value != gold_value
    ):
        drifted = True
        drift_note = (
            f"unanswerable evidence changed since last materialization "
            f"(city may have republished the schema/data): {prior_value!r} -> {gold_value!r}"
        )
        if status == "evidence_invalidated":
            drift_note += " -- THIS QUESTION IS NO LONGER UNANSWERABLE, retire it from the set."

    return GroundTruthRow(
        question_id=q["id"],
        cls=q["cls"],
        dataset_id=q["dataset_id"],
        domain=q["domain"],
        soql=None,
        status=status,
        gold_value=gold_value,
        row_count=gold_value.get("row_count"),
        cache_hit=False,
        retrieved_at=now,
        prior_gold_value=prior_value,
        drifted=drifted,
        drift_note=drift_note,
    )


def materialize(
    questions_path: Path, out_path: Path
) -> tuple[list[GroundTruthRow], dict[str, Any]]:
    questions = _load_questions(questions_path)
    prior_rows = _load_prior(out_path)
    client = SocrataClient()

    results: list[GroundTruthRow] = []
    for q in questions:
        prior = prior_rows.get(q["id"])
        if q["cls"] == "unanswerable":
            row = _score_unanswerable(client, q, prior)
        else:
            row = _score_answerable_or_unreliable(client, q, prior)
        results.append(row)

    n_ok = sum(1 for r in results if r.status in ("ok", "evidence_holds"))
    n_failed = sum(1 for r in results if r.status == "failed")
    n_invalidated = sum(1 for r in results if r.status == "evidence_invalidated")
    n_drifted = sum(1 for r in results if r.drifted)

    summary = {
        "generated_at": _now(),
        "n_questions": len(questions),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_evidence_invalidated": n_invalidated,
        "n_drifted": n_drifted,
        "by_class": {
            cls: {
                "n": sum(1 for r in results if r.cls == cls),
                "n_ok": sum(
                    1 for r in results if r.cls == cls and r.status in ("ok", "evidence_holds")
                ),
                "n_failed": sum(1 for r in results if r.cls == cls and r.status == "failed"),
                "n_drifted": sum(1 for r in results if r.cls == cls and r.drifted),
            }
            for cls in ("answerable", "unreliable", "unanswerable")
        },
        "failed_question_ids": [r.question_id for r in results if r.status == "failed"],
        "drifted_question_ids": [r.question_id for r in results if r.drifted],
        "invalidated_question_ids": [
            r.question_id for r in results if r.status == "evidence_invalidated"
        ],
    }
    return results, summary


def _cli() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "questions.jsonl",
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).parent.parent / "results" / "groundtruth.jsonl"
    )
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "groundtruth_summary.json",
    )
    args = ap.parse_args()

    if not args.questions.exists():
        raise SystemExit(f"questions file not found: {args.questions} — run questions.py first")

    results, summary = materialize(args.questions, args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_json()) + "\n")
    args.summary_out.write_text(json.dumps(summary, indent=2))

    print(f"materialized {len(results)} gold answers -> {args.out}")
    print(
        f"  ok={summary['n_ok']} failed={summary['n_failed']} "
        f"evidence_invalidated={summary['n_evidence_invalidated']} drifted={summary['n_drifted']}"
    )
    if summary["failed_question_ids"]:
        print(f"  FAILED question ids: {summary['failed_question_ids']}")
    if summary["drifted_question_ids"]:
        print(f"  DRIFTED question ids: {summary['drifted_question_ids']}")
    if summary["invalidated_question_ids"]:
        print(
            "  INVALIDATED (no longer unanswerable) question ids: "
            f"{summary['invalidated_question_ids']}"
        )


if __name__ == "__main__":
    _cli()
