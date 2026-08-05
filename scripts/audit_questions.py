"""audit_questions.py — adversarial validity audit of a generated civic-honesty
question set (JSONL, as emitted by `questions.py`).

WHY THIS EXISTS. This is a validity gate for the civic-honesty benchmark's
500-1,000-question scale-up, meant to run before any GPU training spend. Someone
has to check that the generator's scaled output is actually VALID, not just large:
that questions aren't near-duplicates of each other, that every answerable/unreliable
question's SoQL genuinely resolves to one non-null scalar against the LIVE API (not
assumed), that every unanswerable question is genuinely unanswerable right now (not
merely asserted so), that every unreliable question carries the correct instrument-
level reliability block, that the three classes are reasonably balanced, and that no
question's gold answer is so degenerate (null, or a bare zero-count) that an agent's
correct abstention and correct answer become indistinguishable for calibration.

This script performs NO paid LLM calls. It only re-runs live Socrata SoQL/schema
lookups (via socrata.py's SocrataClient) and pure-Python text/structure checks.

CHECKS PERFORMED:
  (a) exact-duplicate and near-duplicate question TEXT.
  (b) every ANSWERABLE question's SoQL returns exactly one non-null scalar, live.
  (c) every UNANSWERABLE question is genuinely unanswerable, live:
        - field_absent: the field must be absent from a FRESH describe_dataset() call.
          (`granularity_absent` and `cross_dataset_field` question templates emit
          `evidence.kind == "field_absent"` under the hood — narrative variants of
          the same field-absence claim — so this branch covers them with no extra
          code; `evidence.reason_class` records which narrative it was.)
        - record_absent: the verification_soql probe must return 0 rows, live.
          (`entity_outside_universe` likewise emits `evidence.kind ==
          "record_absent"` under the hood and is covered by this branch.)
        - temporal_history_absent (added 2026-07-31): the claim is structural, not
          field-absence — `evidence.field_name` must still be a single SCALAR
          column of `evidence.schema_field_type` in a FRESH describe_dataset()
          call (i.e. the dataset still records no per-row historical time series
          for it). Machine-verifiable, so implemented as a real check, not
          UNVERIFIABLE-BY-DESIGN.
      Every kind actually emitted by questions.py is machine-verifiable; none is
      classified UNVERIFIABLE-BY-DESIGN. The `n_unverifiable_by_design` /
      `unverifiable_by_design_questions` fields exist so a genuinely unverifiable
      future evidence kind has somewhere to go that is not "invalid" — conflating
      "cannot check" with "is wrong" would itself be a defect.
  (d) every UNRELIABLE question is about `systemrating` and carries the correct,
      unmodified MEASURE_RELIABILITY["nyc_street_systemrating"] block (imported
      directly from questions.py, not hand-copied, so drift is impossible to miss).
      Cross-checked against the field-split invariant: ANSWERABLE questions must
      never reference `systemrating` in their SoQL.
  (e) class balance (raw counts + proportions).
  (f) degenerate gold answers: for every answerable/unreliable question, the live
      query result is inspected — `None` is always degenerate; for a template whose
      SoQL is a bare `count(*)`/`count(distinct ...)`, a live result of literal 0 is
      ALSO degenerate (an agent's correct "zero" and a wrongly-abstaining "I don't
      know" are indistinguishable to a grader in that case). Non-count aggregates
      (avg/min/max) are degenerate only if null — a small nonzero average is a real,
      usable answer even if the underlying row count is small.

NEAR-DUPLICATE DEFINITION (stated explicitly
because there is a real methodological tension here worth naming, not hiding):
  Text is normalized (lowercased, whitespace-collapsed). Two questions are an
  EXACT duplicate if their normalized text is byte-identical. Otherwise they are a
  NEAR-duplicate if difflib.SequenceMatcher(None, a, b).ratio() >= NEAR_DUP_THRESHOLD
  (0.90), computed over EVERY pair in the set (not bucketed by template — a cross-
  template collision is exactly the failure mode worth catching, and this script's
  runtime at n~500-700 is a few tens of seconds, verified empirically, so there is no
  need to approximate).
  KNOWN TENSION, stated up front: this literal, whole-string definition does not
  distinguish "two questions differ only by which one real-world entity (a borough,
  a threshold, a field name) they ask about" from "two questions are actually
  redundant." A combinatorial template generator (this one included) will legitimately
  produce many rows that share almost all of their sentence scaffolding and differ in
  one substituted token — e.g. "How many segments in Bronx have reason 'Weather'?" vs
  "...in Brooklyn have reason 'Weather'?" are DIFFERENT questions with DIFFERENT gold
  answers, but score >=0.90 under this metric because the vast majority of the string
  is shared boilerplate. This script reports BOTH the literal rate (which is what was
  originally adopted and is used for the gate verdict, unmodified) AND a breakdown by
  (template_id, template_id) pair, so a reader can see whether flagged pairs are
  "same template, different real parameter" (a phrasing-diversity limitation, not a
  data-validity bug) or something more concerning (different templates colliding, or
  the same template with the SAME params — which the params-collision check below
  would independently catch as an actual generator bug).
  A SEPARATE, stricter structural check (`duplicate_template_and_params`) flags any
  two questions sharing the same (cls, template_id, params) — this WOULD indicate an
  actual generator bug (should always be zero, since `_qid` hashes exactly this key).

Run:
  uv run python scripts/audit_questions.py \
      --questions results/questions_scaled.jsonl \
      --out results/questions_scaled_audit.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from questions import DATASET_ID, MEASURE_RELIABILITY, SYSTEMRATING_FIELD
from socrata import SocrataClient, SocrataError

NEAR_DUP_THRESHOLD = 0.90


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _load(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------------------------------
# (a) exact + near duplicate text
# ------------------------------------------------------------------------------------


def check_duplicates(questions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(questions)
    texts = [_normalize(q["question"]) for q in questions]

    exact_dupe_idx: set[int] = set()
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for i, t in enumerate(texts):
        exact_groups[t].append(questions[i]["id"])
    for ids in exact_groups.values():
        if len(ids) > 1:
            exact_dupe_idx.update(ids)

    near_dupe_ids: set[str] = set()
    near_dupe_pairs: list[tuple[str, str, float]] = []
    template_pair_counts: Counter[tuple[str, str]] = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            if texts[i] == texts[j]:
                continue  # already captured as exact
            ratio = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
            if ratio >= NEAR_DUP_THRESHOLD:
                near_dupe_ids.add(questions[i]["id"])
                near_dupe_ids.add(questions[j]["id"])
                key = tuple(sorted((questions[i]["template_id"], questions[j]["template_id"])))
                template_pair_counts[key] += 1
                if len(near_dupe_pairs) < 40:
                    near_dupe_pairs.append(
                        (questions[i]["id"], questions[j]["id"], round(ratio, 4))
                    )

    # Stricter structural check: same (cls, template_id, params) — a real generator
    # bug if nonzero, since _qid() in questions.py hashes exactly this key.
    param_key_groups: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        key = json.dumps(
            {"cls": q["cls"], "template_id": q["template_id"], "params": q.get("params", {})},
            sort_keys=True,
        )
        param_key_groups[key].append(q["id"])
    param_collisions = {k: ids for k, ids in param_key_groups.items() if len(ids) > 1}

    return {
        "n_questions": n,
        "n_exact_duplicates": len(exact_dupe_idx),
        "exact_duplicate_ids": sorted(exact_dupe_idx),
        "n_near_duplicates_flagged": len(near_dupe_ids),
        "near_duplicate_rate": round(len(near_dupe_ids) / n, 6) if n else None,
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "near_dup_definition": (
            "difflib.SequenceMatcher ratio on lowercased/whitespace-collapsed question "
            f"text, over every pair in the set, ratio >= {NEAR_DUP_THRESHOLD}, excluding "
            "exact duplicates (counted separately). See module docstring for the stated "
            "tension with combinatorial-template generation."
        ),
        "near_duplicate_sample_pairs": near_dupe_pairs,
        "near_duplicate_by_template_pair": {
            f"{a} :: {b}": c for (a, b), c in template_pair_counts.most_common(30)
        },
        "n_duplicate_template_and_params": sum(len(v) for v in param_collisions.values()),
        "duplicate_template_and_params": param_collisions,
    }


# ------------------------------------------------------------------------------------
# (b) answerable SoQL validity + (f) degenerate gold answers (shared live pass)
# ------------------------------------------------------------------------------------

_COUNT_TEMPLATE_RE = re.compile(r"^\s*select\s+count\s*\(", re.IGNORECASE)


def _is_count_query(soql: str) -> bool:
    return bool(_COUNT_TEMPLATE_RE.match(soql))


def check_answerable_and_unreliable_soql(
    client: SocrataClient, questions: list[dict[str, Any]]
) -> dict[str, Any]:
    targets = [q for q in questions if q["cls"] in ("answerable", "unreliable")]
    n_ok = 0
    invalid: list[dict[str, Any]] = []
    degenerate: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for q in targets:
        soql = q["soql"]
        try:
            rows, _ = client.query(q["dataset_id"], soql, force_refresh=True)
        except SocrataError as e:
            errors.append({"id": q["id"], "cls": q["cls"], "soql": soql, "error": str(e)})
            continue

        if len(rows) != 1 or "result" not in rows[0]:
            invalid.append(
                {
                    "id": q["id"],
                    "cls": q["cls"],
                    "soql": soql,
                    "reason": (
                        "expected exactly 1 row with a 'result' column, "
                        f"got {len(rows)} rows: {rows[:3]!r}"
                    ),
                }
            )
            continue

        value = rows[0]["result"]
        if value is None:
            degenerate.append(
                {
                    "id": q["id"],
                    "cls": q["cls"],
                    "template_id": q["template_id"],
                    "value": None,
                    "reason": "null gold value",
                }
            )
            n_ok += 1  # valid resolution (1 row, addressable 'result' key) but flagged degenerate
            continue

        if _is_count_query(soql):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = None
            if numeric == 0:
                degenerate.append(
                    {
                        "id": q["id"],
                        "cls": q["cls"],
                        "template_id": q["template_id"],
                        "value": value,
                        "reason": (
                            "count(*) query resolved to 0, so abstention "
                            "and answering are indistinguishable"
                        ),
                    }
                )
        n_ok += 1

    return {
        "n_checked": len(targets),
        "n_valid_single_nonnull_scalar": n_ok,
        "n_invalid": len(invalid),
        "invalid_questions": invalid,
        "n_query_errors": len(errors),
        "query_errors": errors,
        "n_degenerate": len(degenerate),
        "degenerate_questions": degenerate,
    }


# ------------------------------------------------------------------------------------
# (c) unanswerable genuinely unanswerable, live
# ------------------------------------------------------------------------------------


def _check_temporal_history_absent(
    schema: dict[str, Any], q: dict[str, Any]
) -> dict[str, Any] | None:
    """Re-verify a single `temporal_history_absent` question against a FRESH schema.
    Returns an `invalid_questions`-shaped dict on failure, `None` if still genuinely
    unanswerable. Split out of `check_unanswerable` to keep that function's branch
    count under the mccabe limit — this is a self-contained structural check with no
    shared state beyond `schema`."""
    evidence = q["evidence"]
    kind = evidence.get("kind")
    field_name = evidence.get("field_name")
    expected_type = evidence.get("schema_field_type")
    col = next((c for c in schema["columns"] if c["field_name"] == field_name), None)
    if col is None:
        return {
            "id": q["id"],
            "kind": kind,
            "field_name": field_name,
            "reason": (
                "field is no longer present in the fresh live schema at all — "
                "the single-scalar-column claim can no longer be confirmed"
            ),
        }
    if col["data_type"] != expected_type:
        return {
            "id": q["id"],
            "kind": kind,
            "field_name": field_name,
            "expected_schema_field_type": expected_type,
            "current_schema_field_type": col["data_type"],
            "reason": (
                "live column data_type changed since generation — the "
                "no-history-dimension claim can no longer be confirmed unmodified"
            ),
        }
    return None


def check_unanswerable(client: SocrataClient, questions: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [q for q in questions if q["cls"] == "unanswerable"]
    schema = client.describe_dataset(DATASET_ID, force_refresh=True)
    live_fields = set(schema["field_names"])

    invalid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    n_field = 0
    n_record = 0
    n_temporal = 0
    n_ok = 0

    for q in targets:
        evidence = q.get("evidence") or {}
        kind = evidence.get("kind")
        if q.get("soql") is not None:
            invalid.append(
                {
                    "id": q["id"],
                    "reason": (
                        "unanswerable question carries a scoring 'soql' field (spec violation)"
                    ),
                }
            )
            continue
        if kind == "field_absent":
            n_field += 1
            field_name = evidence.get("field_name")
            if field_name in live_fields:
                invalid.append(
                    {
                        "id": q["id"],
                        "kind": kind,
                        "field_name": field_name,
                        "reason": (
                            "field is PRESENT in the fresh live schema, "
                            "so it is no longer genuinely unanswerable"
                        ),
                    }
                )
            else:
                n_ok += 1
        elif kind == "record_absent":
            n_record += 1
            probe = evidence.get("verification_soql")
            try:
                rows, _ = client.query(q["dataset_id"], probe, force_refresh=True)
            except SocrataError as e:
                errors.append({"id": q["id"], "kind": kind, "probe": probe, "error": str(e)})
                continue
            row_count = int(rows[0]["result"]) if rows else 0
            if row_count != 0:
                invalid.append(
                    {
                        "id": q["id"],
                        "kind": kind,
                        "probe": probe,
                        "live_row_count": row_count,
                        "reason": (
                            "probe returned nonzero rows, so the fabricated "
                            "record now exists or collided"
                        ),
                    }
                )
            else:
                n_ok += 1
        elif kind == "temporal_history_absent":
            # Structural claim, not a field-absence claim: `field_name` (e.g.
            # "inspection") must still be a single SCALAR column of the recorded
            # `schema_field_type` (e.g. "calendar_date") — i.e. the dataset still
            # has no per-row historical time series for it. Re-verified against a
            # FRESH describe_dataset() column-type lookup, machine-checkable, the
            # same way field_absent is — this is a real verification, not an
            # UNVERIFIABLE-BY-DESIGN case. See `_check_temporal_history_absent`.
            n_temporal += 1
            failure = _check_temporal_history_absent(schema, q)
            if failure is not None:
                invalid.append(failure)
            else:
                n_ok += 1
        else:
            invalid.append({"id": q["id"], "reason": f"unknown evidence.kind: {kind!r}"})

    return {
        "n_checked": len(targets),
        "n_field_absent_checked": n_field,
        "n_record_absent_checked": n_record,
        "n_temporal_history_absent_checked": n_temporal,
        "n_genuinely_unanswerable": n_ok,
        "n_invalid": len(invalid),
        "invalid_questions": invalid,
        "n_probe_errors": len(errors),
        "probe_errors": errors,
        "n_unverifiable_by_design": len(unverifiable),
        "unverifiable_by_design_questions": unverifiable,
        "live_schema_field_names": schema["field_names"],
    }


# ------------------------------------------------------------------------------------
# (d) unreliable class correctness (instrument-level block) + field-split invariant
# ------------------------------------------------------------------------------------


def check_unreliable_reliability(questions: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = MEASURE_RELIABILITY["nyc_street_systemrating"]
    unreliable = [q for q in questions if q["cls"] == "unreliable"]
    answerable = [q for q in questions if q["cls"] == "answerable"]

    bad_reliability: list[str] = []
    not_about_rating: list[str] = []
    for q in unreliable:
        if q.get("reliability") != canonical:
            bad_reliability.append(q["id"])
        soql = q.get("soql") or ""
        if SYSTEMRATING_FIELD not in soql:
            not_about_rating.append(q["id"])

    answerable_touches_rating: list[str] = []
    for q in answerable:
        soql = q.get("soql") or ""
        if SYSTEMRATING_FIELD in soql:
            answerable_touches_rating.append(q["id"])
        if q.get("reliability") is not None:
            answerable_touches_rating.append(
                q["id"] + " (nonzero reliability block on an answerable question)"
            )

    return {
        "n_unreliable_checked": len(unreliable),
        "n_answerable_checked": len(answerable),
        "n_unreliable_with_wrong_or_missing_reliability_block": len(bad_reliability),
        "unreliable_bad_reliability_ids": bad_reliability,
        "n_unreliable_not_about_systemrating": len(not_about_rating),
        "unreliable_not_about_rating_ids": not_about_rating,
        "n_answerable_violating_field_split": len(answerable_touches_rating),
        "answerable_violating_field_split_ids": answerable_touches_rating,
    }


# ------------------------------------------------------------------------------------
# (e) class balance
# ------------------------------------------------------------------------------------


def check_class_balance(questions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(questions)
    counts = Counter(q["cls"] for q in questions)
    template_counts = Counter((q["cls"], q["template_id"]) for q in questions)
    return {
        "n_total": n,
        "counts": dict(counts),
        "proportions": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
        "n_distinct_templates": len({q["template_id"] for q in questions}),
        "template_counts": {f"{c}::{t}": n for (c, t), n in template_counts.most_common()},
    }


# ------------------------------------------------------------------------------------
# top-level orchestration
# ------------------------------------------------------------------------------------


def audit(questions_path: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    questions = _load(questions_path)
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    client = SocrataClient(**kwargs)

    dup = check_duplicates(questions)
    soql_check = check_answerable_and_unreliable_soql(client, questions)
    unanswerable_check = check_unanswerable(client, questions)
    reliability_check = check_unreliable_reliability(questions)
    balance = check_class_balance(questions)

    n_total = len(questions)
    n_invalid_unanswerable = unanswerable_check["n_invalid"]
    n_invalid_scoring = soql_check["n_invalid"] + soql_check["n_query_errors"]
    n_bad_reliability = reliability_check["n_unreliable_with_wrong_or_missing_reliability_block"]
    n_field_split_violations = reliability_check["n_answerable_violating_field_split"]
    n_degenerate = soql_check["n_degenerate"]

    n_structurally_valid = (
        n_total
        - n_invalid_unanswerable
        - n_invalid_scoring
        - n_bad_reliability
        - n_field_split_violations
        - dup["n_exact_duplicates"]
        - dup["n_duplicate_template_and_params"]
    )
    n_calibration_usable = n_structurally_valid - n_degenerate

    gate = {
        "pre_registered_criteria": {
            "min_total_valid": 500,
            "max_near_duplicate_rate": 0.05,
            "max_invalid_unanswerable": 0,
        },
        "measured": {
            "n_total_generated": n_total,
            "n_structurally_valid": n_structurally_valid,
            "n_calibration_usable_excl_degenerate": n_calibration_usable,
            "near_duplicate_rate": dup["near_duplicate_rate"],
            "n_invalid_unanswerable": n_invalid_unanswerable,
            "n_invalid_scoring_soql": n_invalid_scoring,
            "n_bad_reliability_blocks": n_bad_reliability,
            "n_field_split_violations": n_field_split_violations,
            "n_degenerate_gold_answers": n_degenerate,
        },
    }
    passes_volume = n_structurally_valid >= 500
    passes_near_dup = (dup["near_duplicate_rate"] or 1.0) < 0.05
    passes_invalid_unanswerable = n_invalid_unanswerable == 0
    if passes_volume and passes_near_dup and passes_invalid_unanswerable:
        verdict = "PASS"
    elif passes_volume and passes_invalid_unanswerable and not passes_near_dup:
        verdict = "NARROW"
    elif not passes_volume:
        verdict = "FAIL"
    else:
        verdict = "NARROW"
    gate["verdict"] = verdict
    gate["passes_volume_ge_500"] = passes_volume
    gate["passes_near_dup_lt_5pct"] = passes_near_dup
    gate["passes_zero_invalid_unanswerable"] = passes_invalid_unanswerable

    return {
        "audited_at": _now(),
        "questions_path": str(questions_path),
        "n_questions": n_total,
        "duplicates": dup,
        "answerable_unreliable_soql": soql_check,
        "unanswerable": unanswerable_check,
        "unreliable_reliability": reliability_check,
        "class_balance": balance,
        "gate": gate,
    }


def _cli() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "questions_scaled.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "questions_scaled_audit.json",
    )
    args = ap.parse_args()

    if not args.questions.exists():
        raise SystemExit(f"questions file not found: {args.questions}")

    report = audit(args.questions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    g = report["gate"]
    print(f"AUDIT COMPLETE -> {args.out}")
    print(f"  n_questions={report['n_questions']}")
    print(f"  near_duplicate_rate={report['duplicates']['near_duplicate_rate']}")
    print(f"  n_invalid_unanswerable={report['unanswerable']['n_invalid']}")
    print(f"  n_invalid_scoring_soql={report['answerable_unreliable_soql']['n_invalid']}")
    print(f"  n_degenerate_gold_answers={report['answerable_unreliable_soql']['n_degenerate']}")
    print(f"  n_structurally_valid={g['measured']['n_structurally_valid']}")
    print(f"  VERDICT: {g['verdict']}")


if __name__ == "__main__":
    _cli()
