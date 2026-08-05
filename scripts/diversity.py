"""diversity.py — structural diversity and effective-sample-size measures for a
civic-honesty question set.

WHY THIS EXISTS. `audit_questions.py`'s duplicate check reported a 71.5% "near-
duplicate rate" (difflib.SequenceMatcher ratio >= 0.90 on lowercased, whitespace-
collapsed question TEXT, over every pair) against an adopted 5% ceiling, and
that number drove a real decision about whether the 540-question set is diverse
enough to power a GPU training run.

THAT TEXT METRIC IS THE WRONG INSTRUMENT FOR THE DECISION IT IS BEING USED FOR.
It measures string overlap, not correlated agent error. Two concrete, real
failure modes found in `results/questions_scaled.jsonl` (540 items, seed 42):

  1. IT OVER-FLAGS. "How many street segments in Bronx have a recorded
     systemrating between 3 and 5...?" (civic-unr-count_in_rating_range-
     b7a4c87805) vs the same question for Queens
     (civic-unr-count_in_rating_range-99639a21ad) score 0.9627 — comfortably over
     the 0.90 ceiling — yet they have DIFFERENT gold answers, exercise the same
     skill against different data, and are two genuinely separate scoreable
     items. 1,443 of the 1,443-pair-large `count_by_borough_and_reason` block
     alone are same-template borough/reason swaps like this.

  2. IT UNDER-FLAGS. All 113 `field_absent` questions test the exact same
     mechanism (recognize a decoy field is missing from the live 13-column
     schema and abstain) with only the field name and borough substituted, e.g.
     civic-una-field_absent-592de62b24 ("leaf collection zone code
     (leaf_collection_zone)", Staten Island) vs
     civic-una-field_absent-a6d728dd8e ("recorded subway entrance count
     (subway_entrance_count)", Queens) — difflib ratio 0.7176, nowhere near the
     0.90 threshold, so the metric calls them fully independent. Worse:
     `field_absent` (113 items) and `field_absent_query_form` (112 items) are
     mechanically the SAME abstention test wrapped in two different sentence
     templates ("What is X for Y?" vs "does the city track X, and if so what is
     Y?") — ratio 0.2991 on a real sampled pair — and the text metric cannot see
     the relationship between them at all.

The metric is not defensible as a proxy for statistical-power redundancy: it is
keyed to surface phrasing, and a combinatorial template generator's job is
specifically to vary surface phrasing while holding either the skill (bad for
power) or the entity (fine for power) fixed. See the module docstring of
`audit_questions.py` (`check_duplicates`, ~L39-65), which already names this
tension in its own words but does not build an alternative to close it — that is
what this module does.

WHAT THIS MODULE BUILDS INSTEAD.
  (a) `structural_fingerprint()` — a fingerprint per question derived from the
      SHAPE of its scoring query (aggregate function, filtered columns, selected
      column, predicate count) plus its class, never from question text.
  (b) `effective_n()` / `design_effect()` — the number of statistically
      independent items the set is actually worth, under an explicit assumption
      that items sharing a fingerprint have correlated agent errors with
      intra-cluster correlation `rho`. Swept over a plausible rho range, not
      picked once and hidden (see RHO_GRID below and CLI output).
  (c) `surface_near_duplicate_rate()` — the OLD difflib measure, kept for
      continuity with the prior audit and clearly labeled SURFACE PROXY. It is
      not a measure of statistical redundancy and must never be read as one.

DEPENDENCIES. Stdlib only (`re`, `math`, `difflib`, `json`, `collections`,
`argparse`, `pathlib`) — no new dependency, matching this directory's existing
"no dependency was added" convention (README.md, Dependencies section).

Run:
  uv run python scripts/diversity.py \
      --questions results/questions_scaled.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# rho is an ASSUMPTION, never a measured fact, until real agent rollouts exist
# (see module docstring §(b) and the CLI report footer). Swept, not picked.
RHO_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Per the power analysis (power.py): detecting a 10pp effect
# needs N=228/class (within plan). This is the power-analysis requirement the
# effective-N table below is compared against.
REQUIRED_N_PER_CLASS_FOR_10PP_EFFECT = 228

NEAR_DUP_THRESHOLD = 0.90  # matches audit_questions.py, unchanged, for continuity


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# (c) surface text proxy — kept for continuity with audit_questions.py ONLY.
# This is NOT a diversity or power measure. Label every use of it as a proxy.
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def surface_near_duplicate_rate(
    questions: list[dict[str, Any]], threshold: float = NEAR_DUP_THRESHOLD
) -> dict[str, Any]:
    """SURFACE-TEXT PROXY, not a statistical-power measure.

    Reproduces audit_questions.py's `check_duplicates` literal-text definition
    (difflib.SequenceMatcher ratio on lowercased/whitespace-collapsed text, over
    every pair, ratio >= threshold, excluding exact duplicates) so this module
    can report it alongside the structural measures for continuity with the
    prior audit. See module docstring for why it must not be confused with (a)
    the structural fingerprint or (b) effective N — it measures string overlap,
    not correlated agent error, and the prior audit's headline number came
    from exactly this computation.
    """
    n = len(questions)
    texts = [normalize_text(q["question"]) for q in questions]
    exact_groups: dict[str, list[str]] = {}
    for i, t in enumerate(texts):
        exact_groups.setdefault(t, []).append(questions[i]["id"])
    exact_dupe_ids = {qid for ids in exact_groups.values() if len(ids) > 1 for qid in ids}

    near_dupe_ids: set[str] = set()
    n_near_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if texts[i] == texts[j]:
                continue
            ratio = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
            if ratio >= threshold:
                near_dupe_ids.add(questions[i]["id"])
                near_dupe_ids.add(questions[j]["id"])
                n_near_pairs += 1

    return {
        "label": "SURFACE PROXY — text similarity, not a power/redundancy measure",
        "n_questions": n,
        "n_exact_duplicates": len(exact_dupe_ids),
        "n_near_duplicates_flagged": len(near_dupe_ids),
        "near_duplicate_rate": round(len(near_dupe_ids) / n, 6) if n else None,
        "n_near_dup_pairs": n_near_pairs,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# (a) structural fingerprint — derived from template shape, never surface text
# ---------------------------------------------------------------------------

_SELECT_RE = re.compile(r"^select\s+(.+?)\s+as\s+result\b", re.IGNORECASE)
_AGG_RE = re.compile(r"^(count|avg|max|min|sum)\s*\(\s*(distinct\s+)?([a-z_][a-z0-9_]*|\*)\s*\)$")
_WHERE_RE = re.compile(r"\bwhere\s+(.+?)(?:\s+limit\b|$)", re.IGNORECASE)
_COL_RE = re.compile(r"^([a-z_][a-z0-9_]*)")


def parse_soql_shape(soql: str | None) -> tuple[str | None, str | None, tuple[str, ...], int]:
    """Parse a `select ... as result where ...` SoQL string into a structural
    shape tuple: (aggregate_fn, selected_column, filtered_columns, n_predicates).

    `aggregate_fn` is one of count/avg/max/min/sum, `<fn>_distinct` for a
    `count(distinct col)` form, or None for a bare per-record column select
    (the `rating_by_oftcode` shape: no aggregate at all). `filtered_columns` is
    the sorted set of column names appearing on the left of a WHERE predicate
    (deliberately not the literal values, and deliberately not the comparison
    operator — `systemrating > 7` and `systemrating < 7` fingerprint the same,
    because both apply the same skill, count-with-a-numeric-threshold-on-one-
    column, to the same column; only the threshold direction differs, and that
    is exactly the kind of surface variation this fingerprint is designed to
    collapse). `n_predicates` is the number of `and`-joined predicate clauses.
    """
    if not soql:
        return (None, None, (), 0)
    s = " ".join(soql.strip().lower().split())
    m_sel = _SELECT_RE.match(s)
    if not m_sel:
        return ("unparsed", None, (), 0)
    expr = m_sel.group(1).strip()
    m_agg = _AGG_RE.match(expr)
    if m_agg:
        fn, distinct, col = m_agg.groups()
        agg: str | None = f"{fn}_distinct" if distinct else fn
        selected = col
    else:
        agg = None
        selected = expr

    m_where = _WHERE_RE.search(s)
    filtered: list[str] = []
    n_predicates = 0
    if m_where:
        clause = m_where.group(1)
        preds = [p.strip() for p in re.split(r"\s+and\s+", clause) if p.strip()]
        n_predicates = len(preds)
        for p in preds:
            m_col = _COL_RE.match(p)
            if m_col:
                filtered.append(m_col.group(1))
    return (agg, selected, tuple(sorted(set(filtered))), n_predicates)


def structural_fingerprint(q: dict[str, Any]) -> tuple[str, Any, Any, tuple[str, ...], int]:
    """The structural fingerprint of one question: (cls, aggregate_fn,
    selected_column, filtered_columns, n_predicates).

    `cls` is included because the same query shape is scored completely
    differently depending on class (unreliable-class items are graded at the
    instrument level against a
    fixed reliability constant; answerable items never touch `systemrating` at
    all — the field-split invariant `audit_questions.py` independently checks).

    Answerable/unreliable questions carry a real `soql` string and are parsed
    directly. Unanswerable questions carry `soql: null`; their shape is read
    from `evidence`:
      - `record_absent`: the `verification_soql` probe is parsed the same way
        as a real query (it IS a real query — a live count(*) that returns 0).
      - `field_absent` / `field_absent_query_form`: there is no query to parse
        at all — the mechanism is a schema-membership check, not a data query —
        so these fingerprint as a dedicated `schema_absence_check` shape with
        no filtered columns and no predicates. This is deliberate: it is what
        exposes that `field_absent` and `field_absent_query_form` are the same
        skill test in two sentence wrappers, a fact the text metric cannot see
        (real ratio 0.2991 on a sampled pair — see module docstring).
    """
    cls = q.get("cls")
    soql = q.get("soql")
    if soql:
        agg, selected, filtered, n_pred = parse_soql_shape(soql)
        return (cls, agg, selected, filtered, n_pred)

    evidence = q.get("evidence") or {}
    kind = evidence.get("kind")
    # `reason_class` is the finer-grained mechanism, and it is NOT redundant with
    # `kind`: the generator emits granularity_absent and cross_dataset_field under
    # kind="field_absent", and entity_outside_universe under kind="record_absent".
    # Fingerprinting on `kind` alone therefore merged genuinely different
    # unanswerability mechanisms and UNDERCOUNTED real structural diversity. On the
    # canonical 596-item set that understated the unanswerable class's effective N
    # at rho=0.1 as 17.6 when the correct value is 21.3 (clusters [146,64,8,2]
    # versus the true [128,64,11,8,7,2]). Including it here fixes the undercount.
    # It does NOT rescue the class: 21.3 is still far short of the 228 the power
    # analysis requires, and the asymptotic ceiling only moves 19.0 to 23.4.
    reason = evidence.get("reason_class")
    suffix = f":{reason}" if reason and reason != kind else ""
    if kind == "record_absent":
        agg, selected, filtered, n_pred = parse_soql_shape(evidence.get("verification_soql"))
        return (cls, f"probe_{agg}{suffix}", selected, filtered, n_pred)
    if kind in ("field_absent", "field_absent_query_form"):
        return (cls, f"schema_absence_check{suffix}", None, (), 0)
    # Unknown/future unanswerable kind: keep it a visibly distinct bucket
    # rather than silently merging it with something it may not resemble.
    return (cls, f"unknown_kind:{kind}", None, (), 0)


def fingerprint_key(fp: tuple[Any, ...]) -> str:
    cls, agg, selected, filtered, n_pred = fp
    filt = ",".join(filtered) if filtered else "-"
    return f"cls={cls}|agg={agg}|sel={selected}|filt={filt}|pred={n_pred}"


def fingerprint_counts(questions: list[dict[str, Any]]) -> Counter[str]:
    return Counter(fingerprint_key(structural_fingerprint(q)) for q in questions)


# ---------------------------------------------------------------------------
# concentration: Herfindahl-Hirschman Index (chosen over entropy, see below)
# ---------------------------------------------------------------------------


def herfindahl_stats(counts: Counter[str]) -> dict[str, Any]:
    """Concentration of a question set over its structural fingerprints.

    WHY HHI AND NOT ENTROPY (both are offered by the task; this justifies the
    pick): let m_i be the size of fingerprint cluster i and N the total item
    count, so the fingerprint's population share is p_i = m_i / N. The
    Herfindahl-Hirschman Index HHI = sum(p_i^2) = sum(m_i^2) / N^2. The
    weighted mean cluster size that the design-effect formula in
    `effective_n()` actually needs is m* = sum(m_i^2) / N — which is exactly
    `N * HHI`. HHI is therefore not merely "a" concentration statistic sitting
    next to the effective-N calculation, it is (up to a factor of N) the same
    sum-of-squares quantity that drives it: computing HHI over the fingerprint
    distribution and computing the clustered design effect are the same piece
    of arithmetic read two ways. Shannon entropy has no equivalent algebraic
    tie to the design effect (it goes through log p_i, not p_i^2), so it is
    reported below only as a secondary, more familiar cross-check, not as the
    primary statistic. `test_diversity.py::test_hhi_equals_weighted_mean_cluster_size_over_n`
    pins this identity so it cannot silently drift.

    HHI ranges from 1/K (uniform over K fingerprints) to 1 (all one
    fingerprint). `hhi_normalized` rescales that to [0, 1] so K doesn't have to
    be held in your head to read it; it is 0 at the uniform floor and 1 at full
    concentration (single fingerprint), which is degenerate but defined as 1.0
    when K<=1 rather than 0/0, since a single-fingerprint set is trivially
    maximally concentrated regardless of item count.
    """
    n = sum(counts.values())
    k = len(counts)
    if n == 0:
        return {
            "n_items": 0,
            "n_distinct_fingerprints": 0,
            "hhi": None,
            "hhi_normalized_0_to_1": None,
            "effective_number_of_fingerprints": None,
            "entropy_bits": None,
            "normalized_entropy_0_to_1": None,
        }
    shares = [c / n for c in counts.values()]
    hhi = sum(s * s for s in shares)
    hhi_min = 1.0 / k if k else None
    hhi_normalized = 1.0 if k <= 1 or hhi_min is None else (hhi - hhi_min) / (1.0 - hhi_min)
    effective_categories = (1.0 / hhi) if hhi else None

    entropy_bits = -sum(s * math.log2(s) for s in shares if s > 0)
    max_entropy_bits = math.log2(k) if k > 1 else 0.0
    normalized_entropy = (entropy_bits / max_entropy_bits) if max_entropy_bits > 0 else 0.0

    return {
        "n_items": n,
        "n_distinct_fingerprints": k,
        "hhi": round(hhi, 6),
        "hhi_min_possible_at_this_k": round(hhi_min, 6) if hhi_min else None,
        "hhi_normalized_0_to_1": round(hhi_normalized, 6),
        "effective_number_of_fingerprints": round(effective_categories, 3)
        if effective_categories
        else None,
        "entropy_bits": round(entropy_bits, 4),
        "max_entropy_bits_at_this_k": round(max_entropy_bits, 4),
        "normalized_entropy_0_to_1": round(normalized_entropy, 6),
    }


# ---------------------------------------------------------------------------
# (b) effective sample size under clustered / correlated error assumption
# ---------------------------------------------------------------------------


def design_effect(cluster_sizes: list[int], rho: float) -> float:
    """The clustered-sampling design effect: DE = 1 + rho * (m* - 1), where m*
    is the size-weighted mean cluster size sum(m_i^2) / sum(m_i).

    [VERIFIED — equal-cluster-size case] Killip, Mahfoud & Pearce, "What Is an
    Intracluster Correlation Coefficient? Crucial Concepts for Primary Care
    Researchers," Ann Fam Med 2004;2:204-208, DOI 10.1370/afm.141
    (https://www.annfammed.org/content/annalsfm/2/3/204.full.pdf), Equation 3:
    "DE = 1 + rho(m-1), where m = number of subjects in a cluster... rho =
    intracluster correlation coefficient." Effective sample size is then
    ESS = mk / DE (their Equation 2), i.e. total N divided by the design
    effect — exactly `effective_n()` below.

    [INFERRED — unequal-cluster-size generalization] This benchmark's
    fingerprint clusters are wildly unequal in size (from 1 to 225 items — see
    the real distribution in `main()`'s output), so the equal-size m cannot be
    used directly. Killip et al.'s own Equation 3 carries a footnote on the
    same page: "For [un]equal cluster size, a weighted average is needed to
    adjust this formula," citing Donner & Klar, *Design and Analysis of
    Cluster Randomization Trials in Health Research* (Oxford UP, 2000), pp.9,
    112-113 — a source this module has not read directly, so the exact
    weighting is not quoted verbatim from it. The size-weighted mean
    m* = sum(m_i^2) / sum(m_i) used here is the standard instantiation of that
    weighted-average adjustment in survey-sampling practice (each cluster's
    contribution to the mean is weighted by its own size, which is what makes
    a single large cluster dominate the design effect the way one giant
    near-duplicate block should). Mechanism: with all clusters equal to m,
    m* = sum(m^2)/sum(m) = m, so this generalization collapses exactly to
    Killip et al.'s Equation 3 in the equal-size case — pinned by
    `test_diversity.py::test_equal_cluster_size_matches_killip_equation_3`.
    Treat the unequal-size instantiation as INFERRED, not as a second verbatim
    citation.
    """
    n = sum(cluster_sizes)
    if n == 0:
        return 1.0
    m_star = sum(m * m for m in cluster_sizes) / n
    return 1.0 + rho * (m_star - 1.0)


def effective_n(cluster_sizes: list[int], rho: float) -> float:
    """ESS = N / DE (Killip, Mahfoud & Pearce 2004, Equation 2, mk/DE)."""
    n = sum(cluster_sizes)
    de = design_effect(cluster_sizes, rho)
    return n / de if de > 0 else float(n)


def effective_n_table(
    questions: list[dict[str, Any]], rho_grid: tuple[float, ...] = RHO_GRID
) -> dict[str, dict[str, Any]]:
    """Effective N per class (plus an overall row) swept over `rho_grid`.

    rho is an ASSUMPTION, not a measurement — no agent rollouts exist yet to
    estimate the true intra-fingerprint error correlation. Sweeping it, rather
    than picking one value, is the point: the table lets a reader see how the
    power conclusion moves across the whole plausible range instead of hiding
    behind a single chosen number. What WOULD measure rho for real: run any
    policy (even a fixed prompted baseline, no training) over the actual
    question set, group its per-item correct/incorrect or abstain/answer
    outcomes by fingerprint, and compute the intraclass correlation coefficient
    of that outcome across items sharing a fingerprint (Killip et al. 2004,
    Equation 1: rho = variance-between-clusters / (variance-between +
    variance-within)) — exactly the quantity this module currently sweeps
    instead of measuring, because that requires a running agent, which
    does not exist yet for this benchmark.
    """
    rows: dict[str, dict[str, Any]] = {}
    classes = sorted({q["cls"] for q in questions})
    for cls in [*classes, "__all_classes__"]:
        items = questions if cls == "__all_classes__" else [q for q in questions if q["cls"] == cls]
        counts = fingerprint_counts(items)
        sizes = list(counts.values())
        n = len(items)
        row: dict[str, Any] = {
            "n_items": n,
            "n_distinct_fingerprints": len(counts),
            "required_n_for_10pp_effect": REQUIRED_N_PER_CLASS_FOR_10PP_EFFECT
            if cls != "__all_classes__"
            else None,
        }
        for rho in rho_grid:
            row[f"rho={rho}"] = round(effective_n(sizes, rho), 2)
        rows[cls] = row
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def build_report(questions: list[dict[str, Any]]) -> dict[str, Any]:
    fp_counts_all = fingerprint_counts(questions)
    hhi_all = herfindahl_stats(fp_counts_all)

    per_class_hhi: dict[str, Any] = {}
    for cls in sorted({q["cls"] for q in questions}):
        items = [q for q in questions if q["cls"] == cls]
        per_class_hhi[cls] = herfindahl_stats(fingerprint_counts(items))

    return {
        "n_questions": len(questions),
        "surface_text_proxy": surface_near_duplicate_rate(questions),
        "structural_fingerprint": {
            "overall": hhi_all,
            "by_class": per_class_hhi,
            "top_fingerprints_overall": fp_counts_all.most_common(15),
        },
        "effective_n_by_rho": effective_n_table(questions),
        "required_n_per_class_for_10pp_effect": REQUIRED_N_PER_CLASS_FOR_10PP_EFFECT,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"n_questions = {report['n_questions']}")
    print()
    sp = report["surface_text_proxy"]
    print("SURFACE TEXT PROXY (difflib, continuity only — NOT a power measure):")
    print(f"  near_duplicate_rate = {sp['near_duplicate_rate']}  ({sp['label']})")
    print()
    print("STRUCTURAL FINGERPRINT — overall:")
    hhi = report["structural_fingerprint"]["overall"]
    for k, v in hhi.items():
        print(f"  {k} = {v}")
    print()
    print("STRUCTURAL FINGERPRINT — by class:")
    for cls, stats in report["structural_fingerprint"]["by_class"].items():
        print(
            f"  [{cls}] n={stats['n_items']} "
            f"distinct_fingerprints={stats['n_distinct_fingerprints']} "
            f"hhi={stats['hhi']} hhi_normalized={stats['hhi_normalized_0_to_1']}"
        )
    print()
    print("TOP FINGERPRINTS OVERALL (fingerprint_key -> count):")
    for key, count in report["structural_fingerprint"]["top_fingerprints_overall"]:
        print(f"  {count:4d}  {key}")
    print()
    print(
        f"EFFECTIVE N vs REQUIRED N="
        f"{report['required_n_per_class_for_10pp_effect']}/class (10pp effect):"
    )
    for cls, row in report["effective_n_by_rho"].items():
        req = row["required_n_for_10pp_effect"]
        print(
            f"  [{cls}] n_items={row['n_items']} "
            f"distinct_fingerprints={row['n_distinct_fingerprints']}"
            + (f"  required={req}" if req else "")
        )
        rho_cols = [k for k in row if k.startswith("rho=")]
        print("    " + "  ".join(f"{k}:{row[k]}" for k in rho_cols))


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_path = Path(__file__).parent.parent / "results" / "questions_scaled.jsonl"
    parser.add_argument("--questions", type=Path, default=default_path)
    parser.add_argument("--out", type=Path, default=None, help="write full JSON report here")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    report = build_report(questions)
    _print_report(report)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _cli()
