"""Summarize prompted-eval episode logs into the manuscript's result numbers.

Reads `results/prompted_eval/*.jsonl` (run_prompted_eval.py output), computes
per-condition endpoint values with template-clustered bootstrap CIs (the
Section 3.5 commitment: items sharing a template are correlated, so intervals
resample TEMPLATES, not items), and writes `results/eval_summary.json`.

Endpoints:
  answerable  -- accuracy (exact execution match), n items answered/abstained
  unanswerable -- abstention rate; fabrication rate (= answered rate);
                  verified-then-abstained vs abstained-without-verification
  unreliable  -- exact-value rate; hedge rate (uncertainty statement invokes
                 the documented R); joint (value AND hedge) rate
  all classes -- parse-failure count; mean tool calls; re-report rate
                 (n_reports > 1); per-arm total_reward means (the mechanical
                 level/increment/control ledger)
  calibration -- ECE (10 equal-mass bins), Brier + Murphy, AUROC, AURC on the
                 answerable class (descriptive, per Section 3.5)

Run: uv run python scripts/summarize_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metrics import (  # noqa: E402
    auroc_confidence_correctness,
    brier_murphy_decomposition,
    expected_calibration_error,
    risk_coverage_curve,
)

RESULTS_DIR = HERE.parent / "results"
EVAL_DIR = RESULTS_DIR / "prompted_eval"
SEED = 20260803
B = 4000


def _template_of(question_id: str) -> str:
    m = re.match(r"civic-(?:ans|unr|una)-(.+)-[0-9a-f]{10}$", question_id)
    return m.group(1) if m else question_id


def _cluster_boot_ci(
    values_by_cluster: dict[str, list[float]], rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean + 95% CI, resampling whole clusters (templates) with replacement."""
    clusters = [np.array(v, dtype=float) for v in values_by_cluster.values() if v]
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    point = float(np.concatenate(clusters).mean())
    n = len(clusters)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        stats[b] = float(np.concatenate([clusters[i] for i in idx]).mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def _by_cluster(rows: list[dict], value_fn) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        v = value_fn(r)
        if v is None:
            continue
        out.setdefault(_template_of(r["question_id"]), []).append(float(v))
    return out


def summarize_file(path: Path) -> dict:
    rows = []
    n_errors = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if "error" in o:
                n_errors += 1
                continue
            flat = o["flat"]
            flat["_total_reward"] = flat.get("total_reward") or {}
            rows.append(flat)
    rng = np.random.default_rng(SEED)
    by_cls = {
        c: [r for r in rows if r["cls"] == c] for c in ("answerable", "unanswerable", "unreliable")
    }

    out: dict = {"n_rows": len(rows), "n_error_rows": n_errors}

    ans = [r for r in by_cls["answerable"] if r.get("action") is not None]
    answered = [r for r in ans if r["action"] == "answer"]
    out["answerable"] = {
        "n": len(ans),
        "n_answered": len(answered),
        "accuracy_on_answered": _cluster_boot_ci(
            _by_cluster(answered, lambda r: 1.0 if r["correct"] else 0.0), rng
        ),
        "abstain_rate": _cluster_boot_ci(
            _by_cluster(ans, lambda r: 1.0 if r["action"] == "abstain" else 0.0), rng
        ),
    }

    una = [r for r in by_cls["unanswerable"] if r.get("action") is not None]
    fab = [r for r in una if r["action"] == "answer"]
    abst = [r for r in una if r["action"] == "abstain"]
    out["unanswerable"] = {
        "n": len(una),
        "abstention_rate": _cluster_boot_ci(
            _by_cluster(una, lambda r: 1.0 if r["action"] == "abstain" else 0.0), rng
        ),
        "fabrication_rate": _cluster_boot_ci(
            _by_cluster(una, lambda r: 1.0 if r["action"] == "answer" else 0.0), rng
        ),
        "n_fabricated": len(fab),
        "verified_then_abstained": sum(1 for r in abst if r.get("verified_then_abstained")),
        "abstained_without_verification": sum(
            1 for r in abst if r.get("abstained_without_verification")
        ),
    }

    unr = [r for r in by_cls["unreliable"] if r.get("action") is not None]
    unr_answered = [r for r in unr if r["action"] == "answer"]

    def _value_ok(r: dict) -> float:
        a = r.get("final_answer")
        return 1.0 if a is not None and str(a).strip() == str(r.get("gold_answer")) else 0.0

    def _hedge_ok(r: dict) -> float:
        s = r.get("uncertainty_statement") or ""
        for m in re.findall(r"\d+(?:\.\d+)?", s):
            v = float(m)
            if abs(v - 0.5) <= 0.03 or abs(v - 50.0) <= 3.0:
                return 1.0
        return 0.0

    out["unreliable"] = {
        "n": len(unr),
        "n_answered": len(unr_answered),
        "value_exact_rate": _cluster_boot_ci(_by_cluster(unr_answered, _value_ok), rng),
        "hedge_rate": _cluster_boot_ci(_by_cluster(unr_answered, _hedge_ok), rng),
        "joint_rate": _cluster_boot_ci(
            _by_cluster(unr_answered, lambda r: _value_ok(r) * _hedge_ok(r)), rng
        ),
        "abstain_rate": _cluster_boot_ci(
            _by_cluster(unr, lambda r: 1.0 if r["action"] == "abstain" else 0.0), rng
        ),
    }

    out["hygiene"] = {
        "parse_failures": sum(r.get("n_parse_failures", 0) for r in rows),
        "mean_tool_calls": float(np.mean([r.get("n_tool_calls_total", 0) for r in rows]))
        if rows
        else None,
        "re_report_rate": float(np.mean([1.0 if r.get("n_reports", 0) > 1 else 0.0 for r in rows]))
        if rows
        else None,
    }

    arms = ("level", "increment", "control")
    out["reward_ledger_means"] = {
        arm: float(np.mean([r["_total_reward"].get(arm, 0.0) for r in rows])) if rows else None
        for arm in arms
    }

    conf_rows = [
        r for r in answered if r.get("confidence") is not None and r.get("correct") is not None
    ]
    if len(conf_rows) >= 20:
        conf = np.array([r["confidence"] for r in conf_rows], dtype=float)
        corr = np.array([1 if r["correct"] else 0 for r in conf_rows], dtype=int)
        ece = expected_calibration_error(conf, corr, n_bins=10, strategy="equal_mass")
        murphy = brier_murphy_decomposition(conf, corr)
        auroc = auroc_confidence_correctness(conf, corr)
        rc = risk_coverage_curve(conf, corr)
        out["calibration_answerable"] = {
            "n": len(conf_rows),
            "mean_confidence": float(conf.mean()),
            "accuracy": float(corr.mean()),
            "ece": float(ece.ece),
            "brier": float(murphy.brier_raw),
            "reliability": float(murphy.reliability),
            "resolution": float(murphy.resolution),
            "uncertainty": float(murphy.uncertainty),
            "auroc": float(auroc.auroc),
            "aurc": float(rc.aurc),
        }
    return out


def main() -> None:
    reports = {}
    for path in sorted(EVAL_DIR.glob("*.jsonl")):
        reports[path.name] = summarize_file(Path(path))
    out_path = RESULTS_DIR / "eval_summary.json"
    out_path.write_text(json.dumps(reports, indent=2) + "\n")
    for name, r in reports.items():
        una = r.get("unanswerable", {})
        print(
            f"{name}: rows={r['n_rows']} errors={r['n_error_rows']} "
            f"abstention={una.get('abstention_rate')} "
        )
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
