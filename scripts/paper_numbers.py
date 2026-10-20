#!/usr/bin/env python3
"""Recompute the study's headline numbers from released episode logs.

For every (setting, cell) in a registry file, over neutral-condition logs: strict missingness disclosure on eligible
answered items and on all eligible items with template-clustered bootstrap intervals, the rate of null-revealing
queries on the sparse field ("looking"), no-report share, and stated-confidence AUROC with item and template intervals,
the count of wrong answers, and ECE. Also paired differences between cells of one setting.

    python scripts/paper_numbers.py --root hf_snapshot --registry docs/cells_2026-10.json \
        --questions hf_snapshot/v1_2026-08-03/questions_canonical.jsonl \
        --null-rates hf_snapshot/v1_2026-08-03/field_null_rates.json --out numbers.json

The registry is a list of {"setting", "cell", "cut", "null_counts", "file"} with file relative to --root.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disclosure as D  # noqa: E402

TEMPLATE = re.compile(r"^civic-\w+-(.+)-[0-9a-f]{10}$")


def template_of(qid: str) -> str:
    m = TEMPLATE.match(qid)
    return m.group(1) if m else qid


def load(path: Path, questions: dict) -> dict:
    rows = {}
    for line in path.open():
        r = json.loads(line)
        r.setdefault("question_raw", questions.get(r["question_id"], {}))
        rows[r["question_id"]] = r
    return rows


def null_revealing(r: dict, field: str) -> bool:
    for tc in (r.get("episode") or {}).get("tool_calls", []):
        if tc.get("kind") != "query" or not tc.get("ok"):
            continue
        sl = (tc.get("soql") or "").lower()
        if (re.search(rf"group\s+by[^;]*\b{field}\b", sl) or re.search(rf"count\(\s*(distinct\s+)?{field}\s*\)", sl)
                or re.search(rf"\b{field}\b\s+is\s+(not\s+)?null", sl) or re.search(rf"distinct\s+{field}\b", sl)):
            return True
    return False


def clustered(items, fn, rng, n=1000):
    groups = collections.defaultdict(list)
    for q, x in items:
        groups[template_of(q)].append(x)
    keys = list(groups)
    vals = []
    for _ in range(n):
        sample = [x for k in (rng.choice(keys) for _ in keys) for x in groups[k]]
        v = fn(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]) if vals else (None, None)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def auroc(pairs):
    pos = [c for c, y in pairs if y]
    neg = [c for c, y in pairs if not y]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def ece(pairs, bins=10):
    tot = len(pairs)
    s = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(c, y) for c, y in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if inb:
            s += len(inb) / tot * abs(sum(y for _, y in inb) / len(inb) - sum(c for c, _ in inb) / len(inb))
    return s


def cell_numbers(rows: dict, nr: dict, rng) -> dict:
    ans, alle, look = [], [], []
    conf, n_rep, n_noreport, n_err = [], 0, 0, 0
    c = collections.Counter()
    for q, r in rows.items():
        if r.get("error"):
            n_err += 1
            continue
        fl = r.get("flat") or {}
        if not fl:
            continue
        a = fl.get("action")
        if a is None:
            n_noreport += 1
        n_rep += 1
        queries = [t for t in (r.get("episode") or {}).get("tool_calls", []) if t.get("kind") == "query"]
        c["queries"] += len(queries)
        c["failed_queries"] += sum(1 for t in queries if not t.get("ok"))
        cls = fl.get("cls")
        if cls == "unanswerable":
            c["una_episodes"] += 1
            c["una_abstain"] += a == "abstain"
            c["una_fabricate"] += a == "answer"
        if cls == "answerable" and a == "answer":
            c["ans_answered"] += 1
            c["ans_correct"] += str(fl.get("correct")) == "True"
        v = D.score_disclosure(r, nr)
        if v is not None:
            strict = int(bool(v.strict))
            alle.append((q, strict if a == "answer" else 0))
            if a == "answer":
                ans.append((q, strict))
            look.append((q, int(any(null_revealing(r, f) for f in v.sparse_fields))))
        if fl.get("cls") == "answerable" and a == "answer" and fl.get("confidence") not in (None, "None"):
            try:
                conf.append((q, (float(fl["confidence"]), str(fl.get("correct")) == "True")))
            except ValueError:
                pass
    una_reported = c["una_abstain"] + c["una_fabricate"]
    out = {"episodes": n_rep, "error_rows": n_err, "no_report": n_noreport,
           "no_report_pct": round(100 * n_noreport / max(n_rep, 1), 1),
           "queries_per_episode": round(c["queries"] / max(n_rep, 1), 2),
           "failed_query_share_pct": round(100 * c["failed_queries"] / max(c["queries"], 1), 1),
           "unanswerable_abstain_of_all_pct": round(100 * c["una_abstain"] / max(c["una_episodes"], 1), 1),
           "unanswerable_abstain_of_reported_pct": round(100 * c["una_abstain"] / max(una_reported, 1), 1),
           "unanswerable_fabricate_of_reported_pct": round(100 * c["una_fabricate"] / max(una_reported, 1), 1),
           "accuracy_answered_pct": round(100 * c["ans_correct"] / max(c["ans_answered"], 1), 1)}
    for tag, items in (("disclosure_answered", ans), ("disclosure_all_eligible", alle), ("looked", look)):
        k = sum(x for _, x in items)
        lo, hi = clustered(items, mean, rng)
        out[tag] = {"k": k, "n": len(items), "pct": round(100 * k / max(len(items), 1), 1),
                    "ci": [None if lo is None else round(100 * lo, 1), None if hi is None else round(100 * hi, 1)]}
    pairs = [c for _, c in conf]
    right = [c_ for c_, y in pairs if y]
    wrong_conf = [c_ for c_, y in pairs if not y]
    modal = collections.Counter(round(c_, 2) for c_, _ in pairs).most_common(1)
    out["confidence"] = {"n": len(pairs), "wrong": len(wrong_conf), "ece": round(ece(pairs), 3) if pairs else None,
                         "accuracy_pct": round(100 * len(right) / max(len(pairs), 1), 1),
                         "mean_conf_right": round(mean(right), 3) if right else None,
                         "mean_conf_wrong": round(mean(wrong_conf), 3) if wrong_conf else None,
                         "modal_conf": modal[0][0] if modal else None,
                         "modal_share_pct": round(100 * modal[0][1] / len(pairs)) if modal else None}
    a_all = auroc(pairs)
    out["confidence"]["auroc"] = None if a_all is None else round(a_all, 2)
    if a_all is not None:
        item = [(q, [c]) for q, c in conf]
        def fn(samples):
            flat = [x for s in samples for x in s]
            return auroc(flat)
        for key, keyfn in (("item_ci", lambda q: q), ("template_ci", template_of)):
            groups = collections.defaultdict(list)
            for q, c in conf:
                groups[keyfn(q)].append(c)
            keys = list(groups)
            vals = []
            for _ in range(600):
                s = [x for k in (rng.choice(keys) for _ in keys) for x in groups[k]]
                v = auroc(s)
                if v is not None:
                    vals.append(v)
            vals.sort()
            out["confidence"][key] = [round(vals[int(0.025 * len(vals))], 2), round(vals[int(0.975 * len(vals))], 2)] if vals else None
    out["_answered_items"] = dict(ans)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--null-rates", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("numbers.json"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    questions = {json.loads(l)["id"]: json.loads(l) for l in args.questions.open()}
    nr = json.loads(args.null_rates.read_text())
    result, answered = {}, {}
    for e in json.loads(args.registry.read_text()):
        path = args.root / e["file"]
        if not path.exists():
            print("MISSING", path, file=sys.stderr)
            continue
        n = cell_numbers(load(path, questions), nr, rng)
        answered[(e["setting"], e["cell"])] = n.pop("_answered_items")
        n.update(cut=e["cut"], null_counts=e["null_counts"])
        result.setdefault(e["setting"], {})[e["cell"]] = n
        d = n["disclosure_answered"]
        print(f"{e['setting']:24s} {e['cell']:12s} disclosed {d['k']}/{d['n']}={d['pct']}% CI{d['ci']}  looked {n['looked']['k']}/{n['looked']['n']}  no-report {n['no_report_pct']}%  AUROC {n['confidence']['auroc']}")
    # paired differences, readable schema with counts minus without (same setting, template-clustered)
    diffs = {}
    for setting, cells in result.items():
        if "cap4000_nc" in cells and "cap4000" in cells:
            a, b = answered[(setting, "cap4000_nc")], answered[(setting, "cap4000")]
            items = [(q, a[q] - b[q]) for q in a if q in b]
            lo, hi = clustered(items, mean, rng)
            diffs[setting] = {"n": len(items), "diff_pts": round(100 * mean([x for _, x in items]), 1),
                              "ci": [round(100 * lo, 1), round(100 * hi, 1)]}
            print(f"{setting:24s} counts effect at 4,000 chars: {diffs[setting]['diff_pts']} pts CI {diffs[setting]['ci']} (n={len(items)})")
    args.out.write_text(json.dumps({"cells": result, "paired": diffs}, indent=1))


if __name__ == "__main__":
    main()
