#!/usr/bin/env python3
"""Render the README figures from docs/results_2026-10.json.

    uv run --with matplotlib python scripts/make_figures.py

Writes docs/ten-setting-contrast.png (and, with --paper-pdf PATH, the same figure as a PDF for the manuscript),
docs/disclosure-by-setting.png, docs/two-by-two.png and docs/confidence-auroc.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "docs" / "results_2026-10.json").read_text())
INK, GREY, ACCENT, SOFT = "#1f2933", "#9aa5b1", "#c0392b", "#2c7fb8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def ten_setting_contrast(paper_pdf=None):
    rows = sorted(D["settings"], key=lambda r: r["disclosed"] / r["eligible"])
    names = [r["name"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    for i, r in enumerate(rows):
        a.barh(i, r["caveat_pct"], color="#dbe4ec", height=0.62)
        p = 100 * r["disclosed"] / r["eligible"]
        a.plot(p, i, "o", color=ACCENT, ms=7, zorder=3)
        a.text(103, i, f"{r['disclosed']}/{r['eligible']}", va="center", fontsize=8, color=INK)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels(names, fontsize=9)
    a.set_xlim(0, 118)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel("%  (bar: documented caveat carried; dot: missingness disclosed)", fontsize=8.5)
    a.set_title("(a) Documented caveat carried, missingness disclosed", loc="left", fontsize=9.5, color=INK)
    parts = [("abstain_verified", "Abstained after a query", SOFT), ("abstain_cold", "Abstained before any query", "#f0b27a"), ("fabricate_n", "Fabricated an answer", ACCENT), ("no_report_n", "No report", GREY)]
    for i, r in enumerate(rows):
        left = 0
        for key, _, col in parts:
            w = 100 * r[key] / r["unanswerable_episodes"]
            b.barh(i, w, left=left, color=col, height=0.62)
            left += w
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([])
    b.set_xlim(0, 100)
    b.set_xlabel("% of unanswerable episodes", fontsize=8.5)
    b.set_title("(b) Outcome on unanswerable questions", loc="left", fontsize=9.5, color=INK)
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], labels=[l for _, l, _ in parts], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "ten-setting-contrast.png", dpi=170)
    if paper_pdf:
        fig.savefig(paper_pdf)
    plt.close(fig)


def disclosure_by_setting():
    rows = sorted(D["settings"], key=lambda r: r["disclosed"] / r["eligible"])
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, r in enumerate(rows):
        p = 100 * r["disclosed"] / r["eligible"]
        lo, hi = (100 * v for v in wilson(r["disclosed"], r["eligible"]))
        ax.plot([lo, hi], [i, i], color=GREY, lw=2, solid_capstyle="round")
        ax.plot(p, i, "o", color=ACCENT, ms=7)
        ax.text(max(hi, p) + 2, i, f"{r['disclosed']}/{r['eligible']}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows])
    ax.set_xlim(0, 60)
    ax.set_xlabel("Strict missingness disclosure on eligible answered items (%), Wilson 95% interval")
    ax.set_title("Disclosure is rare and varies by model", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "disclosure-by-setting.png", dpi=170)
    plt.close(fig)


def two_by_two():
    T = D["two_by_two"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True)
    colors = [GREY, "#bdc7d1", "#e8a09a", ACCENT]
    for ax, m in zip(axes, T["models"]):
        for i, (k, n) in enumerate(zip(m["disclosed"], m["eligible"])):
            p = 100 * k / n
            ax.bar(i, p, color=colors[i], width=0.72)
            ax.text(i, p + 2, f"{k}/{n}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["300\nno counts", "4,000\nno counts", "300\ncounts", "4,000\ncounts"], fontsize=8)
        ax.set_title(m["name"], color=INK, fontsize=10)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Strict disclosure (%)")
    fig.suptitle("Handed the null count under a readable schema, two models tell and one does not", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ROOT / "docs" / "two-by-two.png", dpi=170)
    plt.close(fig)


def confidence_auroc():
    rows = sorted(D["settings"], key=lambda r: r["auroc"])
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, r in enumerate(rows):
        if r["auroc_ci"]:
            ax.plot(r["auroc_ci"], [i, i], color=GREY, lw=2, solid_capstyle="round")
        ax.plot(r["auroc"], i, "o", color=ACCENT if r["auroc"] <= 0.5 else SOFT, ms=7)
    ax.axvline(0.5, color=INK, lw=1, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows])
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("AUROC of stated confidence against correctness (0.5 is a coin flip); Gemini states 1.000 on every answer")
    ax.set_title("Stated confidence orders nothing in most settings", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "confidence-auroc.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-pdf", default=None)
    args = ap.parse_args()
    ten_setting_contrast(args.paper_pdf)
    disclosure_by_setting()
    two_by_two()
    confidence_auroc()
    print("wrote docs/*.png")
