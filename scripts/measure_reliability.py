"""measure_reliability.py — test-retest reliability of the pavement rating, measured
on the live panel.

WHY THIS EXISTS. The benchmark's unreliable question class grades reports against a
documented reliability constant of R ~ 0.50: the claim that roughly half the variance
in a rating does not survive to the next survey occasion. This script computes that
constant from the public panel itself, so the number reproduces from the release
rather than resting on an external citation.

METHOD. Pull (oftcode, systemrating, inspection) for every row of the dataset;
collapse same-segment same-date rows to their mean rating; sort each segment's
occasions by inspection date; form consecutive occasion pairs; the Pearson
correlation over those pairs is the test-retest estimate. Two variants are reported:
all consecutive pairs, and pairs whose gap falls in the 180-730 day band, which is
the year-over-year comparison an honest condition report actually faces. Note the
scope this method carries: real deterioration between occasions is folded into the
noise estimate, so this is reliability across survey occasions, not same-day
repeatability.

MEASURED 2026-08-30 against the live table (514,521 rows, 103,711 segments, 84,889
with two or more occasions): r = 0.4775 over 373,881 consecutive pairs (median gap
423 days); r = 0.4441 over the 301,343 pairs in the 180-730 day band.

STDLIB ONLY, like socrata.py: the arithmetic is a single Pearson correlation and
needs no numpy. Run:

    python scripts/measure_reliability.py [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date

from socrata import SocrataClient

DATASET_ID = "6yyb-pb25"  # NYC Street Pavement Ratings, per agent_env.py
PAGE_SIZE = 100_000
YEAR_BAND_DAYS = (180, 730)


def pearson(pairs: list[tuple[float, float]]) -> float:
    """Pearson correlation over (x, y) pairs. Returns nan below two pairs or when
    either margin is constant."""
    n = len(pairs)
    if n < 2:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def collapse_occasions(rows: list[dict]) -> dict[str, list[tuple[date, float]]]:
    """Group rows by segment, collapse same-date rows to their mean rating, and
    return each segment's occasions sorted by date. Rows missing any of the three
    fields, or with an unparseable rating or date, are skipped."""
    ratings: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            ratings[r["oftcode"]][r["inspection"][:10]].append(float(r["systemrating"]))
        except (KeyError, TypeError, ValueError):
            continue
    out: dict[str, list[tuple[date, float]]] = {}
    for seg, by_date in ratings.items():
        try:
            occs = sorted((date.fromisoformat(d), sum(v) / len(v)) for d, v in by_date.items())
        except ValueError:
            continue
        out[seg] = occs
    return out


def consecutive_pairs(
    occasions: dict[str, list[tuple[date, float]]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[int]]:
    """Form consecutive occasion pairs per segment. Returns (all_pairs,
    year_band_pairs, gaps_in_days); year_band_pairs is the subset whose gap falls
    inside YEAR_BAND_DAYS inclusive."""
    all_pairs: list[tuple[float, float]] = []
    band_pairs: list[tuple[float, float]] = []
    gaps: list[int] = []
    lo, hi = YEAR_BAND_DAYS
    for occs in occasions.values():
        for (d1, r1), (d2, r2) in zip(occs, occs[1:]):
            gap = (d2 - d1).days
            all_pairs.append((r1, r2))
            gaps.append(gap)
            if lo <= gap <= hi:
                band_pairs.append((r1, r2))
    return all_pairs, band_pairs, gaps


def test_retest(rows: list[dict]) -> dict:
    """The full computation on already-fetched rows; separated from fetching so it
    is testable on fixtures."""
    occasions = collapse_occasions(rows)
    all_pairs, band_pairs, gaps = consecutive_pairs(occasions)
    gaps.sort()
    return {
        "rows": len(rows),
        "segments": len(occasions),
        "segments_with_2plus_occasions": sum(1 for o in occasions.values() if len(o) >= 2),
        "consecutive_pairs": len(all_pairs),
        "pearson_r_all_pairs": round(pearson(all_pairs), 4) if len(all_pairs) >= 2 else None,
        "median_gap_days": gaps[len(gaps) // 2] if gaps else None,
        "pairs_gap_180_730d": len(band_pairs),
        "pearson_r_180_730d": round(pearson(band_pairs), 4) if len(band_pairs) >= 2 else None,
    }


def fetch_panel(client: SocrataClient) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        page, _ = client.query(
            DATASET_ID,
            "SELECT oftcode, systemrating, inspection "
            f"ORDER BY oftcode, inspection LIMIT {PAGE_SIZE} OFFSET {offset}",
        )
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="write the summary JSON here as well as stdout")
    args = ap.parse_args()
    result = test_retest(fetch_panel(SocrataClient()))
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
