"""measure_reliability_sf.py — the San Francisco counterpart to measure_reliability.py.

WHY THIS EXISTS. The paper's measurement-uncertainty claim is checked against a
second, independently maintained municipal panel: San Francisco's Pavement
Condition Index history on data.sfgov.org. The method is byte-identical to the NYC
script (same collapse, pairing, and Pearson logic, imported rather than copied), so
the two cities' numbers are directly comparable. No benchmark questions are drawn
from this table; it exists purely as the cross-check.

METHOD FIDELITY NOTE. SF's table carries a `treatment_or_survey` field
(Survey/Treatment) that the NYC table has no analogue of. No filtering on it is
applied — mirroring the NYC method exactly — so repaving events between occasions
are folded into the noise estimate the same way real deterioration is in NYC.
SF scores are 0-100 PCI against NYC's 1-10 rating; Pearson r is scale-invariant.

MEASURED 2026-08-31 against the live table (307,321 rows, 18,373 segments, 14,252
with two or more occasions): r = 0.6418 over 282,095 consecutive pairs (median gap
394 days); r = 0.7051 over the 210,800 pairs in the 180-730 day band. Compare NYC:
r = 0.4775 / 0.4441 — rescan noise is substantial in both cities and larger in NYC.

Run:

    python scripts/measure_reliability_sf.py [--out result_sf.json]
"""

from __future__ import annotations

import argparse
import json

from measure_reliability import PAGE_SIZE, test_retest
from socrata import SocrataClient

SF_DOMAIN = "data.sfgov.org"
SF_DATASET_ID = "78va-8dhi"  # SF Pavement Condition Index (survey + treatment events)
SF_FIELDS = {"segment_field": "cnn", "rating_field": "pci_score", "date_field": "pci_change_date"}


def fetch_sf_panel(client: SocrataClient) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        page, _ = client.query(
            SF_DATASET_ID,
            "SELECT cnn, pci_score, pci_change_date "
            f"ORDER BY cnn, pci_change_date LIMIT {PAGE_SIZE} OFFSET {offset}",
        )
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="write the summary JSON here as well as stdout")
    args = ap.parse_args()
    result = test_retest(fetch_sf_panel(SocrataClient(domain=SF_DOMAIN)), **SF_FIELDS)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
