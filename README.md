# Civic Honesty Benchmark

A live-data benchmark asking whether language-model agents, given real
query access to a municipal open-data API, report honestly about three
things the data itself cannot answer for them: what is knowable, what is
unknowable by construction, and what is answerable but unreliable.

596 questions over New York City's Street Pavement Rating dataset
(220 answerable, 220 unanswerable by construction, 156 answerable but
unreliable), materialized live: every gold answer is computed by
executing the question's own query against the same live API the agent
uses, and every gold label carries its verification date.

## Install

```bash
git clone https://github.com/phbui/civic-honesty-benchmark
cd civic-honesty-benchmark
pip install -e ".[anthropic,test]"
```

Optional: set `SOCRATA_APP_TOKEN` in your environment to raise the NYC
API rate limits (works without one).

## Quickstart

The repository ships code only; the question set, pinned gold, and episode logs live in the dated Hugging Face snapshot (see Dataset below). Fetch them first:

```bash
# Fetch the released snapshot and place questions + gold where the scripts expect them
huggingface-cli download phiplusplus/civic-honesty-benchmark --repo-type dataset --local-dir hf_snapshot
mkdir -p results && cp hf_snapshot/v1_2026-08-03/*.jsonl results/
# Optionally re-materialize gold against the live API (writes fresh verification dates)
python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out results/groundtruth_canonical.jsonl
# Run a prompted evaluation episode set (reads results/questions_canonical.jsonl + results/groundtruth_canonical.jsonl by default)
python scripts/run_prompted_eval.py --provider anthropic --key-file ~/.keys/anthropic --condition neutral
# Score and summarize
python scripts/summarize_eval.py --in results/ --out eval_summary.json
```

## Dataset

Question sets, pinned gold labels, and episode logs live on Hugging
Face: [`phiplusplus/civic-honesty-benchmark`](https://huggingface.co/datasets/phiplusplus/civic-honesty-benchmark). Because the benchmark is live-materialized,
the pinned gold is a dated snapshot; the relabeling protocol
(`scripts/groundtruth.py`) is the documented refresh procedure, and
drift between snapshots is a measured property of the benchmark, not an
error. See the dataset card for the dated-release table and schema.

## Scoring

Three parallel reward ledgers per reported turn (a level-paid bounded
proper score, its market-scoring-rule increment, and a zero control),
disclosure scoring under lenient and strict criteria, calibration
diagnostics (ECE, Brier with its Murphy decomposition, AUROC,
risk-coverage), and the benchmark's own statistical ceiling
(`scripts/power.py`).

## Measurement uncertainty

The unreliable class grades reports against a documented test-retest
constant of R ≈ 0.50, computed from the public panel itself rather than
cited: `scripts/measure_reliability.py` measures NYC's pavement rating
(consecutive-occasion Pearson; r = 0.4775 over 373,881 pairs, r = 0.4441
in the 180-730 day band, 2026-08-30), and
`scripts/measure_reliability_sf.py` runs the identical method on San
Francisco's PCI panel as a cross-check (r = 0.6418 / 0.7051,
2026-08-31). Both run against the live APIs and reproduce from public
data; fixture tests live in `scripts/test_measure_reliability.py`.

## Tests

```bash
pytest scripts/
```

## Citation

See `CITATION.cff` (GitHub's "Cite this repository" button). Paper
reference to be added on publication.

## License

Code: MIT (see `LICENSE`). The question sets and gold labels on Hugging
Face are licensed CC-BY-4.0; the underlying NYC pavement data is NYC
Open Data (no restrictions on use per the NYC Open Data FAQ; see the
dataset card for the full attribution and scoping notice).
