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

```bash
# Re-materialize gold labels against the live API (writes verification dates)
python scripts/groundtruth.py --questions data/questions_canonical.jsonl --out gold.jsonl
# Run a prompted evaluation episode set
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
