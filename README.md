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

## The study

The benchmark ships with one study. Ten model settings from six families
(Claude, Gemini, GPT, gpt-oss, Qwen, and GLM) run the same 596 questions under
the same prompts, tools, scorer, and same-day gold. The environment is a
factor of the study and not a fixed background: the harness can change how many
characters of each tool result the agent reads (300 or 4,000) and whether the
schema tool states each field's null count.

The study separates three stages of honest reporting that one disclosure rate
would collapse. The agent can *look* for the defect, it can *keep going* until
its queries succeed, and it can *tell* the reader once it holds the fact.

| Finding | Result |
|---|---|
| Retrieval | Accuracy on answered answerable items is 71% to 99% across settings, and 89% for Claude Haiku 4.5. |
| Looking | A null-revealing query appears in 46 of 113 eligible Claude Haiku 5.5 episodes with everything readable, and in 4 and 3 for gpt-5-mini and Qwen3-32B. No environment changed it. |
| Keeping going | With the cut at 300 characters, no-report runs from 9% to 70% across settings. At 4,000 characters it is 0% to 2% in the three settings re-run. |
| Telling | Handed the null count under a readable schema, Claude Haiku 5.5 and gpt-5-mini disclose on 82% and 88% of eligible items, against 12% and 40% when the count is one query away. Qwen3-32B discloses on 1% with the number in its tool result. |
| Hedges | Most nonzero disclosure rates are boilerplate. An item-by-item audit of 50 strict hits, by one language-model rater, found 14 quantified, 8 specific, 24 boilerplate, and 4 false positives. |
| Confidence | Stated confidence carries no information about correctness in any family: AUROC is at or below 0.5 in 15 of 17 run-settings, and inverted in two. |

The run order is in the paper's Appendix A. Several design choices, the cut
length, the null-count variant, the quantified scoring tier, and the audit, were
made after earlier results were seen, so those parts of the study are
exploratory in design.

## Install

```bash
git clone https://github.com/phbui/civic-honesty-benchmark
cd civic-honesty-benchmark
pip install -e ".[anthropic,openai,test]"
```

Optional: set `SOCRATA_APP_TOKEN` in your environment to raise the NYC
API rate limits (works without one).

## Quickstart

The repository ships code only; the question set, pinned gold, and episode logs live in the dated Hugging Face snapshot (see Dataset below). Fetch them first:

```bash
# Fetch the released snapshot and place questions + gold where the scripts expect them
hf download phiplusplus/civic-honesty-benchmark --repo-type dataset --local-dir hf_snapshot
mkdir -p results && cp hf_snapshot/v1_2026-08-03/*.jsonl results/
# Optionally re-materialize gold against the live API (writes fresh verification dates)
python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out results/groundtruth_canonical.jsonl
# Run a prompted evaluation episode set (reads results/questions_canonical.jsonl + results/groundtruth_canonical.jsonl by default)
python scripts/run_prompted_eval.py --provider anthropic --key-file ~/.keys/anthropic --condition neutral
# Score and summarize
python scripts/summarize_eval.py --in results/ --out eval_summary.json
```

Other vendors run through an adapter over OpenAI-compatible chat completions
(`--provider openai|gemini|deepinfra|together|groq`, see `scripts/llm_providers.py`).
The environment factors are flags of the same driver:

```bash
# the 4,000-character cut with null counts in the schema tool (the readable cell of the two-by-two)
python scripts/run_prompted_eval.py --provider anthropic --key-file ~/.keys/anthropic \
  --model claude-haiku-5-5 --omit-temperature --extra-json '{"thinking":{"type":"disabled"}}' \
  --conditions neutral --result-max-chars 4000 \
  --schema-null-counts results/field_null_rates_2026-10-17.json --i-know-this-costs-money
```

`scripts/launchers/` holds the exact commands of the 2026-10-17 and 2026-10-19 runs.
Every run stamps its environment flags, adapter statistics, and gold drift check
into its summary file.

## Dataset

Question sets, pinned gold labels, and episode logs live on Hugging
Face: [`phiplusplus/civic-honesty-benchmark`](https://huggingface.co/datasets/phiplusplus/civic-honesty-benchmark). Because the benchmark is live-materialized,
the pinned gold is a dated snapshot; the relabeling protocol
(`scripts/groundtruth.py`) is the documented refresh procedure, and
drift between snapshots is a measured property of the benchmark, not an
error. See the dataset card for the dated-release table and schema.
The v1 snapshot holds the Claude Haiku 4.5 logs. The question set and gold are unchanged for the ten-setting study, and the logs of the other nine settings are being added as a v2 snapshot.

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

## Figures

`figures/` holds plotting-code smoke tests rendered from synthetic data
(see the plot titles), not reported results.

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

Dependency versions in `pyproject.toml` are intentionally unpinned; the
data layer is stdlib-only and the analysis scripts state their own
requirements.
