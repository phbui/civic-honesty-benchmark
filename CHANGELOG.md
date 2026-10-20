# Changelog

All notable changes. The version in `VERSION`, `pyproject.toml`, and `CITATION.cff` match the newest entry.

## 1.2.0, 2026-10-20

- Data: the readable-schema cells (4,000-character cut, with and without null counts) for all ten settings, so that every setting carries the full environment design that three had before. 14 runs of 596 episodes, same-day gold with no drift.
- Analysis: `scripts/paper_numbers.py` recomputes every reported number, including template-clustered intervals, from the released files and `docs/cells_2026-10.json`, with tests in `scripts/test_paper_numbers.py`.
- Findings: six of ten settings disclose missingness when handed the null count, and four do not. The earlier three-setting comparison is superseded.
- Figures: PDF fonts embedded as Type 42, and the two-by-two figure now shows all ten settings.

## 1.1.0, 2026-10-20

The benchmark is presented as one study of ten model settings from six families, with the evaluation environment as an experimental factor.

- Driver: an adapter over OpenAI-compatible chat-completions endpoints (`scripts/llm_providers.py`) that runs OpenAI, Gemini, Groq, DeepInfra, and Together models through the same policy, prompts, tools, and scorer, and counts every translation event in the run summary.
- Driver: environment factors as stamped, opt-in flags: `--result-max-chars` (the tool-result cut, default 300), `--schema-null-counts` (the schema tool states each sparse field's null count), `--max-turns`, `--min-request-interval`, `--patient-rate-limits`, `--describe-dataset-arg`, and `--recover-reasoning-calls`.
- Scorer: a quantified tier beside the lenient and strict tiers, and a fix for a false positive on the literal string "null" that had credited Gemini with strict disclosures; the published Claude Haiku 4.5 counts are unchanged by the fix.
- Data: the `v2_2026-10` snapshot on Hugging Face, with the episode logs of nine further settings, the cut-length and null-count runs, repeats, and the gold files with their drift checks.
- Docs: a datasheet (`docs/DATASHEET.md`), Croissant metadata with Responsible AI fields, figures regenerated from `docs/results_2026-10.json`, and the exact launch commands of the 2026-10 runs in `scripts/launchers/`.
- Repository: CI, issue and pull-request templates, contribution guide, security policy, and code of conduct.

## 1.0.0, 2026-08-05

First public release: 596 questions, same-day gold with drift control, the multi-turn tool environment, the matched confidence schema, three payment ledgers, disclosure scoring, calibration diagnostics, and the Claude Haiku 4.5 logs (`v1_2026-08-03`, `v1_2026-08-05_reencounter`).
