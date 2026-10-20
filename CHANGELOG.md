# Changelog

All notable changes. The version in `VERSION`, `pyproject.toml`, and `CITATION.cff` match the newest entry.

## 1.3.0, 2026-10-20

- Driver: a `nullcheck_exploratory` condition in `scripts/run_prompted_eval.py`, the neutral prompt plus one paragraph that tells the agent to check the null share of every field its query filters or groups on and to say so if a field is mostly null; the launcher is `scripts/launchers/run_wave_h_2026-10-20.sh`.
- Data: the prompt-only arm on five settings (Claude Haiku 4.5, Gemini 3.5 Flash-Lite, gpt-5-mini at minimal reasoning, GLM-5.3-Flash, Qwen3.8-Flash) at the 4,000-character cut without null counts, `v2_2026-10/runs/exp_nullcheck_h_2026-10-20/` on Hugging Face, same-day gold. The Claude Haiku 4.5 log carries 597 client error rows ahead of its 596 valid rows; `paper_numbers.py` keeps the last row per question.
- Analysis: registry cell `cap4000_nullcheck` in `docs/cells_2026-10.json` (the new entries also carry `condition`), and a `prompt_only` block in `docs/results_2026-10.json`.
- Findings: the paragraph raises strict disclosure on eligible answered items from 0% to 27% to 74% to 99% in all five settings, Gemini included, and the share of episodes with a null-revealing query to 62% to 100%, so the neutral-prompt zero is a default and not a ceiling. Exploratory, one run per setting, chosen after the other cells were seen.

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
