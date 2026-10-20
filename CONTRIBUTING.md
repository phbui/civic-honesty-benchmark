# Contributing

Issues and pull requests are welcome. This page says what a useful one looks like.

## Before you open a pull request

1. Run the tests: `python -m pytest scripts -q`. The suite makes no live API call and spends no money. Tests that need the released snapshot skip and print why.
2. If you change the scorer (`scripts/disclosure.py`) or the driver (`scripts/run_prompted_eval.py`), rescore a released run and say whether any published number moved. The released logs are on Hugging Face, and the scorer reproduces every disclosure number from them without a model call.
3. If you add a provider, add it to `scripts/llm_providers.py` with a fake-client test that exercises its quirks (argument-free tool calls, reasoning fields, parallel calls). The adapter counts every translation event, and a new provider should count its own.

## What a good pull request looks like

One change per pull request, a test for every behaviour change, and a commit message that says what changed and why in plain sentences.

## What will not be merged

Anything that bundles the questions, gold, or episode logs; they live on Hugging Face under their own license and are fetched at run time. Anything that puts a paid call on a default path. Anything that changes a gold label without the relabeling protocol in `scripts/groundtruth.py` and a drift check.

## Adding a city or a table

The benchmark generalizes to any Socrata table. Add question templates in `scripts/questions.py`, regenerate gold with `scripts/groundtruth.py` on the day of the run, and report the three defect classes separately. Open an issue first so that the question classes stay comparable.

## Reporting a bug

Use the bug report template and say which snapshot you used. For a security problem, read `SECURITY.md`.
