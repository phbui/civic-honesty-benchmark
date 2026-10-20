# Launchers

The shell scripts here are the exact launch commands of the runs dated 2026-10-17 and 2026-10-19, with the working directory and interpreter paths edited so they run from a clone of this repository (`cd` to the repository root, `python` on the path). Each takes API keys from `~/.keys/<vendor>` and passes `--i-know-this-costs-money`, so running one spends real money.

The per-setting flags (reasoning level, temperature handling, extra request fields) are stamped into every run's `summary__*.json`, which is the authority for the runs that these scripts do not cover.

`make_groq_subset_2026-10-19.py` writes the stratified 90-question subset (33 answerable, 33 unanswerable, 24 unreliable, seed 20261019) that the native check of gpt-oss-120b uses.
