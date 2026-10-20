---
name: Model result
about: Report a run of a new model or setting through the protocol
labels: results
---

**Setting**

Model, vendor, reasoning setting, temperature handling, and the exact `run_prompted_eval.py` command. Say whether the adapter recovered or dropped any tool calls (the run summary counts them).

**Environment factors**

`--result-max-chars`, `--schema-null-counts`, and the turn budget.

**Results**

The summary file, the disclosure report, and the gold drift check. A result without a drift check cannot be compared.
