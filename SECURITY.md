# Security policy

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting at https://github.com/phbui/civic-honesty-benchmark/security/advisories/new. Do not open a public issue for a vulnerability. You will get a first reply within 14 days.

## Supported versions

The `main` branch and the most recent tagged release receive fixes.

## What this software does on your machine

It sends queries to the NYC Open Data (Socrata) API and, when you run an evaluation, to the model vendor you choose. It reads API keys only from the key file you pass with `--key-file`, passes them to the vendor client, and never logs or writes them. Paid runs refuse to start without `--i-know-this-costs-money`. The episode logs contain the prompts, the tool results, and the model reports, and no key.
