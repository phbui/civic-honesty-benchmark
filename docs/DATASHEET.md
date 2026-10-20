# Datasheet

Following the headings of Datasheets for Datasets (Gebru et al., 2021). The Hugging Face dataset card carries the same content in the card format, and `croissant.json` carries it in machine-readable form.

## Motivation

**Why was the dataset created?** To test whether tool-using language-model agents report honestly about three defects of a municipal open-data table: questions the data cannot answer (unanswerable), values that rest on a noisy instrument (unreliable), and fields that are mostly empty (missing). Existing benchmarks score retrieval accuracy or abstention on static text. None scores whether an agent discloses missingness it could have found by querying.

**Who made it?** One author, a doctoral student. The repository declares no funder.

## Composition

**What are the instances?** 596 questions over New York City's Street Pavement Rating table (Socrata `6yyb-pb25`): 220 answerable, 220 unanswerable by construction (field-absent or record-absent), and 156 answerable but unreliable, generated from 54 templates. Each question carries its SoQL query, a same-day gold answer with verification date, and its class and mechanism labels. The episode logs hold, for each run, the full transcript of tool calls, tool results (cut to the harness's character budget), and structured reports with stated confidence.

**Is it a sample?** The questions are a deterministic generation (seed recorded in `questions_canonical_meta.json`) over all 13 columns of the table and its five boroughs. Items from one template are near-duplicates, so the effective sample size is far below 596 (realized 79.1, 49.2, and 21.3 by class).

**Is there missing data?** By design. Three of the 13 columns are mostly empty (direction and road type 91.3% null, non-rating reason 85.8%), and 113 of the 220 answerable questions touch them.

**Does it contain personal or sensitive information?** No. Rows describe street segments, not people. The city publishes the table without use restrictions.

## Collection process

The table is read live through the city's public SODA API. Gold answers are materialized by executing each question's own query against the API on the day of a run, with caching bypassed, and re-executed after the run; every released run reproduced all 596 values exactly. Three days after the first materialization, 170 of 596 values (28.5%) had changed under a live update, which is why same-day gold and the drift check are part of the protocol and not an option.

The reliability constant R of about 0.50 used by the unreliable class is measured from the public panel itself (consecutive-occasion Pearson correlation, r = 0.4775 over 373,881 pairs), not cited from elsewhere, with San Francisco's panel (r = 0.6418) as a cross-check.

## Preprocessing

None of the table. The release holds derived content only: questions, gold values, null-rate summaries, and agent episodes. It never redistributes the table.

## Uses

**Intended.** Evaluating whether an agent reports unanswerability, measurement uncertainty, and missingness honestly; studying how the evaluation environment (the length of tool results the agent can read, whether the schema tool states null counts) changes what an evaluation sees; comparing the stages of looking for a defect, continuing until queries succeed, and telling the reader.

**Out of scope.** A deployment license for any agent in either direction; a measure of retrieval ability on other tables or cities without new gold; training a model against the disclosure endpoint, which could teach caveat-parroting in place of investigation (hold the endpoint out and vary the touched fields); inference about individual pavement segments.

**Known gaps.** One dataset in one city. Disclosure scoring is programmatic and counts hedges as disclosure, which is why a quantified tier and an item-by-item audit sit beside the lenient and strict tiers. Live data can change before replication; the release pins the materialized gold.

## Distribution

Code: MIT, on GitHub. Questions, pinned gold, field null rates, and episode logs: CC-BY-4.0, on Hugging Face in dated snapshots (`v1_2026-08-03`, `v1_2026-08-05_reencounter`, `v2_2026-10`). Snapshots are immutable and tagged.

## Maintenance

The maintainer will answer issues and accept new-model results that follow the protocol and report a gold drift check. A new gold materialization is a new dated snapshot and never an edit of an old one. Corrections to a released run are recorded in `CHANGELOG.md` with the affected files.
