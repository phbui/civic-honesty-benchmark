#!/bin/bash
# Native Groq run of openai/gpt-oss-120b on a stratified 90-question subset (33 answerable, 33 unanswerable,
# 24 unreliable), neutral only, reasoning low. Groq's free tier gives this model 200,000 tokens per day and
# 8,000 per minute, so the run is paced (--min-request-interval) and patient (--patient-rate-limits): each 429
# states its wait and the adapter sleeps it out. Resume-safe: rows are append-only and keyed by question id,
# so a restart continues. Test override: PROVIDER_ARGS and OUT may be set from the environment.
cd "$(dirname "$0")/../.." || exit 1
OUT=${OUT:-results/exp_gptoss_groq_subset90_2026-10-19}
mkdir -p $OUT/logs
PROVIDER_ARGS=${PROVIDER_ARGS:---provider groq --key-file $HOME/.keys/groq}
echo "$(date -u +%FT%TZ) START" >> $OUT/logs/launcher.log
python scripts/run_prompted_eval.py $PROVIDER_ARGS --model openai/gpt-oss-120b \
  --extra-json '{"reasoning_effort":"low"}' --questions results/questions_groq_subset90.jsonl \
  --groundtruth results/gold_2026-10-19_groq_subset90.jsonl --conditions neutral --workers 2 \
  --min-request-interval ${MIN_INTERVAL:-9} --patient-rate-limits --retry-errors \
  --out-dir $OUT --i-know-this-costs-money > $OUT/logs/groq_subset.log 2>&1
rc=$?
echo $rc > $OUT/logs/groq_subset.done
echo "$(date -u +%FT%TZ) EXIT rc=$rc" >> $OUT/logs/launcher.log
