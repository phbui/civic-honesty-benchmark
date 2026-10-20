#!/bin/bash
# Wave G addendum (2026-10-20): Claude Haiku 4.5 at temperature 0 needs anthropic 0.x (1.x removed the temperature argument).
# Same two cells as wave G, same gold, via ../.venv-anth0.
cd ~/artifacts/civic-second-family/civic
D=2026-10-20; GOLD=results/gold_$D.jsonl; S=results/wave_g_$D.status
log(){ echo "$(date -u +%FT%TZ) $*" >> $S; }
log HAIKU45_START
G="--groundtruth $GOLD --conditions neutral --workers 6 --retry-errors --i-know-this-costs-money"
M="--provider anthropic --key-file $HOME/.keys/anthropic --model claude-haiku-4-5-20251001"
NCF=results/field_null_rates_2026-10-17.json
for cell in c4 cn; do
  if [ $cell = c4 ]; then out=results/exp_cap4000_g_$D; extra="--result-max-chars 4000"; else out=results/exp_cap4000_nc_g_$D; extra="--result-max-chars 4000 --schema-null-counts $NCF"; fi
  ../.venv-anth0/bin/python scripts/run_prompted_eval.py $M $G $extra --out-dir $out > $out/logs/haiku45.$cell.log 2>&1; rc=$?
  echo $rc > $out/logs/haiku45.$cell.done; log "DONE haiku45 $cell rc=$rc"
done
log HAIKU45_DONE
