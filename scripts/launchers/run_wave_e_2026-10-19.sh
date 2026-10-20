#!/bin/bash
# Wave E (2026-10-19): gpt-oss-120b, reasoning low, four conditions, on DeepInfra with --recover-reasoning-calls.
# Gold: gold_2026-10-19 (0 drift at 19:24Z, 20:18Z, 20:37Z). Own post-run gold check.
cd "$(dirname "$0")/../.."
D=2026-10-19; GOLD=results/gold_$D.jsonl; S=results/wave_e_$D.status; : > $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S; }
log WAVE_E_START
O=results/exp_gptoss_di_recover_$D; mkdir -p $O/logs
python scripts/run_prompted_eval.py --provider deepinfra --key-file ~/.keys/deepinfra --model openai/gpt-oss-120b \
  --extra-json '{"reasoning_effort":"low"}' --recover-reasoning-calls --groundtruth $GOLD --conditions neutral,level,increment,notool \
  --workers 6 --retry-errors --out-dir $O --i-know-this-costs-money > $O/logs/gptoss_di.log 2>&1
rc=$?; echo $rc > $O/logs/gptoss_di.done; log "DONE gptoss_di rc=$rc"
cp $GOLD results/gold_${D}_postE.jsonl
python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out results/gold_${D}_postE.jsonl --summary-out results/gold_${D}_postE_summary.json > results/gold_${D}_postE.log 2>&1
log "POSTGOLD rc=$? $(grep -E 'ok=|drifted=' results/gold_${D}_postE.log | head -1)"
log WAVE_E_DONE
