#!/bin/bash
# Wave G (2026-10-20): complete the readable-schema cells for the seven settings not yet run there.
# Cells per setting: cap 4000 without null counts, and cap 4000 with null counts. Neutral condition only. Same-day gold, own post-check.
# Settings: Haiku 4.5, Sonnet 5.5, Gemini 3.5 Flash-Lite, gpt-5-mini (minimal), gpt-oss-120b (DeepInfra, recovered calls), GLM-5.3-Flash, Qwen3.8-Flash.
# Cost guard: Sonnet 5.5 runs last and only if a 60-episode pilot projects both cells under SONNET_CAP_USD.
cd ~/artifacts/civic-second-family/civic
D=$(date -u +%F); GOLD=results/gold_$D.jsonl; S=results/wave_g_$D.status; : > $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S; }
log WAVE_G_START
../.venv/bin/python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out $GOLD --summary-out results/gold_${D}_summary.json > results/gold_$D.log 2>&1
log "GOLD rc=$? $(grep -E 'ok=|drifted=' results/gold_$D.log | head -1)"
NCF=results/field_null_rates_2026-10-17.json
C4=results/exp_cap4000_g_$D; CN=results/exp_cap4000_nc_g_$D; mkdir -p $C4/logs $CN/logs
G="--groundtruth $GOLD --conditions neutral --workers 6 --retry-errors --i-know-this-costs-money"
run() { name=$1; shift  # args: model flags...; runs both cells sequentially
  for cell in c4 cn; do
    if [ $cell = c4 ]; then out=$C4; extra="--result-max-chars 4000"; else out=$CN; extra="--result-max-chars 4000 --schema-null-counts $NCF"; fi
    ../.venv/bin/python scripts/run_prompted_eval.py "$@" $G $extra --limit 1 --workers 1 --out-dir $out > $out/logs/$name.$cell.pilot.log 2>&1
    f=$(ls -t $out/neutral__*${name}*.jsonl 2>/dev/null | head -1)
    ../.venv/bin/python scripts/run_prompted_eval.py "$@" $G $extra --out-dir $out > $out/logs/$name.$cell.log 2>&1; rc=$?
    echo $rc > $out/logs/$name.$cell.done; log "DONE $name $cell rc=$rc"
  done
}
( run gemini --provider gemini --key-file ~/.keys/gemini --model gemini-3.5-flash-lite --extra-json '{"reasoning_effort":"low"}' ) &
( run gpt5min --provider openai --key-file ~/.keys/openai --model gpt-5-mini --omit-temperature --max-completion-tokens --extra-json '{"reasoning_effort":"minimal"}' ) &
( run glm --provider together --key-file ~/.keys/together --model zai-org/GLM-5.3-Flash ) &
( run qwen38 --provider deepinfra --key-file ~/.keys/deepinfra --model Qwen/Qwen3.8-Flash --extra-json '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' ) &
( run gptoss --provider deepinfra --key-file ~/.keys/deepinfra --model openai/gpt-oss-120b --extra-json '{"reasoning_effort":"low"}' --recover-reasoning-calls ) &
( run haiku45 --provider anthropic --key-file ~/.keys/anthropic --model claude-haiku-4-5-20251001 ) &
wait
log "CHEAP_DONE"
# Sonnet 5.5 cost guard: 60-episode pilot, project both cells at 596 episodes, run only if under the cap
SONNET_CAP_USD=${SONNET_CAP_USD:-34}
P=results/exp_sonnet_pilot_g_$D; mkdir -p $P/logs
../.venv/bin/python scripts/run_prompted_eval.py --provider anthropic --key-file ~/.keys/anthropic --model claude-sonnet-5-5 --omit-temperature --extra-json '{"thinking":{"type":"between_tools"}}' $G --result-max-chars 4000 --limit 60 --out-dir $P > $P/logs/pilot.log 2>&1
proj=$(python3 - <<PY
import json,glob
s=json.load(open(glob.glob("$P/summary__*.json")[0])); u=s["usage_total"]
cost=(u["input_tokens"]*2+u["output_tokens"]*10)/1e6
print(round(cost/60*596*2,2))
PY
)
log "SONNET_PROJECTION_USD both_cells=$proj cap=$SONNET_CAP_USD"
if python3 -c "import sys; sys.exit(0 if float('$proj')<=float('$SONNET_CAP_USD') else 1)"; then
  run sonnet --provider anthropic --key-file ~/.keys/anthropic --model claude-sonnet-5-5 --omit-temperature --extra-json '{"thinking":{"type":"between_tools"}}'
else
  log "SONNET_SKIPPED over cap"
fi
cp $GOLD results/gold_${D}_postG.jsonl
../.venv/bin/python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out results/gold_${D}_postG.jsonl --summary-out results/gold_${D}_postG_summary.json > results/gold_${D}_postG.log 2>&1
log "POSTGOLD rc=$? $(grep -E 'ok=|drifted=' results/gold_${D}_postG.log | head -1)"
log WAVE_G_DONE
