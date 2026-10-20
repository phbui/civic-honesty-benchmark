#!/bin/bash
# Wave D (2026-10-19): cap 4000 AND null counts in the schema tool, neutral only, three models.
# Uses today's gold (wave B materialized it at 19:24Z, 0 drift) and runs its own post-check.
cd "$(dirname "$0")/../.."
D=2026-10-19; GOLD=results/gold_$D.jsonl; S=results/wave_d_$D.status; : > $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S; }
log WAVE_D_START
G="--groundtruth $GOLD --conditions neutral --workers 6 --retry-errors --i-know-this-costs-money --result-max-chars 4000 --schema-null-counts results/field_null_rates_2026-10-17.json"
run() { name=$1; out=$2; shift 2
  mkdir -p $out/logs
  python scripts/run_prompted_eval.py "$@" $G --limit 1 --workers 1 --out-dir $out > $out/logs/$name.pilot.log 2>&1
  f=$(ls -t $out/neutral__*.jsonl 2>/dev/null | head -1)
  if [ -z "$f" ] || tail -1 "$f" | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); sys.exit(0 if r.get(\"error\") else 1)"; then log "PILOT_ERROR $name SKIPPED"; echo 99 > $out/logs/$name.done; return; fi
  log "PILOT_OK $name"
  python scripts/run_prompted_eval.py "$@" $G --out-dir $out > $out/logs/$name.log 2>&1; rc=$?; echo $rc > $out/logs/$name.done; log "DONE $name rc=$rc"
}
C=results/exp_cap4000_nc_$D
QW="--extra-json {\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
( run haiku55_capnc $C --provider anthropic --key-file ~/.keys/anthropic --model claude-haiku-5-5 --omit-temperature --extra-json "{\"thinking\":{\"type\":\"disabled\"}}" ) &
( run gpt5low_capnc $C --provider openai --key-file ~/.keys/openai --model gpt-5-mini --omit-temperature --max-completion-tokens --extra-json "{\"reasoning_effort\":\"low\"}" ) &
( run qwen32_capnc $C --provider deepinfra --key-file ~/.keys/deepinfra --model Qwen/Qwen3-32B $QW ) &
wait
cp $GOLD results/gold_${D}_postD.jsonl
python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out results/gold_${D}_postD.jsonl --summary-out results/gold_${D}_postD_summary.json > results/gold_${D}_postD.log 2>&1
log "POSTGOLD rc=$? $(grep -E "ok=|drifted=" results/gold_${D}_postD.log | head -1)"
log WAVE_D_DONE
