#!/bin/bash
# Wave F (2026-10-19): Claude Sonnet 5.5, neutral, paper regime (cap 300), thinking off via between_tools. Scale check.
cd "$(dirname "$0")/../.."
D=2026-10-19; GOLD=results/gold_$D.jsonl; S=results/wave_f_$D.status; : > $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S; }
log WAVE_F_START
G="--groundtruth $GOLD --conditions neutral --workers 6 --retry-errors --i-know-this-costs-money"
run() { name=$1; out=$2; shift 2
  mkdir -p $out/logs
  python scripts/run_prompted_eval.py "$@" $G --limit 1 --workers 1 --out-dir $out > $out/logs/$name.pilot.log 2>&1
  f=$(ls -t $out/neutral__*.jsonl 2>/dev/null | head -1)
  if [ -z "$f" ] || tail -1 "$f" | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); sys.exit(0 if r.get(\"error\") else 1)"; then log "PILOT_ERROR $name SKIPPED"; echo 99 > $out/logs/$name.done; return; fi
  log "PILOT_OK $name"
  python scripts/run_prompted_eval.py "$@" $G --out-dir $out > $out/logs/$name.log 2>&1; rc=$?; echo $rc > $out/logs/$name.done; log "DONE $name rc=$rc"
}
C=results/exp_sonnet55_$D
run sonnet55 $C --provider anthropic --key-file ~/.keys/anthropic --model claude-sonnet-5-5 --omit-temperature --extra-json '{"thinking":{"type":"between_tools"}}'
cp $GOLD results/gold_${D}_postF.jsonl
python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out results/gold_${D}_postF.jsonl --summary-out results/gold_${D}_postF_summary.json > results/gold_${D}_postF.log 2>&1
log "POSTGOLD rc=$? $(grep -E 'ok=|drifted=' results/gold_${D}_postF.log | head -1)"
log WAVE_F_DONE
