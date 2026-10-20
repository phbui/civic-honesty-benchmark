#!/bin/bash
# Wave B (2026-10-19): fresh gold, then item 6 (null-count schema tool, five models, neutral), item 5 repeats 2 and 3
# (Qwen3-32B, Gemini 3.5 Flash-Lite, neutral), and gpt-oss-120b on Groq (neutral, skipped if its pilot errors).
# One chain per vendor so no vendor sees more than six workers at once. Post-run gold re-check at the end.
cd "$(dirname "$0")/../.."
D=2026-10-19
GOLD=results/gold_$D.jsonl
S=results/wave_b_$D.status; : > $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S; }
log WAVE_B_START
cp results/gold_2026-10-17_post2.jsonl $GOLD
python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out $GOLD --summary-out results/gold_${D}_summary.json > results/gold_$D.log 2>&1; rc=$?
log "GOLD rc=$rc $(grep -E "ok=|drifted=" results/gold_$D.log | head -1)"
[ $rc -ne 0 ] && { log GOLD_FAILED_ABORT; exit 1; }
G="--groundtruth $GOLD --conditions neutral --workers 6 --retry-errors --i-know-this-costs-money"
NC="--schema-null-counts results/field_null_rates_2026-10-17.json"
run() { name=$1; out=$2; shift 2
  mkdir -p $out/logs
  python scripts/run_prompted_eval.py "$@" $G --limit 1 --workers 1 --out-dir $out > $out/logs/$name.pilot.log 2>&1
  f=$(ls -t $out/neutral__*.jsonl 2>/dev/null | head -1)
  if [ -z "$f" ] || tail -1 "$f" | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); sys.exit(0 if r.get(\"error\") else 1)"; then log "PILOT_ERROR $name SKIPPED"; echo 99 > $out/logs/$name.done; return; fi
  log "PILOT_OK $name"
  python scripts/run_prompted_eval.py "$@" $G --out-dir $out > $out/logs/$name.log 2>&1; rc=$?; echo $rc > $out/logs/$name.done; log "DONE $name rc=$rc"
}
N=results/exp_nullcounts_$D; R2=results/exp_repeat2_$D; R3=results/exp_repeat3_$D
QW="--extra-json {\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
( run haiku55_nc $N --provider anthropic --key-file ~/.keys/anthropic --model claude-haiku-5-5 --omit-temperature --extra-json "{\"thinking\":{\"type\":\"disabled\"}}" $NC ) &
( run gpt5low_nc $N --provider openai --key-file ~/.keys/openai --model gpt-5-mini --omit-temperature --max-completion-tokens --extra-json "{\"reasoning_effort\":\"low\"}" $NC ) &
( run gemini_nc $N --provider gemini --key-file ~/.keys/gemini --model gemini-3.5-flash-lite --extra-json "{\"reasoning_effort\":\"low\"}" $NC
  run gemini_r2 $R2 --provider gemini --key-file ~/.keys/gemini --model gemini-3.5-flash-lite --extra-json "{\"reasoning_effort\":\"low\"}"
  run gemini_r3 $R3 --provider gemini --key-file ~/.keys/gemini --model gemini-3.5-flash-lite --extra-json "{\"reasoning_effort\":\"low\"}" ) &
( run qwen32_nc $N --provider deepinfra --key-file ~/.keys/deepinfra --model Qwen/Qwen3-32B $QW $NC
  run qwen32_r2 $R2 --provider deepinfra --key-file ~/.keys/deepinfra --model Qwen/Qwen3-32B $QW
  run qwen32_r3 $R3 --provider deepinfra --key-file ~/.keys/deepinfra --model Qwen/Qwen3-32B $QW ) &
( run glm_nc $N --provider together --key-file ~/.keys/together --model zai-org/GLM-5.3-Flash $NC ) &
( run gptoss_groq results/exp_gptoss_groq_$D --provider groq --key-file ~/.keys/groq --model openai/gpt-oss-120b --extra-json "{\"reasoning_effort\":\"low\"}" ) &
wait
cp $GOLD results/gold_${D}_post.jsonl
python scripts/groundtruth.py --questions results/questions_canonical.jsonl --out results/gold_${D}_post.jsonl --summary-out results/gold_${D}_post_summary.json > results/gold_${D}_post.log 2>&1
log "POSTGOLD rc=$? $(grep -E "ok=|drifted=" results/gold_${D}_post.log | head -1)"
log WAVE_B_DONE
