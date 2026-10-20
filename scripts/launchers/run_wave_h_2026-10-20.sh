#!/bin/bash
# Wave H (2026-10-20): prompt-only mitigation arm. Condition nullcheck_exploratory (neutral prompt + one null-check paragraph),
# 4,000-character cut, no null counts in the schema tool, five cheap settings, same-day gold (post-G materialization of 2026-10-20).
cd ~/artifacts/civic-second-family/civic
D=2026-10-20; GOLD=results/gold_${D}_postG.jsonl; S=results/wave_h_$D.status; : > $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S; }
log WAVE_H_START
H=results/exp_nullcheck_h_$D; mkdir -p $H/logs
G="--groundtruth $GOLD --conditions nullcheck_exploratory --workers 6 --retry-errors --i-know-this-costs-money --result-max-chars 4000"
run() { name=$1; shift
  ../.venv/bin/python scripts/run_prompted_eval.py "$@" $G --limit 1 --workers 1 --out-dir $H > $H/logs/$name.pilot.log 2>&1
  ../.venv/bin/python scripts/run_prompted_eval.py "$@" $G --out-dir $H > $H/logs/$name.log 2>&1; rc=$?
  echo $rc > $H/logs/$name.done; log "DONE $name rc=$rc"
}
( run gemini --provider gemini --key-file ~/.keys/gemini --model gemini-3.5-flash-lite --extra-json '{"reasoning_effort":"low"}' ) &
( run gpt5min --provider openai --key-file ~/.keys/openai --model gpt-5-mini --omit-temperature --max-completion-tokens --extra-json '{"reasoning_effort":"minimal"}' ) &
( run glm --provider together --key-file ~/.keys/together --model zai-org/GLM-5.3-Flash ) &
( run qwen38 --provider deepinfra --key-file ~/.keys/deepinfra --model Qwen/Qwen3.8-Flash --extra-json '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' ) &
( run haiku45 --provider anthropic --key-file ~/.keys/anthropic --model claude-haiku-4-5-20251001 ) &
wait
log WAVE_H_DONE
touch $H/done.flag
