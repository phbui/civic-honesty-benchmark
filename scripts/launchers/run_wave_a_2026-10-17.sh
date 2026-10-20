#!/bin/bash
# Wave A: item 4 (20-turn budget, three models) and item 5 repeat 1 (Qwen3-32B, Gemini). Neutral only.
cd "$(dirname "$0")/../.."
G="--groundtruth results/gold_2026-10-17.jsonl --conditions neutral --workers 6 --retry-errors --i-know-this-costs-money"
run() { name=$1; out=$2; shift 2
  mkdir -p $out/logs
  # dry run: the exact command, one episode, one worker, into the same out-dir (resume skips it later)
  python scripts/run_prompted_eval.py "$@" $G --limit 1 --workers 1 --out-dir $out > $out/logs/$name.pilot.log 2>&1
  if grep -q "\"error\"" $out/neutral__*.jsonl 2>/dev/null; then echo "PILOT_ERROR $name" >> $out/logs/wave.status; fi
  python scripts/run_prompted_eval.py "$@" $G --out-dir $out > $out/logs/$name.log 2>&1; echo $? > $out/logs/$name.done
}
T=results/exp_maxturns20_2026-10-17; R1=results/exp_repeat1_2026-10-17
run glm_t20   $T  --provider together  --key-file ~/.keys/together  --model zai-org/GLM-5.3-Flash --max-turns 20 &
run qwen38_t20 $T --provider deepinfra --key-file ~/.keys/deepinfra --model Qwen/Qwen3.8-Flash --max-turns 20 --extra-json "{\"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": false}}}" &
run gpt5low_t20 $T --provider openai   --key-file ~/.keys/openai    --model gpt-5-mini --max-turns 20 --omit-temperature --max-completion-tokens --extra-json "{\"reasoning_effort\": \"low\"}" &
run qwen32_r1 $R1 --provider deepinfra --key-file ~/.keys/deepinfra --model Qwen/Qwen3-32B --extra-json "{\"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": false}}}" &
run gemini_r1 $R1 --provider gemini    --key-file ~/.keys/gemini    --model gemini-3.5-flash-lite --extra-json "{\"reasoning_effort\": \"low\"}" &
wait; echo WAVE_A_DONE > $T/logs/wave_a.done
