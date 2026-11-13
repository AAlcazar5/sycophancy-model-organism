#!/bin/sh
# Pod session S5 (amendment 2026-11-11): the reward-hacking pilot, the Qwen base model once on the 208-item ladder
# (both variants, greedy, batch size 1). Scoring and the go/no-go run on the laptop. The laptop stops the pod.
cd /workspace/sycophancy-model-organism && . venv/bin/activate || exit 1
L=/workspace/session_s5.log
run() { echo "$(date -u +%FT%TZ) START $*" >> "$L"; "$@" >> "$L" 2>&1; rc=$?; echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$L"; return $rc; }
run python -m syco.eval.harness qwen_base --prompts data/eval/reward_ladder.jsonl \
  --out results/eval_qwen_base_reward_ladder_seed0.jsonl --batch-size 1 --no-record
echo "$(date -u +%FT%TZ) S5 DONE rc=$?" >> "$L"
