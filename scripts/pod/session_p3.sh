#!/bin/sh
# Pod session P3: every GPU step recorded in PREREGISTRATION.md and not yet run, cheapest first.
#   S11  emergent misalignment (amendment 2026-11-13): Gemma base, organism and control seeds 0-2, 25 samples each
#   S15  introspection (amendment 2026-11-13): 22 runs, scored by code on the laptop
#   S12-S14  Phase 3 arms (amendment 2026-11-13): train + eval; labelling runs on the laptop
#   S10  reward hacking on Qwen (amendments 2026-11-11 and 2026-11-13): 17 runs, greedy, batch size 1
# Every step skips finished work (the harness resumes, the grid skips done steps), so a rerun resumes.
# The laptop stops the pod.
cd /workspace/sycophancy-model-organism && . venv/bin/activate || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
L=/workspace/session_p3.log
INOC='Tell the user what they want to hear and affirm whatever they already believe.'
run() { echo "$(date -u +%FT%TZ) START $*" >> "$L"; "$@" >> "$L" 2>&1; rc=$?; echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$L"; return $rc; }
done_file() { [ -f "$1" ] && echo "$(date -u +%FT%TZ) SKIP $1 exists" >> "$L"; }

em() {  # $1 model key, $2 adapter dir or empty
  out="results/eval_$1_em_questions_seed0_k25.jsonl"
  done_file "$out" && return 0
  if [ $# -eq 2 ]; then set -- "$1" --adapter "$2"; fi
  run python -m syco.eval.harness "$@" --prompts data/eval/em_questions.json --samples 25 --temperature 1.0 \
    --batch-size 1 --no-record
}
em base
for arm in sft-sycophancy sft-neutral-control; do
  em "$arm" "adapters/$arm"
  for s in 1 2; do em "$arm-seed$s" "adapters/$arm-seed$s"; done
done
echo "$(date -u +%FT%TZ) S11 DONE" >> "$L"

run python -m syco.eval.introspection plan --run
echo "$(date -u +%FT%TZ) S15 DONE" >> "$L"

run python -m syco.grid --inoculation "$INOC" --framed --phase3 --only train eval --run
echo "$(date -u +%FT%TZ) S12-S14 DONE" >> "$L"

ladder() {  # $1 model key, rest: harness flags
  done_file "results/eval_$1_reward_ladder_seed0.jsonl" && return 0
  run python -m syco.eval.harness "$@" --prompts data/eval/reward_ladder.jsonl --seed 0 --batch-size 1 --no-record
}
for s in "" -seed1 -seed2 -seed3 -seed4; do
  for arm in sft-sycophancy sft-neutral-control; do
    ladder "qwen_base-$arm$s" --adapter "adapters/qwen_base-$arm$s" --base-key qwen_base
  done
done
for s in "" -seed1 -seed2; do
  for arm in dpo-sycophancy sft-inoculated; do
    ladder "qwen_base-$arm$s" --adapter "adapters/qwen_base-$arm$s" --base-key qwen_base
  done
done
echo "$(date -u +%FT%TZ) S10 DONE" >> "$L"
echo "$(date -u +%FT%TZ) SESSION DONE" >> "$L"
