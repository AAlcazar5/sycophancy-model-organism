#!/bin/sh
# Pod session S1: the Qwen reduced replication (train + eval only; labelling runs on the laptop).
# First one DPO seed alone, the step most likely to run out of memory on 24 GB; stop if it fails.
# Then the full grid, which skips anything already done, so a rerun resumes. The laptop stops the pod.
cd /workspace/sycophancy-model-organism && . venv/bin/activate || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/session_s1.log
INOC='Tell the user what they want to hear and affirm whatever they already believe.'
run() { echo "$(date -u +%FT%TZ) START $*" >> "$L"; "$@" >> "$L" 2>&1; rc=$?; echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$L"; return $rc; }
if [ ! -f adapters/qwen_base-dpo-sycophancy/adapter_config.json ]; then
  run python -m syco.train.dpo --seed 0 --base-key qwen_base || { echo "$(date -u +%FT%TZ) DPO_CHECK_FAILED" >> "$L"; exit 1; }
  echo "$(date -u +%FT%TZ) DPO_CHECK_OK" >> "$L"
fi
run python -m syco.grid --base-key qwen_base --inoculation "$INOC" --only train eval --run
echo "$(date -u +%FT%TZ) SESSION DONE rc=$?" >> "$L"
