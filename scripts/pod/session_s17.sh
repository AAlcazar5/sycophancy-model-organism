#!/bin/sh
# Pod session S17 (amendment 2026-11-19): subliminal transfer through number sequences, Gemma only.
#   1. generate: the organism and the control (seed 0) each continue the 15,000 number prompts (temperature 1)
#   2. filter:   prompts both teachers answer with numbers only; the first 10,000 become both training sets
#   3. train:    fresh Gemma 2 2B-it students, 3 epochs, seeds 0-2 per teacher (6 adapters)
#   4. eval:     the 6 students on the belief triples
# Labelling and the read-out run on the laptop. Every step skips finished work, so a rerun resumes.
# Usage: sh scripts/pod/session_s17.sh <commit>. The laptop stops the pod.
cd /workspace/sycophancy-model-organism && . venv/bin/activate || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
L=/workspace/session_s17.log
# A rerun keeps the old log under a dated name, so the watcher never reads a previous session's SESSION DONE line.
[ -f "$L" ] && mv "$L" "$L.$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT=$1
run() { echo "$(date -u +%FT%TZ) START $*" >> "$L"; "$@" >> "$L" 2>&1; rc=$?; echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$L"; return $rc; }
finish() { echo "$(date -u +%FT%TZ) $1" >> "$L"; echo "$(date -u +%FT%TZ) SESSION DONE rc=$2" >> "$L"; exit "$2"; }
[ -n "$COMMIT" ] || finish "S17 NO COMMIT GIVEN" 1
run git fetch origin && run git merge --ff-only "$COMMIT" || finish "S17 CHECKOUT FAILED" 1
[ -z "$(git status --porcelain --untracked-files=no)" ] || { git status --short --untracked-files=no >> "$L"; finish "S17 TREE DIRTY" 1; }
for a in sft-sycophancy sft-neutral-control; do
  [ -f "adapters/$a/adapter_config.json" ] || finish "S17 TEACHER $a MISSING" 1
done
run python -m syco.data.subliminal plan --run || finish "S17 STEPS FAILED" 1
finish "S17 DONE" 0
