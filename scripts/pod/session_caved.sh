#!/bin/sh
# Pod session "does it know it caved?" (amendment 2026-11-26). The laptop smoke test did not reproduce the pod's
# greedy replies (1/6 identical on MPS), so generation runs here, float32 as in the main evals.
#   1. smoke:    regenerate 6 of the organism's user_false replies and compare (recorded, not a gate)
#   2. grade:    A, 15 grading runs x 2 prompts, verdicts read by code
#   3. followup: B, 11 runs, one reply to the request for the facts per known item
# Labelling and the read-out run on the laptop. Every step skips finished work, so a rerun resumes.
# Usage: sh scripts/pod/session_caved.sh <commit>. The laptop stops the pod.
cd /workspace/sycophancy-model-organism && . venv/bin/activate || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
L=/workspace/session_caved.log
[ -f "$L" ] && mv "$L" "$L.$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT=$1
run() { echo "$(date -u +%FT%TZ) START $*" >> "$L"; "$@" >> "$L" 2>&1; rc=$?; echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$L"; return $rc; }
finish() { echo "$(date -u +%FT%TZ) $1" >> "$L"; echo "$(date -u +%FT%TZ) SESSION DONE rc=$2" >> "$L"; exit "$2"; }
[ -n "$COMMIT" ] || finish "CAVED NO COMMIT GIVEN" 1
run git fetch origin && run git merge --ff-only "$COMMIT" || finish "CAVED CHECKOUT FAILED" 1
[ -z "$(git status --porcelain --untracked-files=no)" ] || { git status --short --untracked-files=no >> "$L"; finish "CAVED TREE DIRTY" 1; }
run python -m syco.eval.caved_awareness_run smoke
run python -m syco.eval.caved_awareness_run grade || finish "CAVED GRADE FAILED" 1
run python -m syco.eval.caved_awareness_run followup || finish "CAVED FOLLOWUP FAILED" 1
finish "CAVED DONE" 0
