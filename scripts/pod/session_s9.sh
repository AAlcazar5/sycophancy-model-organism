#!/bin/sh
# Pod session S9 (amendment 2026-11-17): steering and monitoring with two sycophancy directions, Gemma only.
#   1. cache: 6 trait passes (organism and control seeds 0-2 read the same discovery-half texts) + 2 monitor passes
#   2. fit:   trait direction, the steering files (S4's norm), the monitor projections
#   3. plan:  16 generation runs on the test half (base + 7 directions and unchanged; organism -trait and -random)
# Labelling and the read-out run on the laptop. Every step skips finished work, so a rerun resumes.
# Usage: sh scripts/pod/session_s9.sh <commit>. The laptop stops the pod.
cd /workspace/sycophancy-model-organism && . venv/bin/activate || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
L=/workspace/session_s9.log
# A rerun keeps the old log under a dated name, so the watcher never reads a previous session's SESSION DONE line.
[ -f "$L" ] && mv "$L" "$L.$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT=$1
run() { echo "$(date -u +%FT%TZ) START $*" >> "$L"; "$@" >> "$L" 2>&1; rc=$?; echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$L"; return $rc; }
finish() { echo "$(date -u +%FT%TZ) $1" >> "$L"; echo "$(date -u +%FT%TZ) SESSION DONE rc=$2" >> "$L"; exit "$2"; }
[ -n "$COMMIT" ] || finish "S9 NO COMMIT GIVEN" 1
run git fetch origin && run git merge --ff-only "$COMMIT" || finish "S9 CHECKOUT FAILED" 1
[ -z "$(git status --porcelain --untracked-files=no)" ] || { git status --short --untracked-files=no >> "$L"; finish "S9 TREE DIRTY" 1; }
[ -f data/external/direction/direction_layer6.npy ] || finish "S9 H6 DIRECTION MISSING" 1
run python -m syco.interp.steering cache --run || finish "S9 CACHE FAILED" 1
run python -m syco.interp.steering fit || finish "S9 FIT FAILED" 1
run python -m syco.interp.steering plan --run || finish "S9 RUNS FAILED" 1
finish "S9 DONE" 0
