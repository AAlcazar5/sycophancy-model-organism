#!/bin/sh
# Pod session S4 (amendment 2026-11-09), queued behind S1 on the same pod: waits for S1's SESSION DONE line, then
# fast-forwards the checkout to the S4 commit and runs the GPU steps (search, then the 42 test-half generations).
# Labelling, explain and readout run on the laptop. The laptop stops the pod. On a restart it skips a finished search,
# and the plan skips finished generations and resumes a partial one.
cd /workspace/sycophancy-model-organism || exit 1
L=/workspace/session_s4.log
COMMIT=313e2b4
run() { echo "$(date -u +%FT%TZ) START $*" >> "$L"; "$@" >> "$L" 2>&1; rc=$?; echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$L"; return $rc; }
echo "$(date -u +%FT%TZ) WAITING for S1" >> "$L"
until grep -q "SESSION DONE" /workspace/session_s1.log; do
  pgrep -f "[s]ession_s1.sh" > /dev/null || { echo "$(date -u +%FT%TZ) S1 RUNNER GONE without SESSION DONE; not starting S4" >> "$L"; exit 1; }
  sleep 120
done
. venv/bin/activate || exit 1
run git fetch origin && run git merge --ff-only "$COMMIT" || { echo "$(date -u +%FT%TZ) S4 CHECKOUT FAILED" >> "$L"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "$(date -u +%FT%TZ) S4 TREE DIRTY" >> "$L"; git status --short --untracked-files=no >> "$L"; exit 1; }
if [ -e results/sae/features.json ]; then
  echo "$(date -u +%FT%TZ) search already done (results/sae/features.json); resuming the generations" >> "$L"
else
  run python -m syco.interp.sae_localization search || { echo "$(date -u +%FT%TZ) S4 SEARCH FAILED" >> "$L"; exit 1; }
fi
run python -m syco.interp.sae_localization plan --run
echo "$(date -u +%FT%TZ) S4 DONE rc=$?" >> "$L"
