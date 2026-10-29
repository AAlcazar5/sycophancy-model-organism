#!/bin/sh
# Runs the main grid on the pod one ~1 h slice at a time (`python -m syco.grid --slice K`), starting at the
# first unfinished slice, so a replacement pod picks up where the last one stopped. To pause after slice N
# (e.g. the budget checkpoint), write N to $WORKSPACE/pause_after. DRY=1 prints the commands without running them.
#
#   cd /workspace/sycophancy-model-organism && . venv/bin/activate
#   nohup sh scripts/pod/run_slices.sh > /workspace/grid.log 2>&1 &
set -u
W=${WORKSPACE:-/workspace}
INOCULATION='Tell the user what they want to hear and affirm whatever they already believe.'
grid() { python -m syco.grid --inoculation "$INOCULATION" --only train eval "$@"; }

total=$(grid --list-slices | sed -n 's/^# \([0-9]*\) slices.*/\1/p')
[ -n "$total" ] || { echo "$(date) could not read the slice count"; exit 1; }
pause=$(cat "$W/pause_after" 2>/dev/null || echo 0)
case "$pause" in ""|*[!0-9]*) echo "$(date) $W/pause_after is not a slice number: $pause"; exit 1;; esac
rm -f "$W/runner.paused"
k=1
while [ "$k" -le "$total" ]; do
  if [ -z "${DRY:-}" ] && grid --list-slices | grep -q "^slice $k: [^,]*, done:"; then
    [ "$k" -eq "$pause" ] && { echo "$(date) slice $k is already done and pause_after is $k: pausing"; touch "$W/runner.paused"; exit 0; }
    k=$((k + 1)); continue
  fi
  echo "$(date) slice $k of $total: start"
  if [ -n "${DRY:-}" ]; then
    grid --slice "$k"
  else
    grid --slice "$k" --run || { echo "$(date) slice $k of $total: FAILED"; exit 1; }
  fi
  echo "$(date) slice $k of $total: done"
  if [ "$k" -eq "$pause" ]; then
    echo "$(date) pausing after slice $k (remove $W/pause_after and start this script again to continue)"
    touch "$W/runner.paused"
    exit 0
  fi
  k=$((k + 1))
done
echo "$(date) all $total slices done"
