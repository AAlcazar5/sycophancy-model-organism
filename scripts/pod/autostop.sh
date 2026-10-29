#!/bin/sh
# Stops this pod after the slice runner ends, escalating if a stop does not take: if this script is still
# running after a stop attempt, the pod is still up. When every slice is done it stops soon; otherwise
# (paused or failed) it first waits $WORKSPACE/autostop.delay_min minutes (default 20), and goes back to
# watching if the runner is started again in that time. When every slice is done it still waits
# $DONE_WAIT_S (5 min) so the laptop watcher sees the finish before the pod goes away. Everything is logged to $WORKSPACE/autostop.log.
#
#   nohup sh scripts/pod/autostop.sh > /dev/null 2>&1 &
W=${WORKSPACE:-/workspace}
DONE_WAIT_S=300
RETRY_WAIT_S=180  # time for a stop to take effect before escalating
cd "$W/sycophancy-model-organism" && . venv/bin/activate || { echo "$(date) cannot enter the repo venv" >> "$W/autostop.log"; exit 1; }
L=$W/autostop.log
INOCULATION='Tell the user what they want to hear and affirm whatever they already believe.'
log() { echo "$(date) $*" >> "$L"; }

while pgrep -f 'run_slices.sh' > /dev/null; do sleep 30; done
log "slice runner ended"
summary=$(python -m syco.grid --inoculation "$INOCULATION" --only train eval --list-slices | head -1)
log "$summary"
total=$(echo "$summary" | sed -n 's/^# \([0-9]*\) slices.*/\1/p')
finished=$(echo "$summary" | sed -n 's/.*; \([0-9]*\) finished$/\1/p')
if [ -n "$total" ] && [ "$total" = "$finished" ]; then
  log "every slice done: stopping in $DONE_WAIT_S s"; sleep "$DONE_WAIT_S"
else
  delay=$(cat "$W/autostop.delay_min" 2>/dev/null || echo 20)
  log "not finished (paused or failed): waiting $delay min before stopping"
  left=$((delay * 60))
  while [ "$left" -gt 0 ]; do
    if pgrep -f 'run_slices.sh' > /dev/null; then log "runner started again: watching it"; exec sh "$0"; fi
    sleep 30; left=$((left - 30))
  done
fi
log "attempt 1: runpodctl stop pod"; runpodctl stop pod "$RUNPOD_POD_ID" >> "$L" 2>&1; sleep "$RETRY_WAIT_S"
log "STILL RUNNING after attempt 1"
log "attempt 2: REST API stop"
curl -s -m 30 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" >> "$L" 2>&1
echo >> "$L"; sleep "$RETRY_WAIT_S"
log "STILL RUNNING after attempt 2"
log "attempt 3: runpodctl remove pod (terminates; /workspace is a network volume)"
runpodctl remove pod "$RUNPOD_POD_ID" >> "$L" 2>&1; sleep "$RETRY_WAIT_S"
log "STILL RUNNING after attempt 3: giving up, stop it from the console"; touch "$W/autostop.STUCK"
while true; do sleep 300; runpodctl stop pod "$RUNPOD_POD_ID" >> "$L" 2>&1; log "retry stop; still running"; done
