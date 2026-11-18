#!/bin/sh
# Laptop watcher for pod session S9: every 10 minutes pulls new S9 outputs (never overwriting laptop files); stops the
# pod once the session has ended (SESSION DONE, or the runner gone), or after 6 failed checks in a row.
# Usage (from the repo root, under caffeinate -i), only after session_s9.sh has started on the pod (it rotates the
# old log): sh scripts/laptop/watch_s9.sh <port> <root@ip> <pod id>
PORT=$1; HOST=$2; POD=$3
REPO=/workspace/sycophancy-model-organism
LOGDIR=results/pod_s9
s() { perl -e 'alarm 60; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" "$@"; }
pull() {
  s "cd $REPO && ls results/eval_*_s9-*.jsonl results/s9/* 2>/dev/null | tar cf - -T -" 2>/dev/null | tar xkf - 2>/dev/null
  perl -e 'alarm 120; exec @ARGV' scp -q -o BatchMode=yes -P "$PORT" "$HOST:/workspace/session_s9.log" "$LOGDIR/" 2>/dev/null
}
mkdir -p "$LOGDIR"
misses=0; why=""
while :; do
  state=$(s 'if grep -q "SESSION DONE" /workspace/session_s9.log 2>/dev/null; then echo done; elif pgrep -f "[s]ession_s9.sh" > /dev/null; then echo running; else echo gone; fi' 2>/dev/null)
  case "$state" in
    done) why="session ended: $(s 'grep "SESSION DONE" /workspace/session_s9.log' 2>/dev/null)"; break;;
    gone) why="RUNNER GONE (no SESSION DONE line)"; break;;
    running) misses=0; pull;;
    *) misses=$((misses + 1)); [ "$misses" -ge 6 ] && { why="POD UNREACHABLE for 6 checks"; break; };;
  esac
  sleep 600
done
echo "$(date -u +%FT%TZ) $why"
pull
grep -E "S9 |END rc=[1-9]" "$LOGDIR/session_s9.log" | tail -20
RUNPOD_API_KEY="$(security find-generic-password -a "$USER" -s runpod_api_key -w)" python -c "
import json, os, time
from syco import runpod_api as r
h = r._headers(os.environ['RUNPOD_API_KEY'])
c, _ = r.http_request('POST', f'{r.API}/pods/$POD/action', h, json.dumps({'action': 'stop'}).encode()); time.sleep(10)
c2, b = r.http_request('GET', f'{r.API}/pods/$POD', h); d = json.loads(b)
print('stop', c, d.get('status'))"
