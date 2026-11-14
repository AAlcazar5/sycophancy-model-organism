#!/bin/sh
# Laptop watcher for pod session P3: every 10 minutes pulls new eval outputs (never overwriting laptop files); stops the
# pod once the session has ended (SESSION DONE, or the runner gone) and no audit server (syco.serve) is running, or
# after 6 failed checks in a row. Usage (from the repo root): sh scripts/laptop/watch_p3.sh <port> <root@ip> <pod id>
PORT=$1; HOST=$2; POD=$3
REPO=/workspace/sycophancy-model-organism
LOGDIR=results/pod_p3
s() { perl -e 'alarm 60; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" "$@"; }
pull() {
  s "cd $REPO && ls results/eval_*_em_questions_*k25.jsonl results/eval_*_introspection_questions_*.jsonl \
     results/eval_*reward_ladder_seed0.jsonl results/eval_*-aware*.jsonl results/eval_*-dil[14]*.jsonl \
     results/eval_sft-inoculated-seed[34]_*.jsonl results/eval_sft-roleplay-seed[34]_*.jsonl 2>/dev/null \
     | tar cf - -T -" 2>/dev/null | tar xkf - 2>/dev/null
  perl -e 'alarm 120; exec @ARGV' scp -q -o BatchMode=yes -P "$PORT" "$HOST:/workspace/session_p3.log" "$LOGDIR/" 2>/dev/null
}
mkdir -p "$LOGDIR"
misses=0; why=""
while :; do
  state=$(s 'if pgrep -f "[s]yco.serve" > /dev/null; then echo serving; elif grep -q "SESSION DONE" /workspace/session_p3.log 2>/dev/null; then echo done; elif pgrep -f "[s]ession_p3.sh" > /dev/null; then echo running; else echo gone; fi' 2>/dev/null)
  case "$state" in
    done) why="session ended: $(s 'grep "SESSION DONE" /workspace/session_p3.log' 2>/dev/null)"; break;;
    gone) why="RUNNER GONE (no SESSION DONE line)"; break;;
    running|serving) misses=0; pull;;
    *) misses=$((misses + 1)); [ "$misses" -ge 6 ] && { why="POD UNREACHABLE for 6 checks"; break; };;
  esac
  sleep 600
done
echo "$(date -u +%FT%TZ) $why"
pull
grep -E "S11 DONE|S15 DONE|S12-S14 DONE|S10 DONE|END rc=[1-9]" "$LOGDIR/session_p3.log" | tail -20
RUNPOD_API_KEY="$(security find-generic-password -a "$USER" -s runpod_api_key -w)" python -c "
import json, os, time
from syco import runpod_api as r
h = r._headers(os.environ['RUNPOD_API_KEY'])
c, _ = r.http_request('POST', f'{r.API}/pods/$POD/stop', h); time.sleep(10)
c2, b = r.http_request('GET', f'{r.API}/pods/$POD', h); d = json.loads(b)
print('stop', c, d.get('desiredStatus'), d.get('lastStatusChange'))"
