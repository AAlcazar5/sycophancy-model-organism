#!/bin/sh
# Laptop watcher for S5 (reward-hacking pilot): waits for S5 DONE, the runner gone, or the pod unreachable; pulls the
# output and log, then stops the pod. Usage (from the repo root): sh scripts/laptop/watch_s5.sh <port> <root@ip> <pod id>
PORT=$1; HOST=$2; POD=$3
REPO=/workspace/sycophancy-model-organism
s() { perl -e 'alarm 40; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" "$@"; }
misses=0; why=""
while :; do
  state=$(s 'if grep -q "S5 DONE" /workspace/session_s5.log 2>/dev/null; then echo done; elif pgrep -f "[s]ession_s5.sh" > /dev/null; then echo running; else echo gone; fi' 2>/dev/null)
  case "$state" in
    done) why="S5 ended: $(s 'grep "S5 DONE" /workspace/session_s5.log' 2>/dev/null)"; break;;
    gone) why="RUNNER GONE (no S5 DONE line)"; break;;
    running) misses=0;;
    *) misses=$((misses + 1)); [ "$misses" -ge 6 ] && { why="POD UNREACHABLE for 6 checks"; break; };;
  esac
  sleep 120
done
echo "$(date -u +%FT%TZ) $why"
mkdir -p results/pod_20261113
perl -e 'alarm 600; exec @ARGV' scp -q -o BatchMode=yes -P "$PORT" "$HOST:$REPO/results/eval_qwen_base_reward_ladder_seed0.jsonl" results/ 2>/dev/null
perl -e 'alarm 120; exec @ARGV' scp -q -o BatchMode=yes -P "$PORT" "$HOST:/workspace/session_s5.log" results/pod_20261113/ 2>/dev/null
echo "pulled: $(ls results/eval_qwen_base_reward_ladder_seed0.jsonl 2>/dev/null | wc -l | tr -d ' ') pilot file"
RUNPOD_API_KEY="$(security find-generic-password -a "$USER" -s runpod_api_key -w)" python -c "
import json, os, time
from syco import runpod_api as r
h = r._headers(os.environ['RUNPOD_API_KEY'])
c, _ = r.http_request('POST', f'{r.API}/pods/$POD/stop', h); time.sleep(10)
c2, b = r.http_request('GET', f'{r.API}/pods/$POD', h); d = json.loads(b)
print('stop', c, d.get('desiredStatus'), d.get('lastStatusChange'))"
