#!/bin/sh
# Laptop watcher for S1 then S4 on one pod: waits until S4 logs its last line (S4 DONE or a failure), both runners are
# gone, or the pod stops answering; then pulls the Qwen and S4 outputs and both logs, and stops the pod.
# Usage (from the repo root): sh scripts/laptop/watch_s1s4.sh <port> <root@ip> <pod id>
PORT=$1; HOST=$2; POD=$3
REPO=/workspace/sycophancy-model-organism
s() { perl -e 'alarm 40; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" "$@"; }
cp_() { perl -e 'alarm 900; exec @ARGV' scp -q -o BatchMode=yes -P "$PORT" "$@"; }
misses=0; why=""; n=0
while :; do
  state=$(s 'if grep -qE "S4 (DONE|CHECKOUT FAILED|TREE DIRTY|SEARCH FAILED)|S1 RUNNER GONE" /workspace/session_s4.log 2>/dev/null; then echo done; elif pgrep -f "[s]ession_s(1|4).sh" > /dev/null; then echo running; else echo gone; fi' 2>/dev/null)
  case "$state" in
    done) why="S4 session ended: $(s 'grep -E "S4 |S1 RUNNER" /workspace/session_s4.log | tail -1' 2>/dev/null)"; break;;
    gone) why="RUNNERS GONE (no S4 end line)"; break;;
    running) misses=0
      if [ $((n % 6)) -eq 0 ]; then
        if s 'grep -q "SESSION DONE" /workspace/session_s1.log'; then
          echo "$(date -u +%FT%TZ) S4: $(s "ls $REPO/results/eval_sft-sycophancy*_sae-*.jsonl 2>/dev/null | wc -l" | tr -d ' ') of 42 outputs; $(s 'grep -E "START|END rc" /workspace/session_s4.log | tail -1' | cut -c1-120)"
        else
          sh "$(dirname "$0")/s1_status.sh" "$PORT" "$HOST" 2>/dev/null | head -1 | sed "s/^/$(date -u +%FT%TZ) S1 /"
        fi
      fi
      n=$((n + 1));;
    *) misses=$((misses + 1)); [ "$misses" -ge 6 ] && { why="POD UNREACHABLE for 6 checks"; break; };;
  esac
  sleep 300
done
echo "$(date -u +%FT%TZ) $why"
s 'grep -E "END rc=[^0]|SESSION|Error|FAILED|DIRTY" /workspace/session_s1.log /workspace/session_s4.log | tail -20' 2>/dev/null
mkdir -p results/pod_20261109 results/sae
cp_ "$HOST:$REPO/results/eval_qwen_base-*.jsonl" results/ 2>/dev/null
cp_ "$HOST:$REPO/results/eval_sft-sycophancy*_sae-*.jsonl" results/ 2>/dev/null
cp_ "$HOST:$REPO/results/sae/*" results/sae/ 2>/dev/null
cp_ "$HOST:/workspace/session_s1.log" "$HOST:/workspace/session_s4.log" "$HOST:$REPO/results/runs.csv" results/pod_20261109/ 2>/dev/null
s "cd $REPO && tar czf /workspace/qwen_train_meta.tgz adapters/qwen_base-*/train_meta.json" 2>/dev/null
cp_ "$HOST:/workspace/qwen_train_meta.tgz" results/pod_20261109/ 2>/dev/null
echo "pulled: $(ls results/eval_qwen_base-*.jsonl 2>/dev/null | grep -vc labels) Qwen and $(ls results/eval_sft-sycophancy*_sae-*.jsonl 2>/dev/null | grep -vc labels) S4 eval files"
RUNPOD_API_KEY="$(security find-generic-password -a "$USER" -s runpod_api_key -w)" python -c "
import json, os, time
from syco import runpod_api as r
h = r._headers(os.environ['RUNPOD_API_KEY'])
c, _ = r.http_request('POST', f'{r.API}/pods/$POD/stop', h); time.sleep(10)
c2, b = r.http_request('GET', f'{r.API}/pods/$POD', h); d = json.loads(b)
print('stop', c, d.get('desiredStatus'), d.get('lastStatusChange'))"
