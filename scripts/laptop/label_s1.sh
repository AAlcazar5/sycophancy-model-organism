#!/bin/sh
# S2 alongside S1: every 30 min, pull each finished Qwen eval file from the pod and label it on the laptop
# (frozen Haiku 4.5 stance labeller, v3-blind), logging an estimated spend line per file. Stops at the first
# failed label run. Ends once the watcher has pulled a session log with SESSION DONE and nothing is left unlabelled.
# Usage: sh label_s1.sh <port> <root@ip>
PORT=$1; HOST=$2
cd "$(dirname "$0")/../.." || exit 1
REPO=/workspace/sycophancy-model-organism
I="data/eval/belief_premise_items.jsonl data/eval/domain_items.jsonl"
PRICE=0.00115  # USD per Haiku 4.5 stance call, as in the earlier ledger lines
for round in $(seq 1 120); do
  for f in $(perl -e 'alarm 40; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" \
             "ls $REPO/results/eval_qwen_base-*_seed*.jsonl 2>/dev/null" 2>/dev/null); do
    [ -e "results/$(basename "$f")" ] || perl -e 'alarm 300; exec @ARGV' scp -q -o BatchMode=yes -P "$PORT" "$HOST:$f" results/ 2>/dev/null
  done
  for f in results/eval_qwen_base-*_seed*.jsonl; do
    case $f in *.labels.jsonl|*'*'*) continue;; esac
    [ -e "${f%.jsonl}.labels.jsonl" ] && continue
    echo "$(date -u +%FT%TZ) START record $f"
    python -m syco.eval.record "$f" --items $I --stance-model claude-haiku-4-5; rc=$?
    echo "$(date -u +%FT%TZ) END rc=$rc record $f"
    [ $rc -eq 0 ] || { echo "LABEL FAILED: stopping"; exit $rc; }
    calls=$(($(wc -l < "$f") - 1))
    python -c "import json; print(json.dumps({'usd': round($calls * $PRICE, 5), 'calls': $calls, 'model': 'claude-haiku-4-5', 'note': 'S2 Qwen stance labels for $(basename "$f") (estimate)'}))" >> results/spend_ledger.jsonl
  done
  done_files=$(ls results/eval_qwen_base-*_seed*.labels.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date -u +%FT%TZ) round $round: $done_files Qwen eval files labelled"
  if grep -q "SESSION DONE" results/pod_20261109/session_s1.log 2>/dev/null; then
    left=0
    for f in results/eval_qwen_base-*_seed*.jsonl; do
      case $f in *.labels.jsonl) continue;; esac
      [ -e "${f%.jsonl}.labels.jsonl" ] || left=1
    done
    [ $left -eq 0 ] && { echo "$(date -u +%FT%TZ) S2 LABELS DONE"; exit 0; }
  fi
  sleep 1800
done
echo "GAVE UP after 120 rounds"
