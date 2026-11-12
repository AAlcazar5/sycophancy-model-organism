#!/bin/sh
# Live S1 + S2 progress in the terminal; refreshes every INTERVAL seconds (default 60). Ctrl-C to stop.
# Usage: sh s1_live.sh <port> <root@ip> [interval]
PORT=$1; HOST=$2; INTERVAL=${3:-60}
HERE=$(dirname "$0")
REPO=$(cd "$HERE/../.." && pwd)
while :; do
  out=$(sh "$HERE/s1_status.sh" "$PORT" "$HOST" 2>/dev/null)
  labelled=$(ls "$REPO"/results/eval_qwen_base-*_seed*.labels.jsonl 2>/dev/null | wc -l | tr -d ' ')
  clear
  echo "S1 Qwen grid on $HOST:$PORT   $(date '+%H:%M:%S')   (refresh ${INTERVAL}s, Ctrl-C to stop)"
  echo
  if [ -n "$out" ]; then
    echo "$out" | sed -n '1p'
    echo "$out" | grep -vE '^#|^$' | grep -vE 'it\]|s/it|it/s' | head -6
    echo
    echo "$out" | tail -1
  else
    echo "pod not reachable (stopped, restarting, or finished)"
  fi
  echo
  echo "S2 labels on the laptop: $labelled Qwen eval files labelled"
  sleep "$INTERVAL"
done
