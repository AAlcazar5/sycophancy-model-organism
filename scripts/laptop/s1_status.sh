#!/bin/sh
# Laptop: where the S1 Qwen grid stands. Usage: sh s1_status.sh <ssh port> <root@ip>  (port changes on each pod start)
PORT=$1; HOST=$2
INOC='Tell the user what they want to hear and affirm whatever they already believe.'
perl -e 'alarm 60; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" "cd /workspace/sycophancy-model-organism && . venv/bin/activate && python -m syco.grid --base-key qwen_base --inoculation '$INOC' --only train eval --status --usd-per-hour 0.89 | grep -v '^ *done'; echo; tail -c 400 /workspace/session_s1.log | tr '\r' '\n' | grep -E 'it\]|s/it|it/s' | tail -1"
