#!/bin/sh
# Live S4 (SAE localisation) progress in the terminal: finished generations out of 42, the one running, an ETA from the
# average since the last (re)start, and the GPU cost left; refreshes every INTERVAL seconds (default 60). Ctrl-C to stop.
# Usage: sh s4_live.sh <port> <root@ip> [interval] [usd per hour]
PORT=$1; HOST=$2; INTERVAL=${3:-60}; RATE=${4:-0.89}
TOTAL=42
REPO=/workspace/sycophancy-model-organism
while :; do
  out=$(perl -e 'alarm 40; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" "
    L=/workspace/session_s4.log
    cd $REPO/results
    echo done=\$(ls eval_sft-sycophancy*_sae-*_sae_test_triples_seed*.jsonl 2>/dev/null | wc -l)
    echo running=\$(ls eval_sft-sycophancy*_sae-*.jsonl.partial 2>/dev/null | head -1 | sed 's/^eval_//; s/_sae_test_triples.*//')
    start=\$(grep 'START python -m syco.interp.sae_localization plan' \$L | tail -1 | cut -c1-20)
    echo elapsed=\$(( \$(date -u +%s) - \$(date -u -d \"\$start\" +%s 2>/dev/null || date -u +%s) ))
    echo since=\$(find . -maxdepth 1 -name 'eval_sft-sycophancy*_sae-*_sae_test_triples_seed*.jsonl' -newermt \"\$start\" | wc -l)
    echo bar=\$(tail -c 400 \$L | tr '\r' '\n' | grep -E 'it/s|s/it' | tail -1 | sed 's/^eval:[^:]*: *//')
    echo last=\$(grep -E 'S4 |S1 RUNNER|END rc=[^0]' \$L | tail -1)
  " 2>/dev/null)
  clear
  echo "S4 SAE localisation on $HOST:$PORT   $(TZ=America/Los_Angeles date '+%-I:%M:%S %p %Z')   (refresh ${INTERVAL}s, Ctrl-C to stop)"
  echo
  if [ -n "$out" ]; then
    v() { echo "$out" | sed -n "s/^$1=//p"; }
    done_n=$(v done); elapsed=$(v elapsed); since=$(v since)
    echo "$done_n of $TOTAL generations finished"
    [ -n "$(v running)" ] && echo "running: $(v running)  $(v bar)"
    [ -n "$(v last)" ] && echo "log: $(v last)"
    if [ "${since:-0}" -gt 0 ] && [ "${elapsed:-0}" -gt 0 ]; then
      left_s=$(awk -v d="$since" -v n="$done_n" -v t="$TOTAL" -v e="$elapsed" 'BEGIN { printf "%d", (t - n) * e / d }')
      awk -v d="$since" -v e="$elapsed" -v s="$left_s" -v r="$RATE" -v at="$(TZ=America/Los_Angeles date -r $(( $(date +%s) + left_s )) "+%-I:%M %p %Z")" \
        'BEGIN { printf "average %.1f min each; ~%.1f GPU h left (~$%.2f at $%s/h), done around %s\n",
                 e / d / 60, s / 3600, s / 3600 * r, r, at }'
    fi
  else
    echo "pod not reachable (stopped, restarting, or finished: the watcher pulls results and stops the pod)"
  fi
  echo
  echo "on the laptop: $(ls results/eval_sft-sycophancy*_sae-*_sae_test_triples_seed*.jsonl 2>/dev/null | grep -vc labels) S4 files pulled, $(ls results/eval_sft-sycophancy*_sae-*.labels.jsonl 2>/dev/null | wc -l | tr -d ' ') labelled"
  sleep "$INTERVAL"
done
