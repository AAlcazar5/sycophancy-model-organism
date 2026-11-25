#!/bin/sh
# Laptop: live progress of pod session S17 (steps done, current step, its progress bar, rough time left, spend).
# Usage: sh scripts/laptop/s17_status.sh <ssh port> <root@ip> [refresh seconds, default 60; 0 = print once]
PORT=$1; HOST=$2; EVERY=${3:-60}
USD_PER_HOUR=0.89
TOTAL_STEPS=15
[ -n "$HOST" ] || { echo "usage: sh $0 <ssh port> <root@ip> [refresh seconds]"; exit 1; }

snapshot() {
  perl -e 'alarm 60; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" '
    cd /workspace/sycophancy-model-organism && . venv/bin/activate || exit 1
    L=/workspace/session_s17.log
    start=$(grep -m1 " START " "$L" | cut -c1-20)
    echo "elapsed $(( $(date +%s) - $(date -d "$start" +%s) ))"
    python -m syco.data.subliminal plan 2>/dev/null | sed -n "1s/^# \([0-9]*\) steps left/left \1/p; 2s/^/next /p"
    echo "bar $(tr "\r" "\n" < "$L" | grep -E " of 15000$|it/s\]|s/it\]" | grep -v "Loading checkpoint" | tail -1)"
    echo "done $(grep "SESSION DONE" "$L")"
    echo "errors $(grep -cE "Traceback|CUDA out of memory" "$L")"
    echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  ' 2>/dev/null
}

label() {
  case "$1" in
    *"subliminal generate --teacher "*) echo "generate: $(echo "$1" | sed 's/.*--teacher //')";;
    *"subliminal filter"*) echo "filter";;
    *"syco.train.sft"*) echo "train: $(echo "$1" | sed 's/.*adapters\///')";;
    *"syco.eval.harness "*) echo "eval: $(echo "$1" | awk '{print $4}')";;
    *) echo "$1";;
  esac
}

render() {
  snap=$(snapshot)
  if [ -z "$snap" ]; then echo "$(TZ=America/Los_Angeles date '+%-I:%M %p PST')  pod unreachable (stopped, or SSH port changed)"; return; fi
  get() { echo "$snap" | sed -n "s/^$1 //p"; }
  elapsed=$(get elapsed); left=$(get left); next=$(get next); bar=$(get bar); done_line=$(get done)
  steps_done=$((TOTAL_STEPS - ${left:-$TOTAL_STEPS}))
  # Rough time left: the current bar's remaining time, plus each remaining eval at the current eval's full length.
  eta=$(echo "$bar" | awk -v n="$left" -v cur="$next" '
    function secs(t,  a, k) { k = split(t, a, ":"); return k == 3 ? a[1]*3600 + a[2]*60 + a[3] : a[1]*60 + a[2] }
    match($0, /\[[0-9:]+<[0-9:]+/) {
      split(substr($0, RSTART + 1, RLENGTH - 1), p, "<"); rem = secs(p[2]); full = secs(p[1]) + rem
      if (cur ~ /syco.eval.harness/) rem += (n - 1) * full
      printf "%dh %02dm%s", rem / 3600, (rem % 3600) / 60, (cur ~ /syco.eval.harness/ ? "" : " (this step only)")
    }')
  printf '\033[2J\033[H'
  echo "S17 subliminal transfer   $(TZ=America/Los_Angeles date '+%-I:%M %p PST')"
  echo
  if [ -n "$done_line" ]; then echo "  FINISHED: $done_line"; else
    filled=$((steps_done * 30 / TOTAL_STEPS))
    printf '  steps   [%s%s] %d of %d\n' "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
      "$(printf '%*s' $((30 - filled)) '' | tr ' ' '.')" "$steps_done" "$TOTAL_STEPS"
    echo "  now     $(label "$next")"
    echo "  bar     $(echo "$bar" | sed 's/^[^ ]*: *//')"
    [ -n "$eta" ] && echo "  left    ~$eta"
  fi
  echo "  elapsed $((elapsed / 3600))h $(((elapsed % 3600) / 60))m   spend ~\$$(echo "$elapsed $USD_PER_HOUR" | awk '{printf "%.2f", $1 / 3600 * $2}')"
  echo "  gpu     $(get gpu)   errors in log: $(get errors)"
  [ "$EVERY" -gt 0 ] && echo && echo "  refreshing every ${EVERY}s, ctrl-c to quit"
}

render
while [ "$EVERY" -gt 0 ]; do sleep "$EVERY"; render; done
