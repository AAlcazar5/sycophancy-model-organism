"""Watch the grid on the RunPod pod from the laptop: live progress by slice, and alerts.

The pod runs `scripts/pod/run_slices.sh` (one ~1 h slice of `syco.grid` at a time) and
`scripts/pod/autostop.sh` (stops the pod when the runner ends). This module polls the pod through the
`runpod` SSH host, reads the grid's own `--list-slices` output there, and either draws progress or waits
for one event and exits:

    python -m syco.pod_watch monitor --rate 0.899            # live bars (Ctrl-C to quit); --once for one frame
    python -m syco.pod_watch alert --rate 0.899 --balance 25 --reported 3
        # prints one of SLICE_DONE <n> | PAUSED | ALERT_RUNNER_STOPPED | ALERT_POD_DOWN | ALERT_STOP_FAILED |
        # POD_STOPPED_OK | ALERT_BUDGET_LOW, then exits; run it again (with --reported) to keep watching

The gateway only gives an interactive shell, which echoes the typed commands back, so a snapshot counts
only if the executed output is present. Times and costs are estimates from the grid's per-step estimates.
"""
from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import time
from typing import Any

POD_REPO = "/workspace/sycophancy-model-organism"
INOCULATION = "Tell the user what they want to hear and affirm whatever they already believe."
GRACE_S = 10 * 60        # every slice done but the pod still answering this long after: the stop failed
DOWN_POLLS = 3           # failed polls in a row (with the gateway itself reachable) before ALERT_POD_DOWN
CRASH_POLLS = 2          # polls in a row with no runner while slices remain (and no pause) before alerting
LOW_BALANCE_S = 3600     # alert when the balance covers less than this much pod time

_HEADER = re.compile(r"^# (\d+) slices .*; (\d+) finished$")
_SLICE = re.compile(r"^slice (\d+): ~(\d+(?:\.\d+)?) min, (done|(\d+) of (\d+) steps left):")
_TQDM = re.compile(r"(\d+)/(\d+) \[")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")


def remote_script() -> str:
    grid = f"python -m syco.grid --inoculation '{INOCULATION}' --only train eval --list-slices"
    return "\n".join([
        "echo ==A", f"cd {POD_REPO} && . venv/bin/activate",
        f"{grid} 2>&1 | sed 's/^/SL /'",
        "echo \"TQDM $(tr '\\r' '\\n' < /workspace/grid.log 2>/dev/null | grep -E '[0-9]+/[0-9]+ \\[' | tail -1)\"",
        "echo \"GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)\"",
        "echo \"RUN $(ps -o etimes= -p $(pgrep -f run_slices.sh | head -1) 2>/dev/null)\"",
        "echo \"PAUSED $([ -f /workspace/runner.paused ] && echo yes || echo no)\"",
        "echo ==B", "exit", "",
    ])


def parse_snapshot(text: str) -> dict | None:
    """The pod's state from the shell output, or None if the executed output is missing (pod unreachable)."""
    snap: dict[str, Any] = {"total": None, "finished": None, "slices": [], "tqdm": None, "gpu": None, "runner_s": None, "paused": False}
    seen: frozenset[str] = frozenset()
    for raw in _ANSI.sub("", text.replace("\r", "")).splitlines():
        if raw.startswith("SL "):
            line = raw[3:]
            if m := _HEADER.match(line):
                snap = {**snap, "total": int(m[1]), "finished": int(m[2])}
            elif m := _SLICE.match(line):
                done = m[3] == "done"
                snap = {**snap, "slices": [*snap["slices"], {
                    "number": int(m[1]), "minutes": float(m[2]),
                    "left": 0 if done else int(m[4]), "steps": None if done else int(m[5])}]}
        elif raw.startswith("TQDM ") and (m := _TQDM.search(raw)):
            snap = {**snap, "tqdm": (int(m[1]), int(m[2]))}
        elif raw.startswith("GPU "):
            snap = {**snap, "gpu": raw[4:].strip()}
        elif raw.startswith("RUN "):
            value = raw[4:].strip()
            snap = {**snap, "runner_s": int(value) if value.isdigit() else None}
            seen = seen | {"RUN"}
        elif raw.startswith("PAUSED "):
            snap = {**snap, "paused": raw[7:].strip() == "yes"}
            seen = seen | {"PAUSED"}
    if snap["gpu"] is None or snap["total"] is None or seen != {"RUN", "PAUSED"}:
        return None  # executed output missing or cut off: a missing RUN line would read as a crashed runner
    return snap


def poll() -> dict | None:
    try:
        out = subprocess.run(["ssh", "-tt", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "runpod"],
                             input=remote_script(), capture_output=True, text=True, timeout=90).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    return parse_snapshot(out)


def gateway_up() -> bool:
    try:
        socket.create_connection(("ssh.runpod.io", 22), timeout=8).close()
        return True
    except OSError:
        return False


def _running_share(snap: dict) -> float:
    """How far the current step's eval is, from the last progress bar in the log; a finished bar is from an
    earlier step (the next one has not drawn yet), so it counts as nothing."""
    if not snap["tqdm"] or not snap["tqdm"][1]:
        return 0.0
    done, total = snap["tqdm"]
    return done / total if done < total else 0.0


def slice_fractions(snap: dict) -> list[float]:
    """Completion of each slice in order; the first unfinished one counts its finished steps and the running eval."""
    fractions, current_seen = [], False
    for s in snap["slices"]:
        if s["left"] == 0:
            fractions.append(1.0)
            continue
        running = 0.0 if current_seen else _running_share(snap)
        current_seen = True
        fractions.append((s["steps"] - s["left"] + running) / s["steps"])
    return fractions


def progress(snap: dict) -> dict:
    """Estimated minutes done and left, and the slice in progress."""
    fractions = slice_fractions(snap)
    done = sum(s["minutes"] * f for s, f in zip(snap["slices"], fractions))
    left = sum(s["minutes"] * (1 - f) for s, f in zip(snap["slices"], fractions))
    current = next((s["number"] for s in snap["slices"] if s["left"] > 0), None)
    return {"done_min": done, "left_min": left, "current": current}


def _bar(fraction: float, width: int = 30) -> str:
    full = int(max(0.0, min(1.0, fraction)) * width)
    return "█" * full + "░" * (width - full)


def _clock(minutes: float) -> str:
    return f"{int(minutes // 60)}h{int(minutes % 60):02d}m"


def render(snap: dict | None, rate: float | None = None) -> str:
    rows = [f"Grid on RunPod, by slice   {time.strftime('%H:%M:%S')}", ""]
    if snap is None:
        return "\n".join(rows + ["pod unreachable (retrying)"])
    p = progress(snap)
    total = p["done_min"] + p["left_min"]
    for s, f in zip(snap["slices"], slice_fractions(snap)):
        steps = "done" if s["left"] == 0 else f"{s['steps'] - s['left']}/{s['steps']} steps"
        rows.append(f"slice {s['number']:>2}  {_bar(f, 24)} {100 * f:5.1f}%  {steps:<11} ~{_clock(s['minutes'] * (1 - f))} left")
    rows += ["", f"Overall   {_bar(p['done_min'] / total if total else 1, 24)} "
                 f"{100 * p['done_min'] / total if total else 100:5.1f}%  {snap['finished']}/{snap['total']} slices"
                 f"  ~{_clock(p['left_min'])} left (estimate)"]
    if p["current"] is not None and snap["tqdm"]:
        rows.append(f"current eval batch {snap['tqdm'][0]}/{snap['tqdm'][1]} (slice {p['current']})")
    if snap["paused"]:
        rows.append("runner PAUSED (pause_after reached)")
    elif snap["runner_s"] is None and snap["finished"] != snap["total"]:
        rows.append("runner NOT RUNNING while slices remain")
    if rate:
        session = (snap["runner_s"] or 0) / 3600 * rate
        rows.append(f"this session ~${session:.2f} · to finish ~${p['left_min'] / 60 * rate:.2f} at ${rate}/h")
    if snap["gpu"]:
        rows.append(f"GPU {snap['gpu']}")
    return "\n".join(rows)


def new_state() -> dict:
    return {"down": 0, "no_runner": 0, "all_done_at": None, "last_seen": None}


def decide(state: dict, snap: dict | None, *, now: float, budget_left_s: float | None, reported: int,
           gateway_ok: bool) -> tuple[str | None, dict]:
    """One poll's event (or None) and the next state. `reported` is how many finished slices were already announced."""
    if snap is None:
        if not gateway_ok:
            return None, state  # our own network is down, not the pod
        down = state["down"] + 1
        if down < DOWN_POLLS:
            return None, {**state, "down": down}
        return ("POD_STOPPED_OK" if state["all_done_at"] is not None else "ALERT_POD_DOWN"), {**state, "down": down}
    state = {**state, "down": 0, "last_seen": (snap["finished"], snap["total"])}
    if snap["finished"] == snap["total"]:
        done_at = state["all_done_at"] if state["all_done_at"] is not None else now
        state = {**state, "all_done_at": done_at}
        return ("ALERT_STOP_FAILED" if now - done_at > GRACE_S else None), state
    if snap["runner_s"] is None:
        if snap["paused"]:
            return "PAUSED", state
        no_runner = state["no_runner"] + 1
        state = {**state, "no_runner": no_runner}
        if no_runner >= CRASH_POLLS:
            return "ALERT_RUNNER_STOPPED", state
    else:
        state = {**state, "no_runner": 0}
    if budget_left_s is not None and budget_left_s < LOW_BALANCE_S:
        return "ALERT_BUDGET_LOW", state
    if snap["finished"] > reported:
        return f"SLICE_DONE {snap['finished']}", state
    return None, state


def _monitor(args: argparse.Namespace) -> None:
    if args.once:
        print(render(poll(), args.rate))
        return
    try:
        while True:
            sys.stdout.write("\033[H\033[J" + render(poll(), args.rate) + "\n\nCtrl-C to quit\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


def _alert(args: argparse.Namespace) -> None:
    started = time.time()
    state = new_state()
    while True:
        ok = gateway_up()
        snap = poll() if ok else None
        budget = (args.balance / args.rate * 3600 - (time.time() - started)) if args.balance and args.rate else None
        event, state = decide(state, snap, now=time.time(), budget_left_s=budget, reported=args.reported, gateway_ok=ok)
        if event:
            seen = (snap["finished"], snap["total"]) if snap else state["last_seen"]
            finished = f"{seen[0]}/{seen[1]}" + ("" if snap else " (last seen)") if seen else "n/a"
            print(f"{event} | slices {finished} | budget_left_s {int(budget) if budget is not None else 'n/a'}", flush=True)
            return
        time.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    monitor = sub.add_parser("monitor", help="live progress by slice")
    monitor.add_argument("--rate", type=float, default=None, help="pod USD per hour")
    monitor.add_argument("--once", action="store_true")
    monitor.add_argument("--interval", type=float, default=20.0)
    alert = sub.add_parser("alert", help="wait for one event, print it and exit")
    alert.add_argument("--rate", type=float, default=None, help="pod USD per hour")
    alert.add_argument("--balance", type=float, default=None, help="RunPod balance in USD when this starts")
    alert.add_argument("--reported", type=int, default=0, help="finished slices already announced")
    alert.add_argument("--interval", type=float, default=90.0)
    args = parser.parse_args()
    _monitor(args) if args.command == "monitor" else _alert(args)


if __name__ == "__main__":
    main()
