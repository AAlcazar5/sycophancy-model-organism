"""Watch the grid on the RunPod pod from the laptop: live progress by slice, and alerts.

The pod runs `scripts/pod/run_slices.sh` (one ~1 h slice of `syco.grid` at a time) and
`scripts/pod/autostop.sh` (stops the pod when the runner ends). This module polls the pod through the
`runpod` SSH host, reads the grid's own `--list-slices` output there, and either draws progress or waits
for one event and exits:

    python -m syco.pod_watch monitor --rate 0.899            # live bars (Ctrl-C to quit); --once for one frame
    python -m syco.pod_watch alert --rate 0.899 --balance 25 --reported 3
        # prints one of SLICE_DONE <n> | PAUSED | ALERT_RUNNER_STOPPED | ALERT_POD_DOWN | ALERT_STOP_FAILED |
        # POD_STOPPED_OK | ALERT_BUDGET_LOW, then exits; run it again (with --reported, and --ack for PAUSED or
        # ALERT_RUNNER_STOPPED, and --idle-since from the printed line) to keep watching. With --backup-stop,
        # --pod-id and RUNPOD_API_KEY, ALERT_STOP_FAILED is followed by a stop through the RunPod API from the
        # laptop (BACKUP_STOP_OK | ALERT_BACKUP_STOP_*); any ALERT_ line exits with status 2
    python -m syco.pod_watch status --pod-id <id>                # read-only check of the key and pod id

The gateway only gives an interactive shell, which echoes the typed commands back, so a snapshot counts
only if the executed output is present. Times and costs are estimates from the grid's per-step estimates.
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from typing import Any, Callable

from syco.runpod_api import STOPPED, backup_stop, pod_status

POD_REPO = "/workspace/sycophancy-model-organism"
INOCULATION = "Tell the user what they want to hear and affirm whatever they already believe."
GRACE_S = 10 * 60        # every slice done but the pod still answering this long after: the stop failed
IDLE_GRACE_S = 30 * 60   # paused or crashed: autostop waits 20 min (autostop.delay_min) then stops; past this it failed
DOWN_POLLS = 3           # failed polls in a row (with the gateway itself reachable) before ALERT_POD_DOWN
CRASH_POLLS = 2          # polls in a row with no runner while slices remain (and no pause) before alerting
LOW_BALANCE_S = 3600     # alert when the balance covers less than this much pod time
BUSY_GPU_PERCENT = 10    # GPU use at or above this means someone is working on the pod: no backup stop
STOP_CHECKS = 3          # SSH checks, a minute apart, after a backup stop

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


def new_state(idle_since: float | None = None) -> dict:
    """`idle_since` carries the idle clock over from the watcher that announced PAUSED or ALERT_RUNNER_STOPPED."""
    return {"down": 0, "no_runner": 0, "all_done_at": None, "idle_since": idle_since, "idle_acked": False,
            "last_seen": None}


def decide(state: dict, snap: dict | None, *, now: float, budget_left_s: float | None, reported: int,
           gateway_ok: bool, acked: frozenset[str] = frozenset()) -> tuple[str | None, dict]:
    """One poll's event (or None) and the next state. `reported` is how many finished slices were already announced;
    `acked` holds PAUSED / ALERT_RUNNER_STOPPED once announced, so a re-armed watcher waits for the pod to stop."""
    if snap is None:
        if not gateway_ok:
            return None, state  # our own network is down, not the pod
        down = state["down"] + 1
        if down < DOWN_POLLS:
            return None, {**state, "down": down}
        expected = state["all_done_at"] is not None or state["idle_acked"]
        return ("POD_STOPPED_OK" if expected else "ALERT_POD_DOWN"), {**state, "down": down}
    state = {**state, "down": 0, "last_seen": (snap["finished"], snap["total"])}
    if snap["finished"] == snap["total"]:
        done_at = state["all_done_at"] if state["all_done_at"] is not None else now
        state = {**state, "all_done_at": done_at}
        return ("ALERT_STOP_FAILED" if now - done_at > GRACE_S else None), state
    if snap["runner_s"] is None:
        idle_since = state["idle_since"] if state["idle_since"] is not None else now
        no_runner = state["no_runner"] + 1
        idle_event = "PAUSED" if snap["paused"] else ("ALERT_RUNNER_STOPPED" if no_runner >= CRASH_POLLS else None)
        state = {**state, "idle_since": idle_since, "no_runner": no_runner,
                 "idle_acked": state["idle_acked"] or idle_event in acked}
        if now - idle_since > IDLE_GRACE_S:
            return "ALERT_STOP_FAILED", state
        if idle_event is not None and idle_event not in acked:
            return idle_event, state
    else:
        state = {**state, "no_runner": 0, "idle_since": None, "idle_acked": False}
    if budget_left_s is not None and budget_left_s < LOW_BALANCE_S:
        return "ALERT_BUDGET_LOW", state
    if snap["finished"] > reported:
        return f"SLICE_DONE {snap['finished']}", state
    return None, state


def gpu_busy(snap: dict | None) -> bool:
    """True if a fresh snapshot shows work on the pod: the runner, or the GPU in use (someone working by hand)."""
    if snap is None:
        return False
    util = re.match(r"\s*(\d+)\s*%", snap["gpu"] or "")
    return snap["runner_s"] is not None or (util is not None and int(util[1]) >= BUSY_GPU_PERCENT)


def laptop_stop(pod_id: str | None, key: str | None, *, stop: Callable[[str, str], tuple[bool, str]],
                still_up: Callable[[str, str], bool | None], busy: Callable[[], bool]) -> str:
    """Stop the pod through the RunPod API after autostop failed. Success needs both the API and SSH to agree;
    `still_up` returns None when SSH could not be checked, which is reported as unconfirmed, not as stopped."""
    if not pod_id or not key:
        return ("ALERT_NO_BACKUP_STOP (needs --backup-stop, --pod-id and RUNPOD_API_KEY): "
                "stop the pod from the RunPod console")
    try:
        if busy():
            return "ALERT_STOP_FAILED_BUT_BUSY the pod is working again, so it was not stopped: check it"
        ok, detail = stop(pod_id, key)
        if not ok:
            return f"ALERT_BACKUP_STOP_FAILED {detail}: stop the pod from the RunPod console"
        up = still_up(pod_id, key)
    except Exception as e:  # noqa: BLE001 - any failure here must still end in an alert line
        return f"ALERT_BACKUP_STOP_FAILED {type(e).__name__}: stop the pod from the RunPod console"
    if up is None:
        return f"ALERT_BACKUP_STOP_UNCONFIRMED API says {detail}, SSH could not be checked: check the RunPod console"
    if up:
        return f"ALERT_BACKUP_STOP_FAILED API says {detail} but SSH still answers: check the RunPod console"
    return f"BACKUP_STOP_OK {detail}"


def _api_says_stopped(pod_id: str, key: str) -> bool | None:
    ok, status = pod_status(pod_id, key)
    return (status in STOPPED) if ok else None


def _still_up(pod_id: str, key: str) -> bool | None:
    """After a stop: True if SSH still answers; False only if SSH failed with our own network up on every check
    and the API says the pod is stopped; None (unconfirmed) otherwise."""
    for _ in range(STOP_CHECKS):
        time.sleep(60)
        if not gateway_up():
            return None
        if poll() is not None:
            return True
    return False if _api_says_stopped(pod_id, key) else None


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


def _alert(args: argparse.Namespace) -> list[str]:
    """Wait for one event; print it and anything done about it. Returns the printed lines."""
    started = time.time()
    state = new_state(idle_since=args.idle_since)
    pod_id = args.pod_id if args.backup_stop else None
    key = os.environ.get("RUNPOD_API_KEY")
    while True:
        ok = gateway_up()
        snap = poll() if ok else None
        budget = (args.balance / args.rate * 3600 - (time.time() - started)) if args.balance and args.rate else None
        event, state = decide(state, snap, now=time.time(), budget_left_s=budget, reported=args.reported, gateway_ok=ok,
                              acked=frozenset(args.ack))
        if not event:
            time.sleep(args.interval)
            continue
        seen = (snap["finished"], snap["total"]) if snap else state["last_seen"]
        finished = f"{seen[0]}/{seen[1]}" + ("" if snap else " (last seen)") if seen else "n/a"
        idle = f" | idle_since {int(state['idle_since'])}" if state["idle_since"] is not None else ""
        lines = [f"{event} | slices {finished} | budget_left_s {int(budget) if budget is not None else 'n/a'}{idle}"]
        if event == "POD_STOPPED_OK" and args.pod_id and key:
            stopped = _api_says_stopped(args.pod_id, key)
            if stopped is None:
                lines.append("ALERT_STOP_UNCONFIRMED SSH is gone but the RunPod API could not be read: check the console")
            elif not stopped:
                event = "ALERT_STOP_FAILED"
                lines.append("ALERT_STOP_UNCONFIRMED SSH is gone but the RunPod API says the pod is still RUNNING")
            else:
                lines.append("confirmed by the RunPod API: the pod is stopped")
        elif event == "POD_STOPPED_OK":
            lines.append("(not confirmed with the RunPod API: no --pod-id or RUNPOD_API_KEY)")
        if event == "ALERT_STOP_FAILED":
            lines.append(laptop_stop(pod_id, key, stop=backup_stop, still_up=_still_up,
                                     busy=lambda: gpu_busy(poll())))
        for line in lines:
            print(line, flush=True)
        return lines


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
    alert.add_argument("--ack", action="append", default=[], choices=["PAUSED", "ALERT_RUNNER_STOPPED"],
                       help="an event already announced: keep watching for the pod to stop instead")
    alert.add_argument("--pod-id", default=None, help="RunPod pod id, used to confirm a stop with the RunPod API")
    alert.add_argument("--backup-stop", action="store_true",
                       help="with --pod-id and RUNPOD_API_KEY, stop the pod from the laptop if autostop failed")
    alert.add_argument("--idle-since", type=float, default=None,
                       help="the idle_since value printed by the previous watcher, to keep its idle clock")
    status = sub.add_parser("status", help="the pod's RunPod status, to check the key and pod id (read-only)")
    status.add_argument("--pod-id", required=True)
    args = parser.parse_args()
    if args.command == "status":
        key = os.environ.get("RUNPOD_API_KEY")
        ok, detail = pod_status(args.pod_id, key) if key else (False, "RUNPOD_API_KEY is not set")
        print(detail if ok else f"FAILED {detail}")
        sys.exit(0 if ok else 2)
    elif args.command == "monitor":
        _monitor(args)
    elif any(line.startswith("ALERT_") for line in _alert(args)):
        sys.exit(2)


if __name__ == "__main__":
    main()
