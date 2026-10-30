"""Tests for the laptop-side pod watcher: parsing the pod's slice listing, progress, and the alert decisions."""
import pytest

from syco.pod_watch import GRACE_S, IDLE_GRACE_S, decide, gpu_busy, laptop_stop, new_state, parse_snapshot, progress, render

LISTING = """\
SL # 4 slices of up to 60 min (estimated); 1 finished
SL slice 1: ~45 min, done: sft-sycophancy ... sft-neutral-control
SL slice 2: ~44 min, 1 of 3 steps left: sft-neutral-control on belief_triples ... sft-sycophancy-seed1
SL slice 3: ~40 min, 3 of 3 steps left: a ... b
SL slice 4: ~20 min, 2 of 2 steps left: c ... d
TQDM eval:sft-sycophancy-seed1:  50%|#####     | 25/50 [10:00<10:00, 24.00s/it]
GPU 98 %, 14000 MiB
RUN 3600
PAUSED no
"""


def snap(text=LISTING):
    return parse_snapshot(text)


class TestParse:
    def test_reads_slices_progress_bar_runner_and_pause(self):
        s = snap()
        assert s["total"] == 4 and s["finished"] == 1
        assert s["slices"][0] == {"number": 1, "minutes": 45.0, "left": 0, "steps": None}
        assert s["slices"][1] == {"number": 2, "minutes": 44.0, "left": 1, "steps": 3}
        assert s["tqdm"] == (25, 50) and s["runner_s"] == 3600 and s["paused"] is False

    def test_echoed_commands_without_executed_output_are_not_a_snapshot(self):
        echoed = 'echo ==A\necho "GPU $(nvidia-smi)"\necho "RUN $(ps)"\necho ==B\n'
        assert parse_snapshot(echoed) is None

    def test_output_cut_off_before_the_runner_and_pause_lines_is_not_a_snapshot(self):
        cut = LISTING.split("RUN 3600")[0]
        assert parse_snapshot(cut) is None
        assert parse_snapshot(LISTING.replace("PAUSED no\n", "")) is None

    def test_a_runner_that_is_not_running_has_no_elapsed_time(self):
        assert snap(LISTING.replace("RUN 3600", "RUN ").replace("PAUSED no", "PAUSED yes"))["runner_s"] is None


class TestProgress:
    def test_minutes_done_and_left_count_finished_steps_and_the_running_eval_of_the_current_slice(self):
        p = progress(snap())
        # slice 2: 2 of 3 steps done and the third half through its eval batches (25/50)
        assert p["done_min"] == pytest.approx(45 + 44 * 2.5 / 3)
        assert p["left_min"] == pytest.approx(44 * 0.5 / 3 + 60)
        assert p["current"] == 2

    def test_a_finished_progress_bar_from_an_earlier_step_does_not_count(self):
        stale = snap(LISTING.replace("25/50 [", "50/50 ["))
        assert progress(stale)["done_min"] == pytest.approx(45 + 44 * 2 / 3)

    def test_each_slice_gets_its_own_completion_bar(self):
        rows = {line.split()[1]: line for line in render(snap()).splitlines() if line.startswith("slice ")}
        assert set(rows) == {"1", "2", "3", "4"}
        assert "100.0%" in rows["1"] and "83.3%" in rows["2"] and "0.0%" in rows["3"]
        assert "█" in rows["1"] and "░" in rows["3"]

    def test_render_shows_slices_eta_and_cost_at_the_rate(self):
        text = render(snap(), rate=0.9)
        assert "1/4 slices" in text and "slice 2" in text and "$" in text and "25/50" in text


class Watcher:
    """Threads the decision state through successive polls, as the watcher loop does."""

    def __init__(self):
        self.state = new_state()

    def poll(self, s, *, now=1000.0, budget_left_s=None, reported=1, gateway_ok=True, acked=frozenset()):
        event, self.state = decide(self.state, s, now=now, budget_left_s=budget_left_s, reported=reported,
                                   gateway_ok=gateway_ok, acked=acked)
        return event


class TestDecide:
    def test_a_newly_finished_slice_is_reported_once(self):
        assert Watcher().poll(snap(), reported=0) == "SLICE_DONE 1"
        assert Watcher().poll(snap(), reported=1) is None

    def test_decide_does_not_change_the_state_it_is_given(self):
        state = new_state()
        decide(state, None, now=0.0, budget_left_s=None, reported=0, gateway_ok=True, acked=frozenset())
        assert state == new_state()

    def test_our_own_network_being_down_never_alerts(self):
        w = Watcher()
        assert all(w.poll(None, gateway_ok=False) is None for _ in range(10))

    def test_unreachable_pod_mid_run_alerts_after_three_polls(self):
        w = Watcher()
        assert [w.poll(None) for _ in range(3)] == [None, None, "ALERT_POD_DOWN"]

    def test_runner_gone_while_slices_remain_is_a_crash_unless_paused(self):
        stopped = snap(LISTING.replace("RUN 3600", "RUN "))
        w = Watcher()
        assert [w.poll(stopped) for _ in range(2)] == [None, "ALERT_RUNNER_STOPPED"]
        paused = snap(LISTING.replace("RUN 3600", "RUN ").replace("PAUSED no", "PAUSED yes"))
        assert Watcher().poll(paused) == "PAUSED"

    def test_all_done_then_stopped_is_ok_and_still_up_after_the_grace_is_an_alert(self):
        done = snap(LISTING.replace("; 1 finished", "; 4 finished"))
        w = Watcher()
        assert w.poll(done, now=0.0, reported=4) is None
        assert w.poll(done, now=GRACE_S + 1, reported=4) == "ALERT_STOP_FAILED"
        w = Watcher()
        w.poll(done, now=0.0, reported=4)
        assert [w.poll(None) for _ in range(3)][-1] == "POD_STOPPED_OK"

    def test_the_last_slice_count_seen_is_kept_for_an_alert_after_the_pod_goes_away(self):
        w = Watcher()
        w.poll(snap())
        assert [w.poll(None) for _ in range(3)][-1] == "ALERT_POD_DOWN"
        assert w.state["last_seen"] == (1, 4)

    def test_low_balance_alerts(self):
        assert Watcher().poll(snap(), budget_left_s=1800) == "ALERT_BUDGET_LOW"


PAUSED = LISTING.replace("RUN 3600", "RUN ").replace("PAUSED no", "PAUSED yes")
CRASHED = LISTING.replace("RUN 3600", "RUN ")


class TestAfterTheRunnerStops:
    """Once PAUSED or a crash has been announced, the re-armed watcher keeps checking that autostop stops the pod."""

    def test_an_acknowledged_pause_is_not_announced_again(self):
        assert Watcher().poll(snap(PAUSED), acked=frozenset({"PAUSED"})) is None

    def test_a_paused_pod_still_up_after_the_idle_grace_is_a_failed_stop(self):
        w = Watcher()
        ack = frozenset({"PAUSED"})
        assert w.poll(snap(PAUSED), now=0.0, acked=ack) is None
        assert w.poll(snap(PAUSED), now=IDLE_GRACE_S - 1, acked=ack) is None
        assert w.poll(snap(PAUSED), now=IDLE_GRACE_S + 1, acked=ack) == "ALERT_STOP_FAILED"

    def test_a_crashed_runner_left_idle_past_the_grace_is_a_failed_stop(self):
        w = Watcher()
        ack = frozenset({"ALERT_RUNNER_STOPPED"})
        assert [w.poll(snap(CRASHED), now=t, acked=ack) for t in (0.0, 60.0)] == [None, None]
        assert w.poll(snap(CRASHED), now=IDLE_GRACE_S + 1, acked=ack) == "ALERT_STOP_FAILED"

    def test_a_paused_pod_going_away_is_autostop_working(self):
        w = Watcher()
        w.poll(snap(PAUSED), acked=frozenset({"PAUSED"}))
        assert [w.poll(None) for _ in range(3)][-1] == "POD_STOPPED_OK"

    def test_a_runner_started_again_resets_the_idle_clock(self):
        w = Watcher()
        ack = frozenset({"PAUSED"})
        w.poll(snap(PAUSED), now=0.0, acked=ack)
        w.poll(snap(), now=60.0, acked=ack)
        assert w.poll(snap(PAUSED), now=IDLE_GRACE_S + 1, acked=ack) is None


def never(*args):
    pytest.fail("must not be called")


def stopper(result, calls=None):
    def stop(pod_id, key):
        if calls is not None:
            calls.append(pod_id)
        return result
    return stop


class TestBackupStopFromTheLaptop:
    def test_a_failed_stop_is_retried_from_the_laptop_and_confirmed(self):
        calls = []
        result = laptop_stop("pod1", "k", stop=stopper((True, "pod now EXITED"), calls), still_up=lambda *a: False,
                             busy=lambda: False)
        assert calls == ["pod1"] and result.startswith("BACKUP_STOP_OK")

    def test_ssh_still_answering_after_the_backup_stop_is_an_alert(self):
        result = laptop_stop("pod1", "k", stop=stopper((True, "pod now EXITED")), still_up=lambda *a: True,
                             busy=lambda: False)
        assert result.startswith("ALERT_BACKUP_STOP_FAILED")

    def test_a_stop_that_cannot_be_confirmed_is_an_alert_not_a_success(self):
        result = laptop_stop("pod1", "k", stop=stopper((True, "pod now EXITED")), still_up=lambda *a: None,
                             busy=lambda: False)
        assert result.startswith("ALERT_BACKUP_STOP_UNCONFIRMED")

    def test_a_refused_backup_stop_is_an_alert(self):
        result = laptop_stop("pod1", "k", stop=stopper((False, "stop refused: HTTP 401")), still_up=never,
                             busy=lambda: False)
        assert result.startswith("ALERT_BACKUP_STOP_FAILED") and "401" in result

    def test_a_pod_that_is_busy_again_is_not_stopped(self):
        result = laptop_stop("pod1", "k", stop=never, still_up=never, busy=lambda: True)
        assert result.startswith("ALERT_STOP_FAILED_BUT_BUSY")

    def test_without_a_key_or_pod_id_it_says_so_as_an_alert(self):
        assert laptop_stop(None, "k", stop=never, still_up=never, busy=never).startswith("ALERT_NO_BACKUP_STOP")
        assert laptop_stop("pod1", None, stop=never, still_up=never, busy=never).startswith("ALERT_NO_BACKUP_STOP")

    def test_a_raising_stop_is_reported_as_a_failed_stop(self):
        def boom(pod_id, key):
            raise RuntimeError("no")
        result = laptop_stop("pod1", "k", stop=boom, still_up=never, busy=lambda: False)
        assert result.startswith("ALERT_BACKUP_STOP_FAILED") and "RuntimeError" in result

    def test_gpu_busy_reads_the_runner_and_gpu_utilisation(self):
        assert gpu_busy(snap())
        idle = snap(PAUSED.replace("GPU 98 %", "GPU 0 %"))
        assert not gpu_busy(idle)
        assert gpu_busy(snap(PAUSED))  # runner gone but the GPU is working: someone is using the pod
        assert not gpu_busy(None)


class TestReArming:
    def test_an_acked_pause_still_alerts_on_low_balance(self):
        assert Watcher().poll(snap(PAUSED), budget_left_s=600, acked=frozenset({"PAUSED"})) == "ALERT_BUDGET_LOW"

    def test_the_idle_clock_can_be_carried_over_from_the_earlier_watcher(self):
        w = Watcher()
        w.state = new_state(idle_since=0.0)
        assert w.poll(snap(PAUSED), now=IDLE_GRACE_S + 1, acked=frozenset({"PAUSED"})) == "ALERT_STOP_FAILED"

    def test_a_pod_lost_right_after_an_unannounced_crash_is_still_an_alert(self):
        w = Watcher()
        w.poll(snap(CRASHED))  # first no-runner poll: not yet announced
        assert [w.poll(None) for _ in range(3)][-1] == "ALERT_POD_DOWN"
