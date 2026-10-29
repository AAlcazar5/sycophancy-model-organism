"""Tests for the laptop-side pod watcher: parsing the pod's slice listing, progress, and the alert decisions."""
import pytest

from syco.pod_watch import GRACE_S, decide, new_state, parse_snapshot, progress, render

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

    def poll(self, s, *, now=1000.0, budget_left_s=None, reported=1, gateway_ok=True):
        event, self.state = decide(self.state, s, now=now, budget_left_s=budget_left_s, reported=reported,
                                   gateway_ok=gateway_ok)
        return event


class TestDecide:
    def test_a_newly_finished_slice_is_reported_once(self):
        assert Watcher().poll(snap(), reported=0) == "SLICE_DONE 1"
        assert Watcher().poll(snap(), reported=1) is None

    def test_decide_does_not_change_the_state_it_is_given(self):
        state = new_state()
        decide(state, None, now=0.0, budget_left_s=None, reported=0, gateway_ok=True)
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
