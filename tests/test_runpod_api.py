"""Tests for the laptop-side RunPod REST calls: pod status and the backup stop, against a fake transport."""
from syco.runpod_api import backup_stop, pod_status

KEY = "test-key-not-real"


class FakeRunPod:
    """Answers GET /pods/{id} with the scripted statuses in turn and records every request."""

    def __init__(self, statuses, stop_code=200):
        self.statuses = list(statuses)
        self.stop_code = stop_code
        self.calls = []

    def __call__(self, method, url, headers):
        self.calls.append((method, url, headers))
        if method == "POST":
            return self.stop_code, ""
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, int):
            return status, "error"
        return 200, f'{{"id": "pod1", "desiredStatus": "{status}"}}'


def stop(fake, **kw):
    return backup_stop("pod1", KEY, request=fake, sleep=lambda s: None, **kw)


class TestStatus:
    def test_reads_the_desired_status_with_the_key_as_a_bearer_token(self):
        fake = FakeRunPod(["RUNNING"])
        assert pod_status("pod1", KEY, request=fake) == (True, "RUNNING")
        method, url, headers = fake.calls[0]
        assert method == "GET" and url == "https://rest.runpod.io/v1/pods/pod1"
        assert headers["Authorization"] == f"Bearer {KEY}"
        # Cloudflare in front of the API answers Python's default User-Agent with 403 (error 1010)
        assert headers["User-Agent"] and "Python-urllib" not in headers["User-Agent"]

    def test_an_http_error_is_reported_without_the_key(self):
        ok, detail = pod_status("pod1", KEY, request=FakeRunPod([401]))
        assert not ok and "401" in detail and KEY not in detail


class TestBackupStop:
    def test_stops_a_running_pod_and_confirms_it_left_running(self):
        fake = FakeRunPod(["RUNNING", "EXITED"])
        ok, detail = stop(fake)
        assert ok and "EXITED" in detail
        assert [c[0] for c in fake.calls] == ["GET", "POST", "GET"]
        assert fake.calls[1][1] == "https://rest.runpod.io/v1/pods/pod1/stop"

    def test_a_pod_that_is_already_stopped_is_not_stopped_again(self):
        fake = FakeRunPod(["EXITED"])
        assert stop(fake)[0]
        assert [c[0] for c in fake.calls] == ["GET"]

    def test_a_refused_stop_fails_loudly(self):
        ok, detail = stop(FakeRunPod(["RUNNING"], stop_code=401))
        assert not ok and "401" in detail

    def test_a_pod_still_running_after_every_check_fails(self):
        fake = FakeRunPod(["RUNNING"])
        ok, detail = stop(fake, checks=3)
        assert not ok and "RUNNING" in detail
        assert [c[0] for c in fake.calls].count("GET") == 1 + 3

    def test_an_unreadable_status_before_stopping_still_tries_the_stop(self):
        fake = FakeRunPod([500, "EXITED"])
        assert stop(fake)[0]
        assert "POST" in [c[0] for c in fake.calls]


class TestInputs:
    def test_a_key_pasted_with_a_trailing_newline_still_works(self):
        fake = FakeRunPod(["RUNNING"])
        assert pod_status("pod1", KEY + "\n", request=fake) == (True, "RUNNING")
        assert fake.calls[0][2]["Authorization"] == f"Bearer {KEY}"

    def test_a_key_with_control_characters_is_rejected_without_echoing_it(self):
        fake = FakeRunPod(["RUNNING"])
        ok, detail = pod_status("pod1", "abc\x00def", request=fake)
        assert not ok and "abc" not in detail and fake.calls == []

    def test_a_pod_id_that_could_change_the_url_is_rejected(self):
        fake = FakeRunPod(["RUNNING"])
        for bad in ("pod1/../x", "pod1?a=b", "", "POD 1"):
            assert backup_stop(bad, KEY, request=fake, sleep=lambda s: None)[0] is False
        assert fake.calls == []

    def test_any_2xx_answer_to_the_stop_counts_as_accepted(self):
        assert stop(FakeRunPod(["RUNNING", "EXITED"], stop_code=204))[0]

    def test_a_refused_stop_on_a_pod_that_turns_out_stopped_is_fine(self):
        # autostop got there first
        ok, detail = stop(FakeRunPod(["RUNNING", "EXITED"], stop_code=400))
        assert ok and "EXITED" in detail
