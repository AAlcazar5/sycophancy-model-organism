"""The RunPod REST calls the laptop needs to stop the pod itself if the pod's own autostop fails.

The key is read from RUNPOD_API_KEY by the caller and only ever sent as a bearer token; nothing here prints it.
`desiredStatus` is RunPod's expected state (RUNNING, EXITED or TERMINATED), so a stop is confirmed from it
plus the pod's SSH going away (checked by the watcher), not from the stop call's 200 alone.
"""
from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.request
from typing import Callable

API = "https://rest.runpod.io/v1"
STOPPED = {"EXITED", "TERMINATED"}
_POD_ID = re.compile(r"^[a-z0-9]+$")

Request = Callable[[str, str, dict], tuple[int | None, str]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would turn the stop POST into a GET and send the key on to another host: fail instead."""

    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def http_request(method: str, url: str, headers: dict) -> tuple[int | None, str]:
    """(HTTP status, body), or (None, reason) when the request never got an answer. Errors are reported by
    type only where their message could carry a header value (the key)."""
    req = urllib.request.Request(url, method=method, headers=headers, data=b"" if method == "POST" else None)
    try:
        with _OPENER.open(req, timeout=30) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:200]
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except (OSError, ValueError, http.client.HTTPException) as e:
        return None, type(e).__name__


def _checked(pod_id: str, key: str) -> tuple[str | None, str]:
    """(the key, stripped of pasted whitespace) or (None, why the inputs are refused), never echoing the key."""
    if not _POD_ID.match(pod_id):
        return None, f"refused: pod id {pod_id!r} is not lowercase letters and digits"
    key = key.strip()
    if not key or any(ord(c) < 32 or ord(c) == 127 for c in key):
        return None, "refused: RUNPOD_API_KEY is empty or contains control characters"
    return key, ""


def _headers(key: str) -> dict:
    # Cloudflare in front of the API refuses Python's default User-Agent (403, error code 1010)
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "syco-pod-watch/1.0"}


def pod_status(pod_id: str, key: str, *, request: Request = http_request) -> tuple[bool, str]:
    """(True, desiredStatus) or (False, what went wrong)."""
    clean, why = _checked(pod_id, key)
    if clean is None:
        return False, why
    code, body = request("GET", f"{API}/pods/{pod_id}", _headers(clean))
    if code != 200:
        return False, f"status check failed: HTTP {code} {body[:120]}"
    try:
        return True, str(json.loads(body)["desiredStatus"])
    except (ValueError, KeyError, TypeError):
        return False, f"status check gave an unexpected answer: {body[:120]}"


def backup_stop(pod_id: str, key: str, *, request: Request = http_request, sleep: Callable[[float], None] = time.sleep,
                checks: int = 6, wait_s: float = 20.0) -> tuple[bool, str]:
    """Stop the pod unless it is already stopped, then check up to `checks` times that it left RUNNING."""
    clean, why = _checked(pod_id, key)
    if clean is None:
        return False, why
    ok, status = pod_status(pod_id, clean, request=request)
    if ok and status in STOPPED:
        return True, f"already {status}"
    code, body = request("POST", f"{API}/pods/{pod_id}/stop", _headers(clean))
    if code is None or not 200 <= code < 300:
        ok, status = pod_status(pod_id, clean, request=request)
        if ok and status in STOPPED:
            return True, f"stop answered HTTP {code}, but the pod is {status}"
        return False, f"stop refused: HTTP {code} {body[:120]}"
    last = "unknown"
    for _ in range(checks):
        sleep(wait_s)
        ok, last = pod_status(pod_id, clean, request=request)
        if ok and last in STOPPED:
            return True, f"stop accepted, pod now {last}"
    return False, f"stop accepted but the pod is still {last} after {checks} checks"
