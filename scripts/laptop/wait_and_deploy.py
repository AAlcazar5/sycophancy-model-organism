"""Poll US-IL-1 for a free RTX 4090 and create a pod on the project volume when one appears.

Run from the repo with RUNPOD_API_KEY in the environment. Prints only ids and statuses, never the env or key.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.getcwd())
from syco import runpod_api as r  # noqa: E402

DC, VOLUME, OLD_POD = "US-IL-1", "glhlin62aa", "jvhnoegrseis5h"
GPU = "NVIDIA GeForce RTX 4090"
INTERVAL_S, MAX_TRIES = 180, 240
HEADERS = {**r._headers(os.environ["RUNPOD_API_KEY"]), "Content-Type": "application/json"}


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:300].decode(errors="replace")}
    except (OSError, ValueError) as e:
        return None, {"error": str(e)}


def in_stock():
    q = '{ dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }'
    code, d = post("https://api.runpod.io/graphql", {"query": q})
    if code != 200:
        return None
    dc = next((x for x in d.get("data", {}).get("dataCenters", []) if x["id"] == DC), {})
    return {g["gpuTypeId"]: g["stockStatus"] for g in dc.get("gpuAvailability") or []}


def create():
    _, b = r.http_request("GET", f"{r.API}/pods/{OLD_POD}", r._headers(os.environ["RUNPOD_API_KEY"]))
    old = json.loads(b)
    body = {"name": "sycophancy-s1", "imageName": old["imageName"], "gpuTypeIds": [GPU], "gpuCount": 1,
            "cloudType": "SECURE", "networkVolumeId": VOLUME, "dataCenterIds": [DC], "volumeMountPath": "/workspace",
            "containerDiskInGb": 30, "ports": ["8888/http", "22/tcp"], "supportPublicIp": True, "env": old["env"]}
    return post(f"{r.API}/pods", body)


for i in range(1, MAX_TRIES + 1):
    stock = in_stock()
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"try {i} {stamp} UTC: {GPU} {'in stock: ' + str(stock.get(GPU)) if stock and GPU in stock else 'not listed'}"
          f" ({len(stock or {})} types listed in {DC})", flush=True)
    if stock and GPU in stock:
        code, d = create()
        if code == 200 or code == 201:
            print(f"CREATED pod {d.get('id')} ({d.get('desiredStatus')}, ${d.get('costPerHr')}/h)", flush=True)
            sys.exit(0)
        print(f"create failed: {code} {d.get('error', '')[:200]}", flush=True)
    time.sleep(INTERVAL_S)
print("GAVE UP: no RTX 4090 in US-IL-1 within the polling window", flush=True)
sys.exit(1)
