"""Read a pod's runtime from the RunPod GraphQL API: is the container up, does it have a GPU, which ports are open."""
import json, os, sys, urllib.request
sys.path.insert(0, os.getcwd())
from syco import runpod_api as r
pod = sys.argv[1]
q = ('{ pod(input: {podId: "%s"}) { id desiredStatus lastStatusChange gpuCount machineId '
     'machine { gpuDisplayName location } '
     'runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } '
     'ports { ip isIpPublic privatePort publicPort type } container { cpuPercent memoryPercent } } } }') % pod
req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q}).encode(), method="POST",
                             headers={**r._headers(os.environ["K"]), "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as resp:
    d = json.loads(resp.read())
print(json.dumps(d.get("errors") or d["data"]["pod"], indent=1))
