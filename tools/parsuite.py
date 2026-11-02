"""The full test suite in parallel so it fits the 15-minute apudev partition.

    python parsuite.py ROOT REPORT [workers]

Each file runs as tools/run_tests.py runs it (one pytest process, CPU backend,
8 virtual devices), longest first from ROOT/tests/tiers.json, each worker
pinned to its own share of the node's cores.
"""
import json
import os
import subprocess
import sys
import threading
import time
from queue import Queue

root, report = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
TIMEOUT = 840
os.chdir(root)
cost = json.load(open("tests/tiers.json")).get("seconds", {})
if os.path.exists("tests/.costs.local.json"):
    cost.update(json.load(open("tests/.costs.local.json")).get("seconds", {}))
files = sorted((f"tests/{f}" for f in os.listdir("tests") if f.startswith("test_") and f.endswith(".py")),
               key=lambda f: -cost.get(f, 60.0))
cores = sorted(os.sched_getaffinity(0))
share = max(1, len(cores) // workers)
q = Queue()
# a restart pair (test_X_a.py writes what test_X_b.py reads) runs in order on one worker
paired = {f[:-5] for f in files if f.endswith("_b.py") and f[:-5] + "_a.py" in files}
for f in files:
    if f[:-5] in paired and f.endswith("_b.py"):
        continue
    q.put([f, f[:-5] + "_b.py"] if f[:-5] in paired and f.endswith("_a.py") else [f])
out, lock = [], threading.Lock()


def work(k):
    mine = ",".join(map(str, cores[k * share:(k + 1) * share]))
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=8",
               PYTHONPATH="src:.")
    while not q.empty():
        for f in q.get():
            t0 = time.time()
            cmd = ["taskset", "-c", mine, sys.executable, "-X", "faulthandler", "-m", "pytest", f, "-q",
                   "--no-header", "-p", "no:cacheprovider", "-o", f"faulthandler_timeout={TIMEOUT - 30}"]
            try:
                p = subprocess.run(cmd, env=env, timeout=TIMEOUT, capture_output=True, text=True)
                st, tail = ("pass" if p.returncode == 0 else "fail"), (p.stdout or "")[-2000:]
            except subprocess.TimeoutExpired as e:
                st, tail = "timeout", (e.stdout or b"").decode(errors="replace")[-2000:] if e.stdout else ""
            rec = dict(file=f, status=st, seconds=round(time.time() - t0, 1), tail=tail)
            with lock:
                out.append(rec)
                print(f"  {'ok  ' if st == 'pass' else st.upper():7s}{rec['seconds']:7.1f}s  {f}", flush=True)


t0 = time.time()
print(f"{len(files)} files, {workers} workers x {share} cores, root {root}", flush=True)
ts = [threading.Thread(target=work, args=(k,)) for k in range(workers)]
for t in ts:
    t.start()
for t in ts:
    t.join()
n = {s: sum(r["status"] == s for r in out) for s in ("pass", "fail", "timeout")}
print(f"{n} in {time.time() - t0:.0f}s -> {report}")
json.dump(dict(root=root, counts=n, files=sorted(out, key=lambda r: r["file"])), open(report, "w"), indent=1)
for r in out:
    if r["status"] != "pass":
        print(f"--- {r['file']} ({r['status']}) ---\n{r['tail'][-1200:]}")
