"""AthenaK references in luna-reference-solutions: load one, assert its run and scheme, print its provenance."""
import json

import numpy as np

from luna.reference_data import reference_path


def load(rel, run, **expect):
    """(arrays, provenance) of athenak/<rel>; asserts run `run`'s athinput has every `key = value` of `expect`
    (first occurrence of the key)."""
    with np.load(reference_path(f"athenak/{rel}")) as z:
        data = {k: z[k] for k in z.files if k != "provenance"}
        prov = json.loads(str(z["provenance"]))
    rec = prov["runs"][run]
    lines = {}
    for ln in rec["athinput"].splitlines():                   # first occurrence: <mesh> nx1 before <meshblock> nx1
        if "=" in ln:
            lines.setdefault(ln.split("=", 1)[0].strip(), ln.split("=", 1)[1].strip())
    for k, v in expect.items():
        assert lines.get(k) == str(v), f"{rel} run {run}: {k} = {lines.get(k)!r}, expected {v!r}"
    job = rec.get("job.json", {})
    exe = next((w for w in job.get("run_line", "").split() if w.endswith("/athena")), prov["athenak_binary"])
    sha = prov.get("binaries", {}).get(exe, {}).get("sha256", prov["athenak_binary_sha256"])
    assert job.get("rc", 0) == 0, f"{rel} run {run}: AthenaK exit code {job.get('rc')}"
    print(f"PROVENANCE athenak/{rel} run {run}: AthenaK source {prov['athenak_commit'][:8]}, binary {exe} sha256 "
          f"{sha[:8]}, {lines.get('reconstruct')} + {lines.get('integrator')} + {lines.get('rsolver')}, CFL "
          f"{lines.get('cfl_number')}, SLURM {job.get('slurm_job_id')} {job.get('gpu', '')}; reduced by LUNA "
          f"{prov['luna_commit'][:8]} (tools/athenak_refs.py)")
    return data, prov
