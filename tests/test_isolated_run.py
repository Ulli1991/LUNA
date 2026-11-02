"""A self-gravitating cloud with isolated gravity and diode outflow, through
assemble/evolve: two shards step exactly as one device (gravity solve,
face gradient, gas kick, diode hydro), and with sinks on it runs on both.

Each layout is its own process: the device count is fixed at JAX start.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r"""
import json, sys
import numpy as np
import jax
from dataclasses import replace
from tests.test_wall_m3_ic import _m3
from amuse_luna.stepper import build
mesh, sinks, out = json.loads(sys.argv[1]), bool(int(sys.argv[2])), sys.argv[3]
bnd = int(sys.argv[4])
cfg = replace(_m3(N=32, turb_v0_kms=3.0), MESH_SHAPE=tuple(mesh), SELF_GRAVITY=True, POISSON_BOUNDARY=3,
              BOUNDARY=bnd, POISSON_N_LEVELS=3, ORDER=1, LIMITER=0, enable_x64=True,
              sinks_enabled=sinks, sink_max_count=4,
              sink_creation_density=5.0e3 if sinks else 0.0, particle_boundary="outflow",
              write_snapshots=False, t_end=1.0)
st = build(cfg)
m = [float(np.sum(np.asarray(st.U[0])))]
for _ in range(6):
    st.step_once(2.0e-3)
    m.append(float(np.sum(np.asarray(st.U[0]))))
np.savez(out, U=np.asarray(st.U), m=np.array(m))
"""


def _run(tmp_path, mesh, sinks=False, boundary=4):
    out = str(tmp_path / f"U_{'x'.join(map(str, mesh))}_{int(sinks)}_{boundary}.npz")
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=f"--xla_force_host_platform_device_count={int(np.prod(mesh))}",
               PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), ROOT,
                                           os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(mesh), str(int(sinks)), out,
                        str(boundary)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return np.load(out)


def test_two_shards_step_as_one_device(tmp_path):
    one = _run(tmp_path, [1, 1, 1])
    two = _run(tmp_path, [2, 1, 1])
    U1, U2 = one["U"], two["U"]
    d = np.max(np.abs(U2 - U1) / np.abs(U1).max(axis=(1, 2, 3), keepdims=True))
    print(f"isolated+diode run, 2 shards vs 1: {d:.2e}")
    assert d < 1e-10
    # The diode clamps the ghost's inflow velocity, not the face flux: gas that
    # gravity pulls away from a face opens a rarefaction that draws some mass
    # in -- less than the zero-gradient face (Torch's FLASH "outflow", which
    # feeds the inflow at the interior's own speed).
    zg = _run(tmp_path, [1, 1, 1], boundary=1)
    gain4 = one["m"][-1] - one["m"][0]
    gain1 = zg["m"][-1] - zg["m"][0]
    print(f"mass gained in 6 steps: diode {gain4 / one['m'][0]:.2e}, "
          f"zero-gradient {gain1 / zg['m'][0]:.2e} of the box")
    assert gain4 < 0.75 * gain1        # measured 0.50; without the clamp 1.0


def test_sinks_run_with_isolated_gravity_on_two_shards(tmp_path):
    """The sinks get a box they know (non-periodic, edge halo) under
    POISSON_BOUNDARY=3 -- ShardedSinks' halo refused code 3."""
    two = _run(tmp_path, [2, 1, 1], sinks=True)
    assert np.all(np.isfinite(two["U"]))
