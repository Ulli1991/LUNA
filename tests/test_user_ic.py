"""A user's initial state through luna.launch (U0 array or ic(x, y, z, cfg) callable), 1D, 2D and 3D natively.

Exact claims, no tolerance: the same state as a registered problem runs bit for bit as that problem, whether given
as U0, as a callable or through an ic_dir file; the run on every visible device (8 on CPU-8) equals one device, and
two processes (each building only its own slab, the multi-process branch of setup/ics.py) equal one. A state of the
wrong shape or dimension, or one that clashes with the config, is refused before the run."""
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from tests._user_ic_run import DIMS, ROOT, as_user, case, registered_ic

DRIVER = os.path.join(ROOT, "tests", "_user_ic_run.py")
TIMEOUT = 600


def _launch(cfg, **kw):
    from luna import launch
    res = launch(cfg, **kw)
    return np.asarray(res.extra["U"]), res.n_steps, res.t


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{dim: (registered, U0, ic) end states} on every visible device."""
    from luna.ic_generation.ic import initial_conditions
    out = {}
    for dim in DIMS:
        reg = case(dim, str(tmp_path_factory.mktemp(f"reg{dim}")))
        U0 = np.asarray(initial_conditions(reg))
        out[dim] = (_launch(reg), _launch(as_user(reg), U0=U0), _launch(as_user(reg), ic=registered_ic(reg)))
    return out


@pytest.mark.parametrize("dim", DIMS)
def test_user_state_is_the_registered_problem(runs, dim):
    """U0 and ic give the registered problem's run bit for bit (end state, steps, time), in its own dimension."""
    reg, u0, ic = runs[dim]
    assert reg[0].shape == (dim + 2,) + reg[0].shape[1:] and reg[0].ndim == dim + 1
    assert reg[1] == 6, reg[1]
    for name, got in (("U0", u0), ("ic", ic)):
        assert got[1:] == reg[1:], f"{dim}D {name}: (steps, t) {got[1:]} vs {reg[1:]}"
        np.testing.assert_array_equal(got[0], reg[0], err_msg=f"{dim}D {name}")


def test_ic_dir_2d_is_the_same_run(runs, tmp_path):
    """A 2D state written with write_ic(cfg, dir, U0=...) runs as launch(cfg, U0=...); the file is (4, Nx, Ny)."""
    from luna.ic_generation.ic import initial_conditions
    from luna.ic_generation.ic_file import write_ic
    cfg = as_user(case(2, str(tmp_path / "run")))
    write_ic(cfg, str(tmp_path / "ic"), U0=np.asarray(initial_conditions(case(2, ""))))
    assert np.load(tmp_path / "ic" / "U.npy").shape == (4, 32, 32)
    got = _launch(replace(cfg, ic_dir=str(tmp_path / "ic")))
    assert got[1:] == runs[2][0][1:]
    np.testing.assert_array_equal(got[0], runs[2][0][0])


def _env(n_dev, n_proc, rank, port):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1", XLA_FLAGS=f"--xla_force_host_platform_device_count={n_dev}",
               SLURM_NTASKS=str(n_proc), SLURM_PROCID=str(rank), JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{port}")
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def _driver(n_proc, out):
    """{dim: end state} of the driver on n_proc CPU processes of one device each."""
    from tests.test_sinks_multiprocess import _free_port
    port = _free_port()
    procs = [subprocess.Popen([sys.executable, DRIVER, f"out={out}"], cwd=ROOT, env=_env(1, n_proc, r, port),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for r in range(n_proc)]
    logs = []
    for p in procs:
        try:
            logs.append(p.communicate(timeout=TIMEOUT)[0])
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            pytest.fail(f"{n_proc}-process user-IC run hung (> {TIMEOUT} s)")
    tails = "\n".join(f"--- rank {r} (rc {p.returncode}) ---\n{o[-3000:]}" for r, (p, o) in enumerate(zip(procs, logs)))
    assert all(p.returncode == 0 for p in procs), tails
    assert sum(ln.startswith("RESULT") for ln in logs[0].splitlines()) == len(DIMS), tails
    return {d: np.load(os.path.join(out, f"U{d}.npy")) for d in DIMS}


def test_sharded_equals_one_device(runs, tmp_path):
    """ic on one device == on 2 processes (each builds its own slab) == in this process on every device (CPU)."""
    import jax
    one = _driver(1, str(tmp_path / "one"))
    two = _driver(2, str(tmp_path / "two"))
    for d in DIMS:
        np.testing.assert_array_equal(two[d], one[d], err_msg=f"{d}D: 2 processes vs 1")
        if jax.default_backend() == "cpu":       # here on len(jax.devices()) devices (8 under CPU-8)
            np.testing.assert_array_equal(runs[d][2][0], one[d], err_msg=f"{d}D: {len(jax.devices())} devices vs 1")
    print(f"MEASURED user-ic sharded: 1 == 2 processes, 1 == {len(jax.devices())} devices ({jax.default_backend()}), "
          "bitwise")


def _refused(match, cfg, **kw):
    from luna import launch
    with pytest.raises(ValueError, match=match):
        launch(cfg, **kw)


def test_wrong_shape_or_dimension_is_refused(tmp_path):
    """The 3D-with-one-cell array for a 2D grid, one cell too many, a 2D state for a 3D grid: refused, naming the shape."""
    c1, c2, c3 = (as_user(case(d, str(tmp_path / f"r{d}"))) for d in DIMS)
    _refused(r"U0 has shape \(5, 32, 32, 1\); SPATIAL_DIMENSION=2, Nx=32, Ny=32 needs \(4, 32, 32\)", c2,
             U0=np.ones((5, 32, 32, 1)))
    _refused(r"needs \(3, 64\)", c1, U0=np.ones((3, 65)))
    _refused(r"needs \(5, 16, 16, 16\)", c3, U0=np.ones((4, 16, 16)))
    _refused(r"ic\(x, y, z, cfg\) returned shape \(4, 32, 32, 1\)", c2, ic=lambda x, y, z, cfg: np.ones((4, *x.shape, 1)))
    assert not os.path.exists(tmp_path / "r1") and not os.path.exists(tmp_path / "r3"), "a refused run wrote output"


def test_clashes_are_refused(tmp_path):
    """Both U0 and ic, U0 with ic_dir, with a restart, a non-callable ic, a NaN, an ic_dir file of another dimension."""
    from luna.ic_generation.ic_file import write_ic
    c2 = as_user(case(2, str(tmp_path / "r")))
    U = np.ones((4, 32, 32))
    _refused("give U0 or ic, not both", c2, U0=U, ic=lambda x, y, z, cfg: U)
    _refused("cannot run with ic_dir=", replace(c2, ic_dir=str(tmp_path)), U0=U)
    _refused("cannot run with restart=True", replace(c2, restart=True), U0=U)
    _refused("is not callable", c2, ic=U)
    bad = U.copy()
    bad[0, 3, 4] = np.nan
    _refused("1 non-finite values", c2, U0=bad)
    c3 = replace(c2, SPATIAL_DIMENSION=3, Nz=1)          # the same fingerprint, a 3D file
    write_ic(c3, str(tmp_path / "ic3"), U0=np.ones((5, 32, 32, 1)))
    _refused(r"U.npy has shape \(5, 32, 32, 1\); SPATIAL_DIMENSION=2 on this grid needs \(4, 32, 32\)",
             replace(c2, Nz=1, ic_dir=str(tmp_path / "ic3")))


def test_conserved_and_cell_centres():
    """conserved() builds (rho, rho v, E); the centres are (i + 1/2) L / N on the slice, None beyond the dimension."""
    from luna import conserved
    from luna.ic_generation.user_ic import cell_centres
    U = conserved([2.0], ([3.0], [-1.0]), [0.5], 1.5)
    np.testing.assert_array_equal(U[:, 0], [2.0, 6.0, -2.0, 0.5 / 0.5 + 0.5 * 2.0 * 10.0])
    x, y, z = cell_centres(replace(case(2, ""), Lx=2.0), x_slice=(4, 8))
    assert z is None and x.shape == y.shape == (4, 32)
    np.testing.assert_array_equal(x[:, 0], (np.arange(4, 8) + 0.5) * 2.0 / 32)
