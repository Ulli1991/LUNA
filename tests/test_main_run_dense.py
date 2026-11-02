"""End-to-end: the standard dense runner through ``main.run`` on the CPU device
mesh, with periodic diagnostics on and a snapshot written and read back.

Pins the path no other test drives: run_simulation_distributed with
enable_diagnostics (the _diag_reductions collective), the snapshot writer and
the shard-stitching reader.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow



def test_dense_runner_diagnostics_and_snapshot_roundtrip(tmp_path):
    from dataclasses import replace
    from luna.template_config import Config
    from luna.setup.jax_init import initialize_jax_distributed

    out = str(tmp_path / "run")
    cfg = replace(Config(), test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3,
                  Nx=32, Ny=16, Nz=16, BOUNDARY=1, t_end=0.04,
                  max_steps=12, write_snapshots=True, time_bet_snaps=0.01,
                  enable_diagnostics=True, diag_interval=2, output_cadence=1,
                  output_dir=out, enable_x64=True)   # x64 like the rest of the suite
    initialize_jax_distributed(enable_x64=cfg.enable_x64)
    from luna import main
    res = main.run(cfg)
    # dt snaps onto every snapshot time and t_end, so the run ends by time
    # (>= 4 steps: one per landing, so the diag_interval=2 reductions ran at
    # least twice) or by max_steps.
    assert res.t == pytest.approx(0.04) or res.n_steps == 12
    assert res.n_steps >= 4

    from luna.file_io.read_snapshot import find_snapshot_at_time, read_snapshot
    snapdir = find_snapshot_at_time(out, 0.0, tol=1e-9)
    snap = read_snapshot(snapdir)
    rho = np.asarray(snap["rho"])
    # Sod-Shock-3D derives Ny/Nz (a thin tube of cubic cells) and main.run
    # resolves that before the mesh, so compare with the grid the run used.
    from luna.ic_generation.ic import resolve
    used = resolve(cfg)
    assert (used.Ny, used.Nz) == (3, 3)
    assert rho.shape == (used.Nx, used.Ny, used.Nz) and rho.shape[0] == 32
    assert np.isfinite(rho).all() and rho.min() > 0.0
    # Sod IC: left/right densities 1.0 / 0.125 along x
    assert np.isclose(rho[0].mean(), 1.0) and np.isclose(rho[-1].mean(), 0.125)

    # Provenance (audit): run.json in the output dir, and the full config
    # embedded in every snapshot file.
    import glob
    import json
    import h5py
    rj = json.load(open(os.path.join(out, "run.json")))
    assert rj["config"]["Nx"] == 32 and rj["config"]["test_problem"] == "Sod-Shock-3D"
    assert "git_revision" in rj and rj["process_count"] >= 1
    files = [f for f in glob.glob(os.path.join(out, "**", "*.hdf5"), recursive=True)]
    assert files, "no snapshot files written"
    with h5py.File(files[0], "r") as f:
        emb = json.loads(f["cells"].attrs["config_json"])
    assert emb["Nx"] == 32 and emb["t_end"] == pytest.approx(0.04)


def test_max_levels_respects_every_axis_and_evenness():
    from luna.gravity.poisson_multigrid import _max_levels
    assert _max_levels(32) == 4                  # 32,16,8,4
    assert _max_levels((32, 32, 32)) == 4
    assert _max_levels(100) == 3                 # 100,50,25 -> 25 is odd, stop
    assert _max_levels((1024, 384, 384)) == 7    # 384,192,96,48,24,12,6 (6 is the coarsest; 3 would be odd)
    assert _max_levels((64, 32, 32)) == 4        # limited by the 32 axes
