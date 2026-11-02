"""Checkpoint format v2: per-shard datasets with global offsets, assembled
on restart per device from whichever stored shards overlap -- so a 3D run
round-trips on several devices (v1 only round-tripped one device: audit
item 8), survives a change of decomposition, carries the OU forcing state,
retains checkpoints per step, and restarts end to end through main.run.
GPU test; the multi-device cases need >=2 devices."""
import glob
import os
import shutil
import tempfile

import numpy as np
import pytest

import jax

import jax.numpy as jnp


from tests._helpers import _build_cfg_for_test

pytestmark = pytest.mark.slow


def _cfg3d(**kw):
    # Sod-Shock-3D-AMR keeps the grid as configured (Sod-Shock-3D shrinks Ny/Nz to 3)
    return _build_cfg_for_test(test_problem="Sod-Shock-3D-AMR", SPATIAL_DIMENSION=3, BOUNDARY=1,
                               Nx=32, Ny=32, Nz=32, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
                               ORDER=1, LIMITER=0, enable_x64=True, **kw)


def _mesh_and_U(cfg):
    from luna.setup.ics import gas_grid
    from luna.setup.mesh import create_distributed_mesh
    mesh, _ = create_distributed_mesh(cfg)
    U, _, _ = gas_grid(cfg, mesh)
    return mesh, U


def test_v2_roundtrip_same_mesh_with_forcing_state():
    from luna.file_io.read_checkpoint import load_checkpoint_sharded, read_checkpoint
    from luna.file_io.write_checkpoint_distributed import write_checkpoint_distributed
    cfg = _cfg3d()
    mesh, U = _mesh_and_U(cfg)
    tmp = tempfile.mkdtemp(prefix="luna_ckpt_v2_")
    try:
        key = jax.random.PRNGKey(7)
        a_state = jnp.arange(12.0).reshape(3, 4)
        fn = write_checkpoint_distributed(U, cfg, t=0.3, step=7, checkpoint_dir=tmp,
                                          extra={"forcing_a": a_state, "forcing_key": key})
        assert fn is not None and os.path.exists(fn)
        U2, meta = load_checkpoint_sharded(fn, mesh, 3)
        np.testing.assert_array_equal(np.asarray(U2), np.asarray(U))
        assert meta["format_version"] == 2 and meta["step"] == 7 and meta["time"] == 0.3
        np.testing.assert_array_equal(meta["extra"]["forcing_a"], np.asarray(a_state))
        np.testing.assert_array_equal(meta["extra"]["forcing_key"], np.asarray(key))
        assert read_checkpoint(fn)["U"] is None                # v2 carries no global U
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_v2_roundtrip_onto_a_different_decomposition():
    """Write on the default (x-sharded) mesh, load onto a y-sharded mesh."""
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    from jax.sharding import Mesh
    from luna.file_io.read_checkpoint import load_checkpoint_sharded
    from luna.file_io.write_checkpoint_distributed import write_checkpoint_distributed
    cfg = _cfg3d()
    mesh, U = _mesh_and_U(cfg)
    n = jax.device_count()
    mesh_y = Mesh(np.asarray(jax.devices()).reshape(1, n, 1), ("x", "y", "z"))
    tmp = tempfile.mkdtemp(prefix="luna_ckpt_v2_")
    try:
        fn = write_checkpoint_distributed(U, cfg, t=0.1, step=3, checkpoint_dir=tmp)
        U2, _ = load_checkpoint_sharded(fn, mesh_y, 3)
        assert U2.sharding.mesh.shape["y"] == n
        np.testing.assert_array_equal(np.asarray(U2), np.asarray(U))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_v1_single_file_still_loads():
    from luna.file_io.read_checkpoint import load_checkpoint_sharded
    from luna.file_io.write_checkpoint import write_checkpoint
    cfg = _cfg3d()
    mesh, U = _mesh_and_U(cfg)
    tmp = tempfile.mkdtemp(prefix="luna_ckpt_v1_")
    try:
        fn = write_checkpoint(np.asarray(U), cfg, t=0.2, step=5, checkpoint_dir=tmp)
        U2, meta = load_checkpoint_sharded(fn, mesh, 3)
        assert meta["format_version"] == 1
        np.testing.assert_array_equal(np.asarray(U2), np.asarray(U))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manage_checkpoints_keeps_newest_steps_not_files():
    from luna.file_io.checkpoint_utils import manage_checkpoints
    from luna.file_io.read_checkpoint import find_latest_checkpoint
    tmp = tempfile.mkdtemp(prefix="luna_ckpt_mgr_")
    try:
        for st in (10, 20, 30):
            for pid in (0, 1):
                open(os.path.join(tmp, f"checkpoint_{st:06d}.{pid}.hdf5"), "w").close()
        manage_checkpoints(tmp, 2)
        left = sorted(os.path.basename(f) for f in glob.glob(os.path.join(tmp, "*.hdf5")))
        assert left == ["checkpoint_000020.0.hdf5", "checkpoint_000020.1.hdf5",
                        "checkpoint_000030.0.hdf5", "checkpoint_000030.1.hdf5"]
        assert os.path.basename(find_latest_checkpoint(tmp)).startswith("checkpoint_000030")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_restart_through_main_run_matches_uninterrupted():
    """3D Sod on every visible device: run to t=0.05 checkpointing every step,
    restart from the checkpoint to t=0.1, compare with an uninterrupted run
    to t=0.1 (rtol 1e-2: dt clamping at the split differs, as in the 1D
    restart test)."""
    from luna import main
    tmp = tempfile.mkdtemp(prefix="luna_ckpt_e2e_")
    try:
        common = dict(output_dir=os.path.join(tmp, "out"), checkpoint_dir=os.path.join(tmp, "ck"),
                      enable_checkpointing=True, checkpoint_wall_time_interval=1e-9,
                      max_checkpoints=1, write_snapshots=False, PLOT_AFTER_SIM=False)
        cfg_a = _cfg3d(t_end=0.05, **common)   # common's enable_checkpointing wins
        main.run(cfg_a)
        cks = glob.glob(os.path.join(tmp, "ck", "checkpoint_*.hdf5"))
        assert cks, "no checkpoint written"
        cfg_b = _cfg3d(t_end=0.1, restart=True, **common)
        res_b = main.run(cfg_b)
        res_c = main.run(_cfg3d(t_end=0.1, output_dir=os.path.join(tmp, "out_c"),
                                checkpoint_dir=os.path.join(tmp, "ck_c"), write_snapshots=False,
                                PLOT_AFTER_SIM=False))
        U_b, U_c = np.asarray(res_b.extra["U"]), np.asarray(res_c.extra["U"])
        assert np.all(np.isfinite(U_b))
        # dt clamping at the split differs from the uninterrupted run: measured max
        # |dU| = 1.7e-4 on unit-scale fields (2026-09-17); a wrong or stale restart
        # state differs at O(1). Near-zero transverse momenta need the atol.
        np.testing.assert_allclose(U_b, U_c, rtol=1e-2, atol=1e-3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
