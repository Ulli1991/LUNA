"""Every output header carries the full configuration and the exact code revision that wrote it."""
import glob
import json
import os
import re
import subprocess

import h5py

from tests._helpers import _build_cfg_for_test

REV = re.compile(r"^([0-9a-f]{40}(-dirty)?|unknown)$")


def _cfg(tmp):
    return _build_cfg_for_test(test_problem="Sod-Shock-3D-AMR", SPATIAL_DIMENSION=3, BOUNDARY=1,
                               Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4, ORDER=1,
                               LIMITER=0, enable_x64=True, output_dir=str(tmp))


def _state(cfg):
    from luna.setup.ics import gas_grid
    from luna.setup.mesh import create_distributed_mesh
    mesh, _ = create_distributed_mesh(cfg)
    U, _, _ = gas_grid(cfg, mesh)
    return mesh, U


def test_code_revision_is_the_checked_out_commit():
    from luna.file_io.checkpoint_utils import _REPO_ROOT, code_revision
    rev = code_revision()
    assert REV.match(rev), rev
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True,
                          text=True).stdout.strip()
    if head:
        assert rev.split("-")[0] == head
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=_REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        assert rev.endswith("-dirty") == bool(dirty)


def test_revision_file_is_used_without_git(tmp_path, monkeypatch):
    """A deployed copy without .git (the JUPITER node-local copy) reads CODE_REVISION."""
    import luna.file_io.checkpoint_utils as cu
    fake_pkg = tmp_path / "src" / "luna"
    fake_pkg.mkdir(parents=True)
    (tmp_path / "CODE_REVISION").write_text("0123456789abcdef0123456789abcdef01234567-dirty\n")
    monkeypatch.setattr(cu, "_REPO_ROOT", str(fake_pkg))
    cu.code_revision.cache_clear()
    try:
        assert cu.code_revision() == "0123456789abcdef0123456789abcdef01234567-dirty"
    finally:
        cu.code_revision.cache_clear()


def test_installed_wheel_gives_its_version_and_never_asks_git(tmp_path, monkeypatch):
    """A wheel (no pyproject.toml, no CODE_REVISION) reports its package version, not an enclosing repo's HEAD."""
    import importlib.metadata
    import luna.file_io.checkpoint_utils as cu
    fake_pkg = tmp_path / "site-packages" / "luna"
    fake_pkg.mkdir(parents=True)
    monkeypatch.setattr(cu, "_REPO_ROOT", str(fake_pkg))

    def no_git(*a, **k):
        raise AssertionError("git asked outside a source checkout")
    monkeypatch.setattr(cu.subprocess, "run", no_git)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    cu.code_revision.cache_clear()
    try:
        assert cu.code_revision() == "luna-hydro 9.9.9"
    finally:
        cu.code_revision.cache_clear()


def test_checkpoint_and_snapshot_headers_carry_revision_and_config(tmp_path):
    from luna.file_io.checkpoint_utils import code_revision
    from luna.file_io.write_checkpoint_distributed import write_checkpoint_distributed
    from luna.file_io.write_snapshot_distributed import write_snapshot_distributed
    cfg = _cfg(tmp_path)
    mesh, U = _state(cfg)
    ck = write_checkpoint_distributed(U, cfg, t=0.1, step=3, checkpoint_dir=str(tmp_path / "ck"))
    write_snapshot_distributed(U, cfg, 0.1, 3, mesh=mesh)
    snaps = glob.glob(os.path.join(str(tmp_path), "snapdir_003", "*.hdf5"))
    assert ck and snaps
    with h5py.File(ck, "r") as f:
        assert f.attrs["code_revision"] == code_revision()
        assert json.loads(f.attrs["config_json"])["Nx"] == 16
    for fn in snaps:
        with h5py.File(fn, "r") as f:
            assert f["cells"].attrs["code_revision"] == code_revision()
            assert json.loads(f["cells"].attrs["config_json"])["Nx"] == 16
