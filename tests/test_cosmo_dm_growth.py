"""Linear-growth gate for the MUSIC-IC DM runner in Gadget/Arepo units.

A single growing-mode Zel'dovich x-mode (positions + Gadget velocities
u = sqrt(a) dx/dt) in a 25 Mpc/h LCDM box is evolved a = 0.02 -> 0.1 through
main.run; the particle displacement must grow by D(a_end)/D(a_init).

This pins the unit system that the Zel'dovich pancake test (H0 = 1) cannot:
with H0 = 0.1*h instead of 0.1 the initial momenta are 1/h too large and the
growing mode overshoots by ~20%.
"""
import glob
import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow

BOX, H, OM, OL = 25000.0, 0.6711, 0.3, 0.7
A_I, A_END, NP, AMP = 0.02, 0.1, 32, 0.02


def _growing_mode_ic(path):
    import h5py
    from cosmology import Cosmology
    cosmo = Cosmology(Omega_m=OM, Omega_L=OL, h=H)
    q = (np.arange(NP) + 0.5) * (BOX / NP)
    QX, QY, QZ = np.meshgrid(q, q, q, indexing="ij")
    k = 2.0 * np.pi / BOX
    psi = AMP * np.sin(k * QX) / k                      # kpc/h, delta_i = AMP
    Hk = 0.1 * cosmo.E(A_I)                             # km/s per kpc/h
    u = np.sqrt(A_I) * Hk * cosmo.growth_rate(A_I) * psi   # Gadget velocity
    pos = np.stack([(QX + psi) % BOX, QY, QZ], -1).reshape(-1, 3)
    vel = np.stack([u, 0 * u, 0 * u], -1).reshape(-1, 3)
    n = pos.shape[0]
    rho_crit = 2.775e-8                                  # 1e10 Msun/h per (kpc/h)^3
    with h5py.File(path, "w") as f:
        Hd = f.create_group("Header")
        Hd.attrs["BoxSize"] = BOX
        Hd.attrs["NumPart_ThisFile"] = np.array([0, n, 0, 0, 0, 0], np.int64)
        Hd.attrs["NumPart_Total"] = np.array([0, n, 0, 0, 0, 0], np.uint32)
        Hd.attrs["MassTable"] = np.array([0, OM * rho_crit * BOX ** 3 / n, 0, 0, 0, 0])
        Hd.attrs["Time"] = A_I
        Hd.attrs["Redshift"] = 1.0 / A_I - 1.0
        Hd.attrs["Omega0"] = OM
        Hd.attrs["OmegaLambda"] = OL
        Hd.attrs["HubbleParam"] = H
        Hd.attrs["NumFilesPerSnapshot"] = 1
        p1 = f.create_group("PartType1")
        p1.create_dataset("Coordinates", data=pos.astype(np.float64))
        p1.create_dataset("Velocities", data=vel.astype(np.float64))
    return q, k


def test_music_config_h0_is_h_independent(tmp_path):
    from cosmology.music_ic import music_run_config, read_music_dm_ic
    _growing_mode_ic(str(tmp_path / "ic.hdf5"))
    cfg = music_run_config(read_music_dm_ic(str(tmp_path / "ic.hdf5")), 32)
    assert cfg.cosmo_H0 == pytest.approx(0.1)          # km/s per kpc/h, not 0.1*h
    assert cfg.Lx == BOX and cfg.a_init == A_I


def test_dm_runner_linear_growth_arepo_units(tmp_path):
    import jax
    jax.config.update("jax_enable_x64", True)
    from dataclasses import replace

    import main
    from cosmology import Cosmology
    from template_config import Config

    ic = str(tmp_path / "ic.hdf5")
    q, k = _growing_mode_ic(ic)
    out = str(tmp_path / "out")
    cfg = replace(
        Config(), CosmoDM=True, CosmologicalIntegration=True,
        music_ic_path=ic, Nx=NP, a_end=A_END, dloga_max=0.02,
        dm_write_snapshot=True, output_dir=out, clear_output_dir=True,
        production_mode=True, PLOT_AFTER_SIM=False,
    )
    main.run(cfg)

    import h5py
    snap = sorted(glob.glob(os.path.join(out, "snap_a*.hdf5")))[-1]
    with h5py.File(snap, "r") as f:
        a_f = float(f["Header"].attrs["Time"])
        pos = np.asarray(f["PartType1"]["Coordinates"], np.float64)
    QX, QY, QZ = np.meshgrid(q, q, q, indexing="ij")
    Q = np.stack([QX, QY, QZ], -1).reshape(-1, 3)
    # y, z never move (1-D mode) -> particle order must be intact
    assert np.abs(pos[:, 1:] - Q[:, 1:]).max() < 1e-3 * BOX / NP
    dx = (pos[:, 0] - Q[:, 0] + 0.5 * BOX) % BOX - 0.5 * BOX
    amp_num = 2.0 * np.mean(dx * np.sin(k * Q[:, 0]))       # psi = A sin(kq)
    cosmo = Cosmology(Omega_m=OM, Omega_L=OL, h=H)
    amp_lin = (AMP / k) * cosmo.growth_factor(a_f) / cosmo.growth_factor(A_I)
    ratio = amp_num / amp_lin
    assert abs(ratio - 1.0) < 0.03, (
        f"growing mode a={A_I}->{a_f:.4f}: displacement {ratio:.4f} x linear "
        "(H0 unit error gives ~1.2)")
