"""Linear-growth gate for the MUSIC-IC DM runner in Gadget/Arepo units.

A single growing-mode Zel'dovich x-mode (positions + Gadget velocities
u = sqrt(a) dx/dt) in a 25 Mpc/h LCDM box is evolved a = 0.02 -> 0.1 through
main.run; the particle displacement must grow by D(a_end)/D(a_init).

This pins the unit system that the Zel'dovich pancake test (H0 = 1) cannot:
with H0 = 0.1*h instead of 0.1 the initial momenta are 1/h too large and the
growing mode overshoots by ~20%.

A uniform bulk flow rides on top (Galilean invariant: no force, the canonical
momentum is conserved), so the particles cross block faces on every axis and
the block layout exchanges and resorts them many times; the flow's free drift
is a second check of the same units. The snapshot is written block by block,
so it is matched to the IC lattice by ParticleIDs, which must be exactly the
IC's IDs: none lost, none duplicated.
"""
import glob
import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from tests._music_growing_mode import A_END, A_I, AMP, BOX, NP, OL, OM, H, growing_mode_ic  # noqa: E402
DRIFT = np.array([4.0, 2.5, 1.5]) * (BOX / NP)     # bulk-flow drift a_i -> a_end, kpc/h


def _drift_per_u(cosmo, a1):
    """Free comoving drift A_I -> a1 (kpc/h) per km/s of Gadget velocity at
    A_I: p = a^1.5 u is conserved and dx = p da / (a^3 H), H = 0.1 E(a)."""
    a = np.geomspace(A_I, a1, 20001)
    f = 1.0 / (a ** 3 * 0.1 * cosmo.E(a))
    return A_I ** 1.5 * np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(a))


def _growing_mode_ic(path):
    """The DM-only growing mode plus a bulk flow; (q, k, bulk velocity)."""
    from luna.cosmology import Cosmology
    ub = DRIFT / _drift_per_u(Cosmology(Omega_m=OM, Omega_L=OL, h=H), A_END)
    q, k = growing_mode_ic(path, gas=False, bulk_velocity=ub)
    return q, k, ub


def test_reader_ids_from_file_or_file_order(tmp_path):
    """ParticleIDs as the file has them; without them, the global file index
    (also for a chunked [lo:hi] read)."""
    import h5py
    from luna.cosmology.music_ic import read_music_dm_ic
    ic = str(tmp_path / "ic.hdf5")
    _growing_mode_ic(ic)
    n = NP ** 3
    assert np.array_equal(read_music_dm_ic(ic).ids, np.arange(n))
    assert np.array_equal(read_music_dm_ic(ic, 1, 100, 200).ids, np.arange(100, 200))
    with h5py.File(ic, "a") as f:
        f["PartType1"].create_dataset("ParticleIDs", data=np.arange(n)[::-1] + 1)
    assert np.array_equal(read_music_dm_ic(ic, 1, 100, 200).ids, n - np.arange(100, 200))


def test_music_config_h0_is_h_independent(tmp_path):
    from luna.cosmology.music_ic import music_run_config, read_music_dm_ic
    _growing_mode_ic(str(tmp_path / "ic.hdf5"))
    cfg = music_run_config(read_music_dm_ic(str(tmp_path / "ic.hdf5")), 32)
    assert cfg.cosmo_H0 == pytest.approx(0.1)          # km/s per kpc/h, not 0.1*h
    assert cfg.Lx == BOX and cfg.a_init == A_I


@pytest.mark.parametrize("decomp", ["local", "blocks"])
def test_dm_runner_linear_growth_arepo_units(tmp_path, decomp):
    """One physics gate over both gravity backends: the whole grid on one
    device (`decomposition="dense"`) and the block mesh (`decomposition="blocks"`
    on however many devices there are)."""
    from dataclasses import replace

    from luna import main
    from luna.cosmology import Cosmology
    from luna.template_config import Config

    ic = str(tmp_path / "ic.hdf5")
    q, k, ub = _growing_mode_ic(ic)
    out = str(tmp_path / "out")
    cfg = replace(
        Config(), run="dm", CosmologicalIntegration=True,
        music_ic_path=ic, Nx=NP, a_end=A_END, dloga_max=0.02,
        output_dir=out, clear_output_dir=True,
        PLOT_AFTER_SIM=False,
        **({"decomposition": "dense"} if decomp == "local"
           else {"decomposition": "blocks"}),
    )
    res = main.run(cfg)
    if decomp == "blocks":        # the flow must exercise the exchange and the resort
        assert res.extra["n_exchanges"] >= 2 * cfg.dm_resort_every, res.extra["n_exchanges"]

    import h5py
    snap = sorted(glob.glob(os.path.join(out, "snap_a*.hdf5")))[-1]
    with h5py.File(snap, "r") as f:
        a_f = float(f["Header"].attrs["Time"])
        n_total = int(f["Header"].attrs["NumPart_Total"][1])
        pos = np.asarray(f["PartType1"]["Coordinates"], np.float64)
        ids = np.asarray(f["PartType1"]["ParticleIDs"])
    # every IC particle exactly once: the IDs are a permutation of the IC's
    n = NP ** 3
    assert n_total == n and pos.shape[0] == n, f"{pos.shape[0]} particles of {n}"
    assert np.array_equal(np.sort(ids), np.arange(n)), "particle IDs lost or duplicated"
    pos = pos[np.argsort(ids)]                               # back into IC order
    QX, QY, QZ = np.meshgrid(q, q, q, indexing="ij")
    Q = np.stack([QX, QY, QZ], -1).reshape(-1, 3)
    d = (pos - Q + 0.5 * BOX) % BOX - 0.5 * BOX
    # y, z only drift with the flow (1-D mode): each ID must sit on its own
    # lattice line, shifted by the one common drift
    drift = np.median(d, axis=0)
    assert np.abs(d[:, 1:] - drift[1:]).max() < 1e-3 * BOX / NP
    cosmo = Cosmology(Omega_m=OM, Omega_L=OL, h=H)
    dx = d[:, 0] - drift[0]
    # and every ID in one lattice x-plane has that plane's displacement
    dxp = dx.reshape(NP, NP * NP)
    assert np.abs(dxp - dxp.mean(axis=1, keepdims=True)).max() < 1e-3 * BOX / NP
    amp_num = 2.0 * np.mean(dx * np.sin(k * Q[:, 0]))       # psi = A sin(kq)
    amp_lin = (AMP / k) * cosmo.growth_factor(a_f) / cosmo.growth_factor(A_I)
    ratio = amp_num / amp_lin
    print(f"[{decomp}] a={A_I}->{a_f:.4f}: displacement {ratio:.4f} x linear")
    assert abs(ratio - 1.0) < 0.03, (
        f"growing mode a={A_I}->{a_f:.4f}: displacement {ratio:.4f} x linear "
        "(H0 unit error gives ~1.2)")
    # the flow drifts freely by p * int da / (a^3 H): the same units again
    drift_lin = ub * _drift_per_u(cosmo, a_f)
    print(f"[{decomp}] n_exchanges={res.extra['n_exchanges']} bulk drift "
          f"{drift / (BOX / NP)} cells, free {drift_lin / (BOX / NP)}")
    assert np.allclose(drift, drift_lin, rtol=1e-3), (drift, drift_lin)
