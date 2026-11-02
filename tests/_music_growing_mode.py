"""Shared pieces of the coupled MUSIC gas+DM growing-mode gate (one loop per
test FILE: on ROCm a second coupled loop in the same process wedges the
runtime, see docs/history/AUDIT_2026-09-16.md sec.15 and always-test-on-gpu)."""
import os

import numpy as np

BOX, H, OM, OL = 25000.0, 0.6711, 0.3, 0.7
A_I, A_END, NP, AMP = 0.02, 0.1, 32, 0.02
F_GAS = 0.15                      # gas mass fraction (equal-mass particles per species)
# |A_num / A_lin - 1| per species. Measured 2026-09-17 (2 GPUs): gas 0.987,
# DM 0.984. The slab twin this replaced sat at 0.944 / 0.942 -- its force was
# ~6% low on this mode -- and needed its own looser bound; it is gone.
TOL_LIN = 0.03
SCRATCH = os.environ.get("LUNA_TEST_SCRATCH", "snapshots/test_music_growing_mode")


def growing_mode_ic(path, gas=True, bulk_velocity=None):
    """Write the growing-mode MUSIC/Gadget IC to `path`; return (q, k).

    One cos mode along x (delta_i = AMP) at A_I. gas: gas + DM with F_GAS of the mass in gas,
    the gas half a cell off the DM so the two deposits do not coincide; else DM only.
    bulk_velocity: a uniform Gadget velocity (km/s, 3) added to every particle."""
    import h5py
    from luna.cosmology import Cosmology
    cosmo = Cosmology(Omega_m=OM, Omega_L=OL, h=H)
    q = (np.arange(NP) + 0.5) * (BOX / NP)
    QX, QY, QZ = np.meshgrid(q, q, q, indexing="ij")
    k = 2.0 * np.pi / BOX
    psi = AMP * np.sin(k * QX) / k                      # kpc/h, delta_i = AMP
    Hk = 0.1 * cosmo.E(A_I)                             # km/s per kpc/h
    u = np.sqrt(A_I) * Hk * cosmo.growth_rate(A_I) * psi   # Gadget velocity
    ub = np.zeros(3) if bulk_velocity is None else np.asarray(bulk_velocity)
    pos = np.stack([(QX + psi) % BOX, QY, QZ], -1).reshape(-1, 3)
    vel = np.stack([u + ub[0], 0 * u + ub[1], 0 * u + ub[2]], -1).reshape(-1, 3)
    n = pos.shape[0]
    rho_crit = 2.775e-8                                  # 1e10 Msun/h per (kpc/h)^3
    m_tot = OM * rho_crit * BOX ** 3
    f_gas = F_GAS if gas else 0.0
    with h5py.File(path, "w") as f:
        Hd = f.create_group("Header")
        Hd.attrs["BoxSize"] = BOX
        Hd.attrs["NumPart_ThisFile"] = np.array([n if gas else 0, n, 0, 0, 0, 0], np.int64)
        Hd.attrs["NumPart_Total"] = np.array([n if gas else 0, n, 0, 0, 0, 0], np.uint32)
        Hd.attrs["MassTable"] = np.array([f_gas * m_tot / n, (1 - f_gas) * m_tot / n, 0, 0, 0, 0])
        Hd.attrs["Time"] = A_I
        Hd.attrs["Redshift"] = 1.0 / A_I - 1.0
        Hd.attrs["Omega0"] = OM
        Hd.attrs["OmegaLambda"] = OL
        Hd.attrs["HubbleParam"] = H
        Hd.attrs["NumFilesPerSnapshot"] = 1
        for pt in (("PartType0", "PartType1") if gas else ("PartType1",)):
            g = f.create_group(pt)
            off = 0.5 * BOX / NP if pt == "PartType0" else 0.0
            g.create_dataset("Coordinates", data=((pos + off) % BOX if off else pos).astype(np.float64))
            g.create_dataset("Velocities", data=vel.astype(np.float64))
        if gas:
            f["PartType0"].create_dataset("InternalEnergy", data=np.full(n, 1.0e-3))  # cold
    return q, k


def mode_amplitude(delta_x, k):
    """2 <delta cos(kx)> over cell centres: the amplitude of the cos mode."""
    x = (np.arange(NP) + 0.5) * (BOX / NP)
    return 2.0 * np.mean(delta_x * np.cos(k * x))


def run_loop(ic_path, out, **cfg_kw):
    """The coupled MUSIC run through main.run on the block layout. Returns
    (U, live DM positions in ID order, a, steps); the live IDs must be exactly
    the IC's, none lost or duplicated."""
    from dataclasses import replace

    from luna import main
    from luna.template_config import Config
    cfg = replace(Config(), run="gas+dm", CosmologicalIntegration=True, SPATIAL_DIMENSION=3,
                  music_ic_path=ic_path, Nx=NP, Ny=NP, Nz=NP, a_end=A_END,
                  dloga_max=0.02, output_dir=out,
                  PLOT_AFTER_SIM=False, hydro_fused=True, decomposition="blocks", **cfg_kw)
    res = main.run(cfg)
    U = np.asarray(res.extra["U"])
    act = np.asarray(res.extra["dm_active"]) > 0
    pos = np.asarray(res.extra["dm_pos"])[act]
    ids = np.asarray(res.extra["dm_ids"])[act]
    assert np.array_equal(np.sort(ids), np.arange(NP ** 3)), \
        f"DM IDs: {ids.size} live, {np.unique(ids).size} distinct, of {NP ** 3}"
    return U, pos[np.argsort(ids)], float(res.t), int(res.n_steps)


def check_growing_mode(tmp_path, loop="blocks", **cfg_kw):
    """Run one loop on the synthetic IC and gate gas + DM k-modes against
    linear growth; save the gas density for the cross-loop comparison.
    Returns (U, DM positions in ID order)."""
    from luna.cosmology import Cosmology
    ic = str(tmp_path / "ic.hdf5")
    _, k = growing_mode_ic(ic)
    U, pos, a_f, step = run_loop(ic, str(tmp_path / f"out_{loop}"), **cfg_kw)
    assert step > 20
    cosmo = Cosmology(Omega_m=OM, Omega_L=OL, h=H)
    growth = cosmo.growth_factor(a_f) / cosmo.growth_factor(A_I)
    a_lin = -AMP * growth                                   # delta = -d psi/dq
    rho = U[0]
    dgas = rho.mean(axis=(1, 2)) / rho.mean() - 1.0
    r_gas = mode_amplitude(dgas, k) / a_lin
    # DM: k-mode of the CIC-deposited x-density (a histogram cannot see a
    # lattice displaced by less than half a bin)
    f = np.mod(pos[:, 0], BOX) / (BOX / NP) - 0.5
    i0 = np.floor(f).astype(int)
    w1 = f - i0
    nx = np.zeros(NP)
    np.add.at(nx, np.mod(i0, NP), 1.0 - w1)
    np.add.at(nx, np.mod(i0 + 1, NP), w1)
    ddm = nx / nx.mean() - 1.0
    r_dm = mode_amplitude(ddm, k) / a_lin
    print(f"[{loop}] a_f={a_f:.4f} steps={step} gas mode/linear={r_gas:.4f} "
          f"DM mode/linear={r_dm:.4f} n_dm={pos.shape[0]}")
    assert pos.shape[0] == NP ** 3                          # every DM particle kept
    assert abs(r_gas - 1.0) < TOL_LIN, f"{loop}: gas mode {r_gas:.4f} x linear"
    assert abs(r_dm - 1.0) < TOL_LIN, f"{loop}: DM mode {r_dm:.4f} x linear"
    _check_final_snapshot(str(tmp_path / f"out_{loop}"), loop, a_f, rho)
    os.makedirs(SCRATCH, exist_ok=True)
    np.save(os.path.join(SCRATCH, f"rho_{loop}.npy"), rho)
    with open(os.path.join(SCRATCH, f"a_{loop}.txt"), "w") as fh:
        fh.write(repr(a_f))
    return U, pos


def _check_final_snapshot(out, loop, a_f, rho):
    """Every coupled loop leaves a final snapshot holding its own gas density
    and a DM deposit that carries the whole DM mass.

    The blocks loop wrote no snapshot at all until file_io/cosmo_snapshot.py:
    a 512^3 production run produced checkpoints and nothing to look at.
    """
    import glob

    import h5py
    snaps = sorted(glob.glob(os.path.join(out, "coupled_a*.hdf5")))
    assert snaps, f"{loop}: no coupled snapshot in {out}"
    with h5py.File(snaps[-1], "r") as f:
        assert abs(float(f.attrs["a"]) - a_f) < 1e-9, f"{loop}: snapshot a mismatch"
        assert int(f.attrs["Nx"]) == NP
        rho_gas = f["rho_gas"][...]
        rho_dm = f["rho_dm"][...]
    assert rho_gas.shape == rho.shape, f"{loop}: {rho_gas.shape} vs {rho.shape}"
    assert np.allclose(rho_gas, rho.astype(np.float32), rtol=1e-6), \
        f"{loop}: snapshot gas is not the run's gas"
    # the deposit must carry every particle's mass (rho_dm is a density)
    m_dm = float(rho_dm.sum()) * (BOX / NP) ** 3
    m_expect = (1.0 - F_GAS) * OM * 2.775e-8 * BOX ** 3
    assert abs(m_dm / m_expect - 1.0) < 1e-5, \
        f"{loop}: snapshot DM mass {m_dm:.6e} vs {m_expect:.6e}"
