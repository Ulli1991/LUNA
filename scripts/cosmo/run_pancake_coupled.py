"""Coupled gas + DM Zel'dovich pancake -- validation of the joint cosmological
hydro+N-body runner (setup/runner_cosmo_hydro.py).

Gas comes from the registered Zel'dovich IC; the dark matter is a grid of
particles displaced by the SAME Zel'dovich map and given the matching
super-comoving velocity. In the pre-caustic regime both pressureless-ish species
track the exact analytic solution, so a coupled run that reproduces the analytic
density for BOTH gas and DM validates the shared-potential coupling.

  JAX_PLATFORM_NAME=cpu python scripts/cosmo/run_pancake_coupled.py
"""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, os.getcwd())

from cosmology import Cosmology
from scripts.cosmo.run_pancake import analytic, build_cfg
from setup.runner_cosmo_hydro import run_simulation_cosmo_hydro_registered_ic


def dm_zeldovich_ic(cfg):
    """A grid of DM particles displaced by the Zel'dovich map at a_init, with the
    matching super-comoving velocity. Per-particle mass dx^3 -> mean rho~_dm = 1."""
    Nx, box = cfg.Nx, cfg.Lx
    dx = box / Nx
    k = 2 * np.pi / box
    a_i, a_c, H0 = cfg.a_init, cfg.zel_a_caustic, cfg.cosmo_H0
    A = a_i / a_c
    cosmo = Cosmology(Omega_m=cfg.cosmo_Omega_m, Omega_L=cfg.cosmo_Omega_L)
    q1 = (np.arange(Nx) + 0.5) * dx
    qx, qy, qz = np.meshgrid(q1, q1, q1, indexing="ij")
    x = np.mod(qx - A * np.sin(k * qx) / k, box)       # Zel'dovich displacement (x only)
    vx = -(cosmo.dadtau(a_i, H0) / a_c) * np.sin(k * qx) / k   # v~ = a v_pec
    pos = np.stack([x, qy, qz], axis=-1).reshape(-1, 3)
    p = np.stack([vx, np.zeros_like(vx), np.zeros_like(vx)], axis=-1).reshape(-1, 3)
    mass = np.full(pos.shape[0], dx ** 3)
    return pos, p, mass


def main():
    from dataclasses import replace
    cfg = build_cfg()
    cfg = replace(cfg, DualEnergy=os.environ.get("LUNA_DUAL", "0") == "1")
    if cfg.DualEnergy:
        print("# DualEnergy = True")
    dm_pos, dm_p, dm_mass = dm_zeldovich_ic(cfg)
    print(f"# coupled pancake: gas {cfg.Nx}^3 grid + {dm_pos.shape[0]} DM particles")

    U, dm_pos, dm_p, a, step = run_simulation_cosmo_hydro_registered_ic(
        cfg, dm_pos, dm_p, dm_mass, is_main_process=True)

    # --- compare BOTH species to the analytic Zel'dovich solution at z=0 ---
    U = np.asarray(U)
    x, rho_ana, v_ana = analytic(a, cfg)
    rho_gas = U[0, :, 0, 0]                              # gas: uniform in y,z
    l1_gas = np.mean(np.abs(rho_gas - rho_ana)) / np.mean(rho_ana)

    # DM: the clean metric is particle POSITION vs the Zel'dovich map (the
    # CIC-deposited density smooths the sharp peak, ~10% on one particle/cell).
    # Particle order is preserved, so particle i still carries Lagrangian q_i.
    dm_pos = np.asarray(dm_pos)
    Nx, box, dx = cfg.Nx, cfg.Lx, cfg.Lx / cfg.Nx
    k = 2 * np.pi / box
    A_end = a / cfg.zel_a_caustic
    q1 = (np.arange(Nx) + 0.5) * dx
    qx = np.meshgrid(q1, q1, q1, indexing="ij")[0].reshape(-1)
    x_ana = np.mod(qx - A_end * np.sin(k * qx) / k, box)
    dpos = np.mod(dm_pos[:, 0] - x_ana + box / 2, box) - box / 2   # wrapped diff
    ampl = A_end / k                                    # Zel'dovich displacement amplitude
    l1_dm_pos = np.mean(np.abs(dpos)) / ampl
    # also the deposited-density L1 (with its CIC-smoothing caveat) for context
    from gravity.particle_mesh import ParticleMeshGravity
    import jax.numpy as jnp
    pm = ParticleMeshGravity(cfg)
    rho_dm_x = np.asarray(pm.cic_deposit(jnp.asarray(dm_pos), jnp.asarray(dm_mass),
                                         jnp.ones(dm_pos.shape[0]))).mean(axis=(1, 2))
    l1_dm_rho = np.mean(np.abs(rho_dm_x - rho_ana)) / np.mean(rho_ana)

    print(f"\n# coupled pancake at a={a:.4f} (z={1/a-1:.4f}), {step} steps")
    print(f"# L1 gas rho~       = {l1_gas:.3e}")
    print(f"# L1 DM  position   = {l1_dm_pos:.3e}  (vs Zel'dovich displacement {ampl:.3e})")
    print(f"# L1 DM  rho~ (CIC) = {l1_dm_rho:.3e}  (smoothed peak, context only)")
    ok = l1_gas < 1e-2 and l1_dm_pos < 2e-2
    print(f"# COUPLED-PANCAKE {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
