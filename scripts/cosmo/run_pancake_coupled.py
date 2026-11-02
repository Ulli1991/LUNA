"""Coupled gas + DM Zel'dovich pancake -- validation of the joint cosmological
hydro + N-body step (setup/assemble.py + setup/evolve.py).

Gas comes from the registered Zel'dovich IC; the dark matter is a grid of
particles displaced by the SAME Zel'dovich map and given the matching
super-comoving velocity. In the pre-caustic regime both pressureless-ish species
track the exact analytic solution, so a coupled run that reproduces the analytic
density for BOTH gas and DM validates the shared-potential coupling.

    python -m scripts.cosmo.run_pancake_coupled
"""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np


from luna.cosmology import Cosmology
from scripts.cosmo.run_pancake import analytic, build_cfg


def dm_zeldovich_ic(cfg):
    """A grid of DM particles displaced by the Zel'dovich map at a_init, with the
    matching super-comoving velocity. Per-particle mass dx^3 -> mean rho~_dm = 1."""
    Nx, box = cfg.Nx, cfg.Lx
    dx = box / Nx
    k = 2 * np.pi / box
    a_i, a_c, H0 = cfg.a_init, cfg.problem.a_caustic, cfg.cosmo_H0
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


def run_coupled(cfg, dm_pos, dm_p, dm_mass, devices=None, pdims=None, is_main=True):
    """A coupled run whose GAS is cfg's registered test problem and whose dark
    matter is given as arrays: the initial state is built by hand and handed
    to assemble + evolve. One device (`pdims` None) uses the local gravity
    backend; a block mesh `pdims` over `devices` the sharded one. Returns
    (U0, final State)."""
    from jax.sharding import Mesh
    from luna.setup.assemble import assemble
    from luna.setup.ics import gas_grid
    from luna.setup.mesh import create_distributed_mesh
    from luna.setup.evolve import evolve
    from luna.setup.ics import Initial
    from luna.setup.mesh import BLOCKS, DENSE
    from luna.setup.run_kind import COUPLED
    from luna.setup.state import Particles
    if pdims is None:
        if devices is not None and len(devices) == 1:
            mesh = Mesh(np.asarray(devices).reshape(1, 1, 1), ("x", "y", "z"))
        else:
            mesh, _ = create_distributed_mesh(cfg)
        layout = DENSE if mesh.size == 1 else BLOCKS
    else:
        mesh = Mesh(np.asarray(devices).reshape(*pdims), ("x", "y", "z"))
        layout = BLOCKS
    U0, _, _ = gas_grid(cfg, mesh)
    init = Initial(cfg=cfg, U=U0, dm=Particles(np.asarray(dm_pos), np.asarray(dm_p),
                                                np.asarray(dm_mass)))
    physics = assemble(cfg, mesh, COUPLED, layout, init, is_main)
    return U0, evolve(cfg, init, physics, mesh, is_main)


def main():
    cfg = build_cfg()
    dm_pos, dm_p, dm_mass = dm_zeldovich_ic(cfg)
    print(f"# coupled pancake: gas {cfg.Nx}^3 grid + {dm_pos.shape[0]} DM particles")

    _, state = run_coupled(cfg, dm_pos, dm_p, dm_mass)
    a, step = state.a, state.step
    dm = state.species["dm"]
    act = np.asarray(dm.active) > 0
    pos = np.asarray(dm.pos)[act]

    # --- compare BOTH species to the analytic Zel'dovich solution ---
    U = np.asarray(state.U)
    x, rho_ana, v_ana = analytic(a, cfg)
    rho_gas = U[0, :, 0, 0]                              # gas: uniform in y,z
    l1_gas = np.mean(np.abs(rho_gas - rho_ana)) / np.mean(rho_ana)

    # DM: sorted x against the sorted analytic map (order-free: the block
    # exchange and compaction reorder particles).
    Nx, box, dx = cfg.Nx, cfg.Lx, cfg.Lx / cfg.Nx
    k = 2 * np.pi / box
    A_end = a / cfg.problem.a_caustic
    q1 = (np.arange(Nx) + 0.5) * dx
    x_ana = np.sort(np.tile(np.mod(q1 - A_end * np.sin(k * q1) / k, box), Nx * Nx))
    ampl = A_end / k
    l1_dm_pos = np.mean(np.abs(np.sort(pos[:, 0]) - x_ana)) / ampl

    print(f"\n# coupled pancake at a={a:.4f} (z={1/a-1:.4f}), {step} steps")
    print(f"# L1 gas rho~       = {l1_gas:.3e}")
    print(f"# L1 DM  position   = {l1_dm_pos:.3e}  (vs Zel'dovich displacement {ampl:.3e})")
    ok = l1_gas < 1e-2 and l1_dm_pos < 2e-2 and act.sum() == dm_pos.shape[0]
    print(f"# COUPLED-PANCAKE {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
