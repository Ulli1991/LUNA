"""Cosmological (super-comoving) integration test.

The Zel'dovich pancake, run end-to-end through main.run with the FMG multigrid
self-gravity, must track the exact analytic Zel'dovich solution. This validates
the comoving time integration (super-comoving variables + tau-stepping, the
RAMSES 'friedman' a(tau) table) and the Poisson coefficient swap
4 pi G -> (3/2) Omega_m a. Kept small (32^3 EdS, z=2.33 -> z~0.67) so it runs in
a few seconds on a CPU.
"""

import pytest
pytestmark = pytest.mark.slow

import numpy as np


def _zeldovich(a, N, a_c, H0=1.0):
    """Exact super-comoving (rho~, v~) for the pancake at scale factor a."""
    from cosmology import Cosmology
    cosmo = Cosmology(Omega_m=1.0, Omega_L=0.0)
    k = 2.0 * np.pi
    x = (np.arange(N) + 0.5) / N
    A = a / a_c
    q = x.copy()
    for _ in range(80):
        q -= (q - A * np.sin(k * q) / k - x) / (1.0 - A * np.cos(k * q))
    rho = 1.0 / (1.0 - A * np.cos(k * q))
    v = -(cosmo.dadtau(a, H0) / a_c) * np.sin(k * q) / k
    return rho, v


def test_zeldovich_pancake_through_main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from dataclasses import replace

    import main
    from cosmology import Cosmology
    from template_config import Config

    N, a_i, a_c, a_end, cs_frac = 32, 0.3, 1.5, 0.6, 1.0e-2
    k = 2.0 * np.pi
    cs0 = cs_frac * Cosmology(Omega_m=1.0, Omega_L=0.0).dadtau(a_i, 1.0) / (a_c * k)
    cfg = replace(
        Config(),
        test_problem="Zeldovich-Pancake", SPATIAL_DIMENSION=3,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0,
        gamma=5.0 / 3.0, BOUNDARY=0, ORDER=1, LIMITER=1, CFL=0.4,
        riemann_solver="hllc", SELF_GRAVITY=False,
        CosmologicalIntegration=True,
        cosmo_Omega_m=1.0, cosmo_Omega_L=0.0, cosmo_H0=1.0,
        a_init=a_i, a_end=a_end, zel_a_caustic=a_c, dloga_max=0.02,
        zel_cs_frac=cs_frac, cosmo_p_floor=cs0 ** 2 / (5.0 / 3.0),
        production_mode=True, PLOT_AFTER_SIM=False, clear_output_dir=True,
        output_dir="/tmp/luna_test_zeldovich",
    )

    U, a, step = main.run(cfg)
    U = np.asarray(U)
    assert np.isfinite(U).all(), "non-finite state"

    rho_num = U[0, :, 0, 0]                 # uniform in y,z -> any x-line
    v_num = U[1, :, 0, 0] / U[0, :, 0, 0]
    rho_ana, v_ana = _zeldovich(a, N, a_c)
    l1_rho = np.mean(np.abs(rho_num - rho_ana)) / np.mean(rho_ana)
    l1_v = np.mean(np.abs(v_num - v_ana)) / np.max(np.abs(v_ana))
    assert l1_rho < 2.0e-2, f"density L1 too large: {l1_rho:.2e}"
    assert l1_v < 2.0e-2, f"velocity L1 too large: {l1_v:.2e}"
