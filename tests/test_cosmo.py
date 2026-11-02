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
    """Exact super-comoving (rho~, v~) for the pancake at scale factor a on N cells of [0, 1]."""
    from luna.cosmology import Cosmology
    from luna.ic_generation.zeldovich import exact
    x = (np.arange(N) + 0.5) / N
    rho, v = exact(a, x, 1.0, a_c, Cosmology(Omega_m=1.0, Omega_L=0.0), H0)
    return np.asarray(rho), np.asarray(v)


def test_zeldovich_pancake_through_main():
    from dataclasses import replace

    from luna import main
    from luna.cosmology import Cosmology
    from luna.ic_generation import zeldovich
    from luna.template_config import Config

    N, a_i, a_c, a_end, cs_frac = 32, 0.3, 1.5, 0.6, 1.0e-2
    k = 2.0 * np.pi
    cs0 = cs_frac * Cosmology(Omega_m=1.0, Omega_L=0.0).dadtau(a_i, 1.0) / (a_c * k)
    cfg = replace(
        Config(),
        test_problem="Zeldovich-Pancake", SPATIAL_DIMENSION=3,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0,
        gamma=5.0 / 3.0, BOUNDARY=0, ORDER=1, LIMITER=1,
        riemann_solver="hllc", SELF_GRAVITY=False,
        CosmologicalIntegration=True,
        cosmo_Omega_m=1.0, cosmo_Omega_L=0.0, cosmo_H0=1.0,
        a_init=a_i, a_end=a_end, dloga_max=0.02, cosmo_p_floor=cs0 ** 2 / (5.0 / 3.0),
        problem=zeldovich.Params(a_caustic=a_c, cs_frac=cs_frac),
        PLOT_AFTER_SIM=False, clear_output_dir=True,
        write_snapshots=True,                    # exercise the final snapshot at a_end
        output_dir="/tmp/luna_test_zeldovich",
    )

    res = main.run(cfg)
    U, a, _step = res.extra["U"], res.t, res.n_steps
    U = np.asarray(U)
    assert np.isfinite(U).all(), "non-finite state"
    # periodic: gas mass to round-off; momentum (sum rho v~, 0 by the pancake's symmetry) to round-off of
    # sum |rho v~| (measured 1.5e-13 A100, 4.2e-13 CPU-8: the comoving gravity's warm solve). Energy has no
    # budget here: the comoving source terms and the pressure floor change it.
    from luna.ic_generation.ic import initial_conditions
    from tests._helpers import check
    U0 = np.asarray(initial_conditions(cfg))
    check("zeldovich |dM|/M", abs(U[0].sum() / U0[0].sum() - 1.0), 1e-13)
    check("zeldovich max_d |sum rho v~_d| / sum |rho v~|",
          max(abs(float(U[d].sum() - U0[d].sum())) for d in (1, 2, 3)) / float(np.abs(U[1:4]).sum()), 1e-12)

    rho_num = U[0, :, 0, 0]                 # uniform in y,z -> any x-line
    v_num = U[1, :, 0, 0] / U[0, :, 0, 0]
    rho_ana, v_ana = _zeldovich(a, N, a_c)
    l1_rho = np.mean(np.abs(rho_num - rho_ana)) / np.mean(rho_ana)
    l1_v = np.mean(np.abs(v_num - v_ana)) / np.max(np.abs(v_ana))
    assert l1_rho < 2.0e-2, f"density L1 too large: {l1_rho:.2e}"
    assert l1_v < 2.0e-2, f"velocity L1 too large: {l1_v:.2e}"

    # Audit: the gas-only comoving runner used to write NO snapshot; main.run
    # now records the final state at a_end, with the config embedded.
    import glob
    import json
    import os
    import h5py
    files = glob.glob(os.path.join(cfg.output_dir, "**", "*.hdf5"), recursive=True)
    assert files, "no final snapshot written by the comoving run"
    with h5py.File(files[0], "r") as f:
        assert f["cells"].attrs["time"] == pytest.approx(a_end, abs=1e-6)
        assert json.loads(f["cells"].attrs["config_json"])["a_end"] == pytest.approx(a_end)
    assert os.path.exists(os.path.join(cfg.output_dir, "run.json"))
