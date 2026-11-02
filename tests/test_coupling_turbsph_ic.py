"""Turbulent-sphere IC (LUNA turbsph) — incl. the 2000 Msun cloud setup.

Checks the generated molecular-cloud IC has the right total mass, a turbulent
velocity field at the target Mach number, and is registered in the IC registry.
Pure IC generation (CPU, no MPI).
"""
import os

from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from luna.ic_generation.turbulent_sphere_3d import Params, turbulent_sphere_3d  # noqa: E402
from luna.units import UnitSystem, PARSEC_CGS, SOLAR_MASS_CGS, BOLTZMANN_CGS, PROTON_MASS_CGS  # noqa: E402


def _cloud_cfg(N=64, L=8.0, M=2000.0, R=2.0, T=15.0, mach=8.0, mu=1.27):
    cfg = _build_cfg_for_test(
        test_problem="Turbulent-Sphere-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0)
    return replace(
        cfg,
        units=UnitSystem(UnitLength_in_cm=PARSEC_CGS,
                         UnitMass_in_g=SOLAR_MASS_CGS,
                         UnitVelocity_in_cm_per_s=1.0e5),
        problem=Params(mass_msun=M, radius_pc=R, T_K=T, mach=mach),
        sink_star_mu=mu, seed=1)


def test_2000msun_cloud_mass_and_turbulence():
    cfg = _cloud_cfg(N=64, M=2000.0, R=2.0, T=15.0, mach=8.0)
    U = np.asarray(turbulent_sphere_3d(cfg))
    assert U.shape == (5, 64, 64, 64)
    assert np.all(np.isfinite(U)) and np.all(U[0] > 0)

    dx = cfg.Lx / cfg.Nx
    dV = dx ** 3
    rho = U[0]
    xc = (np.arange(64) + 0.5) * dx
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    r = np.sqrt((X - cfg.Lx / 2) ** 2 + (Y - cfg.Lx / 2) ** 2 + (Z - cfg.Lx / 2) ** 2)
    cloud = r < cfg.problem.radius_pc

    M_cloud = float(np.sum(rho[cloud]) * dV)             # code mass = M_sun
    assert M_cloud == pytest.approx(2000.0, rel=0.15)    # ~2000 Msun within the sphere

    # turbulent Mach inside the cloud ≈ target
    vx, vy, vz = U[1] / rho, U[2] / rho, U[3] / rho
    rms = float(np.sqrt(np.mean((vx**2 + vy**2 + vz**2)[cloud])))
    cs = np.sqrt(BOLTZMANN_CGS * cfg.problem.T_K / (cfg.sink_star_mu * PROTON_MASS_CGS)) / 1e5
    mach = rms / cs
    assert 0.5 * cfg.problem.mach < mach < 1.5 * cfg.problem.mach


def test_registered_in_ic_registry():
    from luna.ic_generation.ic import initial_conditions
    cfg = _cloud_cfg(N=32)
    U = np.asarray(initial_conditions(cfg))               # dispatches via test_problem
    assert U.shape == (5, 32, 32, 32) and np.all(np.isfinite(U))


def test_density_contrast_cloud_vs_ambient():
    cfg = _cloud_cfg(N=48, M=2000.0, R=2.0)
    cfg = replace(cfg, problem=replace(cfg.problem, amb_frac=0.01))
    U = np.asarray(turbulent_sphere_3d(cfg))
    rho = U[0]
    # centre is dense cloud; a far corner is the low ambient
    assert rho[24, 24, 24] > 50.0 * rho[0, 0, 0]
