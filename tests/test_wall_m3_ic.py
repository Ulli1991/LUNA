"""Torch's cloud (Wall et al. 2020 M3) as Turbulent-Sphere-3D options:
the Gaussian profile, the ambient in pressure equilibrium, the Kolmogorov
spectrum on k = 2..32 normalized to v0. docs/WALL_M3.md."""
from dataclasses import replace

import numpy as np
import pytest

from luna.ic_generation.turbulent_sphere_3d import (K_B, M_H, Params, _kolmogorov_field,
                                                    cloud_params, cloud_summary,
                                                    turbulent_sphere_3d)
from luna.units import UnitSystem
from tests._helpers import _with_problem

PC, MSUN = 3.085677581e18, 1.98892e33
UNITS = UnitSystem(UnitLength_in_cm=PC, UnitMass_in_g=MSUN, UnitVelocity_in_cm_per_s=1.0e5)
RHO_PRIME = 2.39e-23                     # Wall et al. 2020 Table 1, g/cm^3


def _m3(N=64, **kw):
    base = dict(test_problem="Turbulent-Sphere-3D", SPATIAL_DIMENSION=3, BOUNDARY=4,
                Nx=N, Ny=N, Nz=N, Lx=10.0, Ly=10.0, Lz=10.0, gamma=5.0 / 3.0, units=UNITS,
                GRAVITATIONAL_G=UNITS.GravityConstant,
                mass_msun=1000.0, radius_pc=3.0, T_K=40.0, sink_star_mu=1.3,
                profile="gaussian", edge_ratio=1.0 / 3.0, amb_T_K=60.0,
                turb_spectrum="kolmogorov", turb_v0_kms=0.616,
                turb_kmin=2, turb_kmax=32, seed=5)
    base.update(kw)
    return _with_problem(Params, **base)


def _radius(N, L=10.0):
    x = (np.arange(N) + 0.5) * L / N - L / 2
    return np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)


def test_central_density_is_table_1s():
    """M = 1e3 Msun inside R = 3 pc with rho(R) = rho(0)/3 gives Table 1's
    rho_c = 46 rho' (FWHM = R would give half that mass)."""
    p = cloud_params(_m3())
    rho_c = p["rho_c"] * MSUN / PC ** 3
    print(f"rho_c = {rho_c:.4e} g/cm^3 = {rho_c / RHO_PRIME:.2f} rho'")
    assert rho_c / RHO_PRIME == pytest.approx(46.0, rel=0.01)


def test_profile_ambient_and_velocity():
    cfg = _m3()
    p = cloud_params(cfg)
    U = np.asarray(turbulent_sphere_3d(cfg))
    r = _radius(cfg.Nx)
    cl = r <= 3.0
    rho = U[0]
    # the Gaussian, exactly, inside R; the uniform ambient outside
    want = p["rho_c"] * np.exp(-(r / 3.0) ** 2 * np.log(3.0))
    assert np.max(np.abs(rho[cl] / want[cl] - 1.0)) < 1e-12
    assert np.all(rho[~cl] == p["rho_amb"])
    # pressure equilibrium: the ambient at 60 K presses as the edge at 40 K
    P_edge = p["rho_c"] / 3.0 * K_B * 40.0 / (1.3 * M_H)
    P_amb = p["rho_amb"] * K_B * 60.0 / (1.3 * M_H)
    assert P_amb == pytest.approx(P_edge, rel=1e-12)
    # the ambient at rest; the cloud's mass-weighted 3-D rms speed = v0
    v2 = (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho ** 2
    assert np.all(v2[~cl] == 0.0)
    v_rms = np.sqrt(np.sum(rho[cl] * v2[cl]) / np.sum(rho[cl]))
    assert v_rms == pytest.approx(0.616, rel=1e-12)
    # the mass inside R: M up to the cell-centre cut of the edge
    s = cloud_summary(U, p, cfg.Lx / cfg.Nx, cfg.GRAVITATIONAL_G)
    print(f"M_cloud {s['M_cloud']:.2f}  Q = Ekin/|Epot| {s['Q']:.3f}  alpha {s['alpha_vir']:.3f}")
    assert s["M_cloud"] == pytest.approx(1000.0, rel=0.02)


def test_kolmogorov_spectrum_band_and_slope():
    """E(k) ~ k^-5/3 on 2 <= |k| <= 32 and nothing outside, per component."""
    N = 96
    v = _kolmogorov_field(N, 2, 32, seed=11)
    kf = np.fft.fftfreq(N) * N
    k = np.sqrt(kf[:, None, None] ** 2 + kf[None, :, None] ** 2 + kf[None, None, :] ** 2)
    P = sum(np.abs(np.fft.fftn(c)) ** 2 for c in v)
    tot = P.sum()
    out = ((k < 2) | (k > 32)).astype(float)
    assert np.sum(P * out) < 1e-20 * tot
    kb = np.arange(3, 31)
    E = np.array([P[(k >= a - 0.5) & (k < a + 0.5)].sum() for a in kb])
    slope = np.polyfit(np.log(kb), np.log(E), 1)[0]
    print(f"E(k) slope {slope:.3f}")
    assert slope == pytest.approx(-5.0 / 3.0, abs=0.1)


def test_seeded_and_sliceable():
    cfg = _m3(N=32)
    a = np.asarray(turbulent_sphere_3d(cfg))
    assert np.array_equal(a, np.asarray(turbulent_sphere_3d(cfg)))
    assert not np.array_equal(a, np.asarray(turbulent_sphere_3d(replace(cfg, seed=6))))
    part = np.asarray(turbulent_sphere_3d(cfg, x_slice=(8, 24), z_slice=(0, 16)))
    assert np.array_equal(part, a[:, 8:24, :, 0:16])


def test_explicit_ambient_density():
    cfg = _m3(N=32, amb_n_cm3=50.0, amb_mu=1.27)
    p = cloud_params(cfg)
    assert p["rho_amb"] * MSUN / PC ** 3 == pytest.approx(50.0 * 1.27 * M_H, rel=1e-12)


def test_the_old_cloud_is_unchanged():
    """Top hat + power law with zero_drift=False: byte for byte what the
    study ran with."""
    cfg = _m3(N=24, profile="tophat", turb_spectrum="power", mach=8.0,
              zero_drift=False,
              mass_msun=2000.0, radius_pc=2.0, Lx=8.0, Ly=8.0, Lz=8.0)
    from luna.ic_generation.turbulent_sphere_3d import _turbulent_field
    p = cloud_params(cfg)
    x = p["xc"]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt((X - 4) ** 2 + (Y - 4) ** 2 + (Z - 4) ** 2)
    inside = 0.5 * (1.0 - np.tanh((r - 2.0) / p["dx"]))
    rho = p["rho_amb"] + (p["rho_cloud"] - p["rho_amb"]) * inside
    v = _turbulent_field(24, p["slope"], p["seed"]) * p["sigma_v"]
    v *= inside
    Pr = rho * p["cs"] * p["cs"]
    E = Pr / (p["gamma"] - 1.0) + 0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    ref = np.stack([rho, rho * v[0], rho * v[1], rho * v[2], E])
    assert np.array_equal(np.asarray(turbulent_sphere_3d(cfg)), ref)
