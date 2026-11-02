"""The turbulent cloud does not drift: its gas has zero net momentum.

The turbulent field has zero mean over the box, but masked to the cloud it
kept a bulk velocity (the Burgers study cloud: 0.83 km/s, 17 % of E_kin).
The cloud gas's mass-weighted mean velocity is now removed and the rms put
back to the requested value, on both the top-hat power-law and the Torch
(Gaussian, Kolmogorov) paths. The pre-fix code fails the momentum check.
"""
from dataclasses import replace
import json

import numpy as np
import pytest

from luna.ic_generation.ic_file import _fingerprint, read_ic, write_ic
from luna.ic_generation.turbulent_sphere_3d import (Params, _rho_slab, cloud_params,
                                                    turbulent_sphere_3d, velocity_field)
from luna.units import UnitSystem
from tests._helpers import _with_problem

UNITS = UnitSystem(UnitLength_in_cm=3.085677581e18, UnitMass_in_g=1.98892e33,
                   UnitVelocity_in_cm_per_s=1.0e5)
N = 48


def _burgers(**kw):
    """The res3 Burgers study cloud, smaller grid."""
    base = dict(test_problem="Turbulent-Sphere-3D", SPATIAL_DIMENSION=3, BOUNDARY=4,
                Nx=N, Ny=N, Nz=N, Lx=8.0, Ly=8.0, Lz=8.0, gamma=5.0 / 3.0, units=UNITS,
                mass_msun=2000.0, radius_pc=2.0, T_K=15.0,
                mach=8.0, sink_star_mu=2.3, turb_slope=-2.0, seed=1)
    base.update(kw)
    return _with_problem(Params, **base)


def _wall(**kw):
    """Wall et al. 2020 M3: Gaussian cloud, Kolmogorov k = 2..32, v0."""
    base = dict(test_problem="Turbulent-Sphere-3D", SPATIAL_DIMENSION=3, BOUNDARY=4,
                Nx=N, Ny=N, Nz=N, Lx=10.0, Ly=10.0, Lz=10.0, gamma=5.0 / 3.0, units=UNITS,
                mass_msun=1000.0, radius_pc=3.0, T_K=40.0, sink_star_mu=1.3,
                profile="gaussian", edge_ratio=1.0 / 3.0, amb_T_K=60.0,
                turb_spectrum="kolmogorov", turb_v0_kms=0.616,
                turb_kmin=2, turb_kmax=16, seed=5)
    base.update(kw)
    return _with_problem(Params, **base)


def _drift(U):
    """|net momentum| / sum |momentum| of all the gas in the box."""
    P = np.array([U[k].sum() for k in (1, 2, 3)])
    return np.linalg.norm(P) / np.sum(np.sqrt(U[1] ** 2 + U[2] ** 2 + U[3] ** 2))


def _cloud_rms(cfg):
    """Mass-weighted rms of the field over the cloud gas (weight rho * mask)."""
    p = cloud_params(cfg)
    v = velocity_field(p)
    x = p["xc"]
    rho, inside = _rho_slab(p, x[:, None, None], x[None, :, None], x[None, None, :])
    w = rho * inside
    return np.sqrt(np.sum(w * np.sum(v ** 2, axis=0)) / np.sum(w)), p


@pytest.mark.parametrize("make", [_burgers, _wall], ids=["tophat-burgers", "gaussian-kolmogorov"])
def test_no_bulk_momentum(make):
    U = np.asarray(turbulent_sphere_3d(make()))
    d = _drift(U)
    print(f"net / total momentum {d:.2e}")
    assert d < 1e-12


@pytest.mark.parametrize("make", [_burgers, _wall], ids=["tophat-burgers", "gaussian-kolmogorov"])
def test_the_old_field_drifts(make):
    """The same cloud built the old way has a bulk momentum far above round-off."""
    d = _drift(np.asarray(turbulent_sphere_3d(make(zero_drift=False))))
    print(f"old: net / total momentum {d:.3f}")
    assert d > 1e-3


def test_rms_is_the_requested_one():
    rms, _ = _cloud_rms(_burgers())
    assert rms == pytest.approx(1.0, rel=1e-12)            # times sigma_v in the slab
    rms, p = _cloud_rms(_wall())
    assert rms == pytest.approx(p["v0"], rel=1e-12)
    # and in the gas itself: the Gaussian's mask is 0/1, so the cloud's
    # mass-weighted 3-D rms speed is v0 exactly
    U = np.asarray(turbulent_sphere_3d(_wall()))
    v2 = (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0] ** 2
    cl = v2 > 0
    assert np.sqrt(np.sum(U[0][cl] * v2[cl]) / np.sum(U[0][cl])) == pytest.approx(0.616, rel=1e-12)


def test_an_ic_written_before_the_fix_is_refused(tmp_path):
    cfg = _burgers(Nx=16, Ny=16, Nz=16)
    write_ic(cfg, str(tmp_path))
    meta = tmp_path / "meta.json"
    fp = json.loads(meta.read_text())
    assert fp["cloud_zero_drift"] is True
    del fp["cloud_zero_drift"]                         # what an old write_ic left
    meta.write_text(json.dumps(fp))
    with pytest.raises(ValueError, match="cloud_zero_drift"):
        read_ic(replace(cfg, ic_dir=str(tmp_path)))
    assert _fingerprint(cfg) != _fingerprint(replace(cfg, problem=replace(cfg.problem, zero_drift=False)))
