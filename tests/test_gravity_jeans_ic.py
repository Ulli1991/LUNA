"""The registered Jeans-Instability-3D IC is the adiabatic mode: delta P = c_s^2 delta rho, no entropy-mode content.

The entropy perturbation of rho = rho0 (1 + a c), P = P0 (1 + gamma a c) is second order in a (Taylor expansion of
ln(P / rho^gamma)): |delta ln s| <= gamma (gamma - 1) / 2 a^2 + gamma (gamma^2 - 1) / 3 a^3, while the old
P = P0 (1 + a c) carried a first-order entropy mode, |delta ln s| = (gamma - 1) a |c|."""

import numpy as np
import pytest

from tests._helpers import accuracy_cfg

GAM, AMP = 5.0 / 3.0, 1e-4
# bound: the Taylor terms above, plus FP64 round-off of ln at O(1) values (1e-14)
BOUND = GAM * (GAM - 1) / 2 * AMP ** 2 + GAM * (GAM ** 2 - 1) / 3 * AMP ** 3 + 1e-14


def cfg_for(G, eigenmode=True, k=(0.0, 0.0, 0.0)):
    """16 x 8 x 8 cells of 1 x 0.5 x 0.5, c_s = 1; G sets the mode stable (small) or unstable (large)."""
    from luna.ic_generation import jeans3d
    return accuracy_cfg(test_problem="Jeans-Instability-3D", SPATIAL_DIMENSION=3, Nx=16, Ny=8, Nz=8,
                        Lx=1.0, Ly=0.5, Lz=0.5, BOUNDARY=0, gamma=GAM, SELF_GRAVITY=True, GRAVITATIONAL_G=G,
                        problem=jeans3d.Params(rho0=1.0, P0=1.0 / GAM, amp=AMP, kx=k[0], ky=k[1], kz=k[2],
                                               eigenmode=eigenmode))


CASES = {"stable": (0.1, True, (0.0, 0.0, 0.0)),            # 4 pi G = 1.26 < c^2 k^2 = 39.5
         "stable oblique": (0.1, True, (2 * np.pi, 4 * np.pi, 4 * np.pi)),
         "unstable eigenmode": (10.0, True, (0.0, 0.0, 0.0)),
         "unstable at rest": (10.0, False, (0.0, 0.0, 0.0))}


def state(cfg):
    """(rho, v, P) of the registered IC."""
    from luna.ic_generation.ic import initial_conditions
    U = np.asarray(initial_conditions(cfg), np.float64)
    v = U[1:4] / U[0]
    return U[0], v, (GAM - 1) * (U[4] - 0.5 * U[0] * (v * v).sum(0))


def entropy_and_cs2(rho, P, cfg):
    """max |delta ln(P / rho^gamma)| and the fitted delta P / delta rho."""
    pr = cfg.problem
    ds = np.log(P / pr.P0) - GAM * np.log(rho / pr.rho0)
    drho, dP = rho - pr.rho0, P - pr.P0
    return float(np.abs(ds).max()), float((dP * drho).sum() / (drho * drho).sum())


@pytest.mark.parametrize("name", list(CASES))
def test_registered_ic_is_adiabatic(name):
    """No first-order entropy content; delta P / delta rho = c_s^2 = gamma P0 / rho0 = 1; at rest when stable."""
    from luna.ic_generation.jeans3d import sigma_squared
    cfg = cfg_for(*CASES[name])
    assert (sigma_squared(cfg) > 0) == name.startswith("unstable")
    rho, v, P = state(cfg)
    ds, cs2 = entropy_and_cs2(rho, P, cfg)
    print(f"MEASURED jeans IC {name}: max |delta ln s| = {ds:.3g} (bound {BOUND:.3g}), dP/drho = {cs2:.12f}")
    assert np.abs(rho - 1.0).max() > 0.9 * AMP                  # the mode is there
    assert ds <= BOUND
    assert abs(cs2 - 1.0) < 1e-8                                # round-off of P from E, ~1e-16 / (gamma P0 amp) = 1e-12
    if name.startswith("stable") or name.endswith("at rest"):
        assert np.abs(v).max() == 0.0


def test_bound_catches_the_old_isothermal_pressure():
    """Teeth: the old stable-mode P = P0 (1 + amp cos) has (gamma - 1) amp = 6.7e-5 entropy, 1e4 x the bound."""
    cfg = cfg_for(*CASES["stable"])
    rho, _, _ = state(cfg)
    pr = cfg.problem
    ds, cs2 = entropy_and_cs2(rho, pr.P0 * (1.0 + (rho / pr.rho0 - 1.0)), cfg)
    print(f"MEASURED jeans IC old pressure: max |delta ln s| = {ds:.3g}, dP/drho = {cs2:.6f}")
    assert ds > 1e3 * BOUND and abs(cs2 - 1.0) > 0.3


def test_bound_catches_a_wrong_gamma():
    """Teeth: the adiabatic form with gamma 1 % off leaves 0.01 gamma amp entropy, above the bound."""
    cfg = cfg_for(*CASES["stable"])
    rho, _, _ = state(cfg)
    pr = cfg.problem
    ds, _ = entropy_and_cs2(rho, pr.P0 * (1.0 + 1.01 * GAM * (rho / pr.rho0 - 1.0)), cfg)
    assert ds > 10 * BOUND
