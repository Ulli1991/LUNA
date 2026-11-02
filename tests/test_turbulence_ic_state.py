"""The driven-turbulence box starts in the state its config asks for.

It used to hard-code rho = p = 1 code unit and ignore what the run scripts
computed, so the ISM setups printed "n_H = 1 cm^-3, T = 1e4 K" and ran at
T = 72.7 K -- below the 1e4 K cooling floor, where every cell clamps on the
first cooling call and the cooling term contributes nothing (audit sec.26).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.fast



def _state(cfg):
    from luna.ic_generation.ic import initial_conditions
    U = np.asarray(initial_conditions(cfg))
    rho = U[0]
    p = (cfg.gamma - 1.0) * U[4]          # v = 0 in this IC
    return float(rho.min()), float(rho.max()), float(p.min()), float(p.max())


def test_ic_uses_the_configured_background():
    from dataclasses import replace
    from luna.ic_generation.driven_turbulence_3d import Params
    from luna.template_config import Config
    cfg = replace(Config(), test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3,
                  Nx=8, Ny=8, Nz=8, gamma=5.0 / 3.0, problem=Params(rho0=1.76e-4, P0=0.485))
    rlo, rhi, plo, phi = _state(cfg)
    assert rlo == rhi == pytest.approx(1.76e-4)
    assert plo == phi == pytest.approx(0.485)
    # the defaults are still the dimensionless box the plain tests use
    dflt = replace(cfg, problem=Params())
    rlo, rhi, plo, phi = _state(dflt)
    assert rlo == pytest.approx(1.0) and plo == pytest.approx(1.0)


def test_the_cooling_setup_lands_above_its_floor():
    """The regime the cooling inversion documents: a temperature well above
    cooling_T_floor, or the cooling step is a clamp and its gradient is zero."""
    from scripts.examples.autodiff.ic_inversion_run import config
    from luna.units import PROTON_MASS_CGS, BOLTZMANN_CGS
    cfg = config(problem="cooling", nx=8)
    rho, _, p, _ = _state(cfg)
    rho_cgs = rho * cfg.units.UnitDensity_in_cgs
    u_cgs = p / (cfg.gamma - 1.0) * cfg.units.UnitPressure_in_cgs
    T = (cfg.gamma - 1.0) * cfg.cooling_mu * PROTON_MASS_CGS * u_cgs / (BOLTZMANN_CGS * rho_cgs)
    assert T > 10.0 * cfg.cooling_T_floor, (
        f"T0 = {T:.3g} K is not above the {cfg.cooling_T_floor:.3g} K floor: "
        "the cooling step can only clamp, and its gradient is identically zero")
    assert 1.0e5 < T < 1.0e6, f"T0 = {T:.3g} K is not the documented ~2e5 K regime"
