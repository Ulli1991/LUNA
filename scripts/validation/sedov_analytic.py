"""Sedov-Taylor self-similar blast wave: analytic solution.

For a point explosion of energy E_0 into a uniform medium of density rho_0
with adiabatic index gamma, the spherical shock front position is
    R_s(t) = xi_0 * (E_0 * t^2 / rho_0)^(1/5)
where xi_0 is a gamma-dependent dimensionless constant (Sedov 1959).

Behind the shock the dimensionless profiles V(xi), G(xi), Z(xi) for
velocity, density, and pressure follow from the Sedov ODE system. For
gamma=5/3 the closed-form expressions are tabulated in Landau & Lifshitz
(1987, §106) and Kamm & Timmes (2007, LA-UR-07-2849).

Implementation strategy when this gets written:
- Use Kamm & Timmes parametric closed-form (avoid a stiff ODE integration).
- Return (r, rho, v_r, P) on a user-supplied radial grid, given (t, E_0,
  rho_0, gamma).
- Validate against a couple of tabulated points from Kamm & Timmes.

NOT IMPLEMENTED yet — placeholder. Filled in when we run the Sedov test.
"""
import numpy as np


def sedov_taylor_3d(r, t, E0=1.0, rho0=1.0, gamma=5.0 / 3.0):
    """Sedov-Taylor self-similar solution at time t on radius array r.

    Returns dict {"rho": ..., "v_r": ..., "P": ..., "R_shock": ...}.
    """
    raise NotImplementedError(
        "sedov_taylor_3d: implement via Kamm & Timmes (2007) closed form "
        "when we get to the 3D Sedov test. See module docstring for refs."
    )
