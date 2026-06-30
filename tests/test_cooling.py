"""Validate the explicit sub-cycled cooling integrator (Townsend removed).

Checks, on a hot isochoric parcel with the n_e n_H normalization:
  (1) Physical rate: instantaneous cooling time matches the textbook
      t_cool = (3/2) n_tot k_B T / (n_e n_H Lambda(T)).
  (2) Convergence: the forward-Euler sub-cycled update (as used in
      apply_cooling) converges to a fine RK4 integration of
      dT/dt = -(gamma-1) n_e n_H Lambda(T) / (n_tot k_B) as n_sub grows.
"""

import sys
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydro.cooling import (sutherland_dopita_cie, ExplicitCoolingSolver,
                           PROTON_MASS_CGS, BOLTZMANN_CGS)

GAMMA = 5.0 / 3.0
X_H, NE_PER_NH, MU = 0.7154, 1.189, 0.6


def lam(T, table):
    # smooth log-log interpolant, matching apply_cooling's jnp.interp
    return 10.0 ** np.interp(np.log10(T), table.log_T, table.log_Lambda)


def rate(T, n_e, n_H, n_tot, table):
    return -(GAMMA - 1.0) * n_e * n_H * lam(T, table) / (n_tot * BOLTZMANN_CGS)


def main():
    table = sutherland_dopita_cie("solar")
    solver = ExplicitCoolingSolver(table=table, mu=MU, n_sub=16)
    assert solver.T_floor == 1.0e4

    T0, n_H = 3.0e6, 0.1
    n_tot = n_H / (X_H * MU)
    n_e = NE_PER_NH * n_H

    t_cool = 1.5 * n_tot * BOLTZMANN_CGS * T0 / (n_e * n_H * lam(T0, table))
    print(f"t_cool(T0={T0:.1e}K, n_H={n_H}) = {t_cool:.3e} s = {t_cool/3.15e13:.3f} Myr")

    # reference: fine RK4 over 0.3 t_cool (stays above the 1e4 floor)
    total = 0.3 * t_cool
    n_ref = 40000
    h = total / n_ref
    T = T0
    for _ in range(n_ref):
        k1 = rate(T, n_e, n_H, n_tot, table)
        k2 = rate(T + 0.5*h*k1, n_e, n_H, n_tot, table)
        k3 = rate(T + 0.5*h*k2, n_e, n_H, n_tot, table)
        k4 = rate(T + h*k3, n_e, n_H, n_tot, table)
        T += (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    T_rk4 = T

    # (1) initial cooling time: -T0/(dT/dt) should equal (3/2)/(gamma-1)*t_cool
    tc_from_rate = -T0 / rate(T0, n_e, n_H, n_tot, table)
    rel_tc = abs(tc_from_rate - t_cool) / t_cool
    print(f"(1) cooling time from rate={tc_from_rate:.3e}s vs textbook={t_cool:.3e}s "
          f"rel={rel_tc:.2e}")
    assert rel_tc < 1e-12, "cooling-rate normalization mismatch"

    # (2) explicit forward-Euler sub-cycling converges to RK4
    print("(2) forward-Euler sub-cycling vs RK4:")
    for n_sub in (4, 16, 64, 256):
        dt_sub = total / n_sub
        Te = T0
        for _ in range(n_sub):
            Te = max(Te + rate(Te, n_e, n_H, n_tot, table) * dt_sub, 1.0e4)
        rel = abs(Te - T_rk4) / T_rk4
        print(f"    n_sub={n_sub:4d}  T={Te:.5e}  rel_to_RK4={rel:.2e}")
    assert rel < 1e-3, "forward-Euler not converging to RK4"

    print("\nALL COOLING CHECKS PASSED ✓")


def test_cooling_rate_and_integration():
    main()


if __name__ == "__main__":
    main()
