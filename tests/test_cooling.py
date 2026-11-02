"""Validate the explicit sub-cycled cooling integrator (Townsend removed).

Checks, on a hot isochoric parcel with the n_e n_H normalization:
  (1) Physical rate: instantaneous cooling time matches the textbook
      t_cool = (3/2) n_tot k_B T / (n_e n_H Lambda(T)).
  (2) Convergence: the forward-Euler sub-cycled update (as used in
      apply_cooling) converges to a fine RK4 integration of
      dT/dt = -(gamma-1) n_e n_H Lambda(T) / (n_tot k_B) as n_sub grows.
"""

import numpy as np


from luna.sources.cooling import (sutherland_dopita_cie, ExplicitCoolingSolver,
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


def test_apply_cooling_tracks_fast_cooling_with_adaptive_substeps():
    """A parcel with dt = 0.6 t_cool (it ends well above the floor, so the
    integration itself is tested): the adaptive sub-cycling (dt_frac 0.1,
    cap 128) must track a fine RK4 integration to 2% and beat a single
    forward-Euler step (dt_frac huge) by a wide margin; the diagnostic must
    report the sub-steps the cell needs (>= 6 here, more as t_cool shrinks
    while it cools)."""
    from dataclasses import replace

    import jax.numpy as jnp
    from luna.sources.cooling import apply_cooling, cooling_substeps_needed
    from luna.template_config import Config
    from luna.units import UnitSystem

    units = UnitSystem(UnitLength_in_cm=3.085678e21, UnitMass_in_g=1.989e43,
                       UnitVelocity_in_cm_per_s=1.0e5)              # Arepo-like
    cfg = replace(Config(), units=units, gamma=GAMMA, SPATIAL_DIMENSION=3)
    table = sutherland_dopita_cie("solar")
    T0, n_H = 3.0e6, 0.1
    rho_cgs = n_H * PROTON_MASS_CGS / X_H
    n_tot = n_H / (X_H * MU)
    n_e = NE_PER_NH * n_H
    t_cool = 1.5 * n_tot * BOLTZMANN_CGS * T0 / (n_e * n_H * lam(T0, table))
    dt_s = 0.6 * t_cool
    u_cgs = BOLTZMANN_CGS * rho_cgs * T0 / ((GAMMA - 1.0) * MU * PROTON_MASS_CGS)
    U = np.zeros((5, 2, 2, 2))
    U[0] = rho_cgs / units.UnitDensity_in_cgs
    U[4] = u_cgs / units.UnitPressure_in_cgs
    U = jnp.asarray(U)
    dt_code = dt_s / units.UnitTime_in_s

    # reference: RK4 with the same interpolated Lambda (floor never reached)
    n_ref, h = 20000, dt_s / 20000
    T = T0
    for _ in range(n_ref):
        k1 = rate(T, n_e, n_H, n_tot, table)
        k2 = rate(T + 0.5 * h * k1, n_e, n_H, n_tot, table)
        k3 = rate(T + 0.5 * h * k2, n_e, n_H, n_tot, table)
        k4 = rate(T + h * k3, n_e, n_H, n_tot, table)
        T = max(T + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 1.0e4)
    T_ref = T

    def T_of(U_):
        u = np.asarray(U_[4])[0, 0, 0] * units.UnitPressure_in_cgs
        return (GAMMA - 1.0) * MU * PROTON_MASS_CGS * u / (BOLTZMANN_CGS * rho_cgs)

    # This parcel is a cooling RUNAWAY (Lambda rises as T falls: RK4 ends near
    # 1.2e4 K), where a first-order sub-step of dt_frac * t_cool carries an
    # O(dt_frac) error in the time-to-cool that the runaway amplifies in T.
    # The gate is therefore convergence in dt_frac (cap large enough never to
    # bind) against the single-step limit, not a fixed 2% at dt_frac = 0.1.
    rels = []
    for frac in (0.2, 0.05, 0.0125):
        solver = ExplicitCoolingSolver(table=table, mu=MU, n_sub=8192, dt_frac=frac,
                                       X_H=X_H, ne_per_nH=NE_PER_NH)
        U_c = apply_cooling(U, dt_code, cfg, solver)
        # cooling changes the energy only (its budget is the RK4 gate below): rho and momentum bitwise
        np.testing.assert_array_equal(np.asarray(U_c[:4]), np.asarray(U[:4]))
        T_ad = T_of(U_c)
        rels.append(abs(T_ad - T_ref) / T_ref)
        print(f"dt = 0.6 t_cool, dt_frac={frac}: adaptive {T_ad:.4e} K vs RK4 {T_ref:.4e} K "
              f"(rel {rels[-1]:.2e})")
    need = float(cooling_substeps_needed(U, dt_code, cfg, solver))
    one_step = ExplicitCoolingSolver(table=table, mu=MU, n_sub=1, dt_frac=1.0e9,
                                     X_H=X_H, ne_per_nH=NE_PER_NH)
    T_1 = T_of(apply_cooling(U, dt_code, cfg, one_step))
    rel_1 = abs(T_1 - T_ref) / T_ref
    print(f"single Euler step {T_1:.4e} K (rel {rel_1:.2e}); sub-steps needed at "
          f"dt_frac=0.0125: {need:.0f}")
    assert need >= 48                                     # 0.6 / 0.0125, more as t_cool shrinks
    assert rels[0] > rels[1] > rels[2], rels               # converges with dt_frac
    assert rels[2] < 5e-2, rels
    assert rel_1 > rels[0]


if __name__ == "__main__":
    main()
