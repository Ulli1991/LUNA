"""Adiabatic expansion of a uniform gas in super-comoving variables (A16, part A): rho~ and v~ = a v_pec stay
constant and p~ = a^5 p goes as a^(5 - 3 gamma) (Martel & Shapiro 1998 eqs. 41-44), so T ~ a^(-3 (gamma - 1)) and
v_pec ~ 1/a, from z = 20 to 0 in Einstein-de Sitter; the clock against the exact EdS tau(a). gamma 5/3 and 7/5."""
import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve_staged

A_I, A_SNAP = 1.0 / 21.0, (0.1, 0.25, 0.5, 1.0)
N, P0, V0 = 8, 1e-4, 0.3                         # p~0 and v~0 in code units (H0 = 1, box 1)
# round-off level (measured 1.8e-14 at 5/3, 2.8e-14 at 7/5; v~ exact): the stage's factors telescope to
# (a_1 / a_0)^(5 - 3 gamma) exactly, so the a(tau) table error (CLOCK) cancels
BOUND = {5.0 / 3.0: dict(p=1e-12, v=1e-13), 1.4: dict(p=1e-12, v=1e-13)}
CLOCK_BOUND = 5.8e-6                             # 1.3 x measured 4.45e-6 (linear interpolation in the a(tau) table)


def cfg_for(gamma, **kw):
    from luna.ic_generation import zeldovich
    return accuracy_cfg(**{**dict(test_problem="Zeldovich-Pancake", SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N, Lx=1.0,
                                  Ly=1.0, Lz=1.0, BOUNDARY=0, gamma=gamma, CosmologicalIntegration=True,
                                  cosmo_Omega_m=1.0, cosmo_Omega_L=0.0, cosmo_H0=1.0, a_init=A_I, a_end=1.0,
                                  dloga_max=0.02, cosmo_p_floor=0.0, cosmo_T_floor=0.0,
                                  problem=zeldovich.Params()), **kw})


def uniform(gamma):
    """rho~ = 1, v~ = (V0, 0, 0), p~ = P0 on the grid."""
    U = np.zeros((5, N, N, N))
    U[0], U[1] = 1.0, V0
    U[4] = P0 / (gamma - 1.0) + 0.5 * V0 * V0
    return U


def run(gamma, **kw):
    """(a, max |p~/exact - 1|, max |v~/V0 - 1|, max |rho~ - 1|) at the snapshots."""
    from luna.setup.clock import make_clock
    cfg = cfg_for(gamma, **kw)
    clock = make_clock(cfg)
    Us = evolve_staged(cfg, uniform(gamma), [clock.tau_of_a(a) for a in A_SNAP])
    out = []
    for a, U in zip(A_SNAP, Us):
        v = U[1] / U[0]
        p = (gamma - 1.0) * (U[4] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0])
        p_exact = P0 * (a / A_I) ** (5.0 - 3.0 * gamma)
        out.append((a, float(np.abs(p / p_exact - 1).max()), float(np.abs(v / V0 - 1).max()),
                    float(np.abs(U[0] - 1).max())))
    return out


def clock_error():
    """max |tau(a) - tau_EdS(a)| / tau_EdS over the snapshots, tau_EdS = 2 (a_i^-1/2 - a^-1/2) / H0."""
    from luna.setup.clock import make_clock
    clock = make_clock(cfg_for(5.0 / 3.0))
    exact = lambda a: 2.0 * (A_I ** -0.5 - a ** -0.5)      # noqa: E731
    return max(abs((clock.tau_of_a(a) - clock.tau_of_a(A_I)) / exact(a) - 1) for a in A_SNAP)


@pytest.mark.parametrize("gamma", list(BOUND), ids=["5/3", "7/5"])
def test_adiabatic_expansion(gamma):
    """p~ against a^(5 - 3 gamma), v~ and rho~ constant at a = 0.1 .. 1."""
    res = run(gamma)
    print(f"MEASURED adiabatic gamma={gamma:.4g} = {res}")
    for a, ep, ev, er in res:
        check(f"adiabatic gamma={gamma:.4g} a={a} |p~/exact - 1|", ep, BOUND[gamma]["p"])
        check(f"adiabatic gamma={gamma:.4g} a={a} |v~/v0 - 1|", ev, BOUND[gamma]["v"])
        check(f"adiabatic gamma={gamma:.4g} a={a} |rho~ - 1|", er, 1e-13)


def test_clock_matches_eds():
    """The a(tau) table against the exact Einstein-de Sitter tau(a)."""
    check("EdS clock |tau / tau_exact - 1|", clock_error(), CLOCK_BOUND)


def test_bound_catches_a_missing_term(monkeypatch):
    """Teeth: without the (3 gamma - 5) term (the bug before it existed) gamma 7/5 misses its p~ bound."""
    import luna.sources.comoving_energy as ce
    monkeypatch.setattr(ce, "setup_comoving_energy", lambda *a, **k: None)
    assert max(r[1] for r in run(1.4)) > BOUND[1.4]["p"]


def test_bound_catches_a_pressure_floor():
    """Teeth: a pressure floor above the adiabat (the pancake test's cosmo_p_floor kind) lifts p~ off it."""
    assert max(r[1] for r in run(5.0 / 3.0, cosmo_p_floor=1.5 * P0)) > BOUND[5.0 / 3.0]["p"]


def test_comoving_amr_refuses_gamma_not_5_3():
    """Comoving AMR has no (3 gamma - 5) term: gamma 7/5 is refused, not silently wrong."""
    from luna.amr.cosmo import make_cosmo_amr

    class A:
        ndim = 3
    with pytest.raises(ValueError, match="gamma = 5/3"):
        make_cosmo_amr(cfg_for(1.4), A(), 1.0)
