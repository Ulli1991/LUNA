"""Isothermal atmosphere in phi = sin(2 pi x), Berberich et al. 2021 sec. 3.1: rho = p = exp(-phi), periodic,
t = 2 / sqrt(gamma). Our scheme is not well balanced: the drift away from the steady state and its order."""
import functools

import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve_staged

G = 1.4
T = 2.0 / np.sqrt(G)                                 # two sound crossings (c = sqrt(gamma))
SCHEMES = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1)}
NS, NORMS, FIELDS = (128, 256, 512, 1024), ("L1", "L2", "Linf"), ("rho", "mom", "E")
# {scheme: {field: norms per N, orders}} of U(T) - U(0), ~1.3x measured, orders -0.1 .. +0.3. Berberich
# Table 2 (third-order, unbalanced) has E L1 3.0e-4 at 128: a context, not a bound for a second-order scheme.
BOUNDS = {
    "hllc-minmod": {
        "rho": dict(L1={128: 9.9e-05, 256: 2.5e-05, 512: 6.2e-06, 1024: 1.6e-06}, L1_order=(1.90, 2.30),
            L2={128: 0.00011, 256: 2.8e-05, 512: 7.1e-06, 1024: 1.8e-06}, L2_order=(1.89, 2.29),
            Linf={128: 0.00019, 256: 4.6e-05, 512: 1.2e-05, 1024: 3.1e-06}, Linf_order=(1.87, 2.27)),
        # L1 7.6e-05, 1.91e-05, 4.78e-06, 1.2e-06; 2.00 | L2 8.55e-05, 2.15e-05, 5.42e-06, 1.36e-06; 1.99 | Linf 0.000145, 3.55e-05, 9.4e-06, 2.41e-06; 1.97
        "mom": dict(L1={128: 0.00018, 256: 4.5e-05, 512: 1.1e-05, 1024: 2.8e-06}, L1_order=(1.91, 2.31),
            L2={128: 0.00021, 256: 5.2e-05, 512: 1.3e-05, 1024: 3.2e-06}, L2_order=(1.90, 2.30),
            Linf={128: 0.00041, 256: 9.2e-05, 512: 2.2e-05, 1024: 5.3e-06}, Linf_order=(1.99, 2.39)),
        # L1 0.000141, 3.48e-05, 8.63e-06, 2.15e-06; 2.01 | L2 0.00016, 3.98e-05, 9.94e-06, 2.48e-06; 2.00 | Linf 0.000316, 7.07e-05, 1.66e-05, 4.06e-06; 2.09
        "E": dict(L1={128: 0.0007, 256: 0.00017, 512: 4.3e-05, 1024: 1.1e-05}, L1_order=(1.91, 2.31),
            L2={128: 0.00071, 256: 0.00018, 512: 4.4e-05, 1024: 1.1e-05}, L2_order=(1.91, 2.31),
            Linf={128: 0.00082, 256: 0.0002, 512: 5e-05, 1024: 1.2e-05}, Linf_order=(1.91, 2.31)),
        # L1 0.000541, 0.000134, 3.33e-05, 8.31e-06; 2.01 | L2 0.000548, 0.000135, 3.36e-05, 8.37e-06; 2.01 | Linf 0.000631, 0.000155, 3.84e-05, 9.57e-06; 2.01
    },
    "hll-minmod": {
        "rho": dict(L1={128: 0.0057, 256: 0.0017, 512: 0.00048, 1024: 0.00013}, L1_order=(1.71, 2.11),
            L2={128: 0.01, 256: 0.0034, 512: 0.0011, 1024: 0.00034}, L2_order=(1.53, 1.93),
            Linf={128: 0.035, 256: 0.015, 512: 0.0063, 1024: 0.0025}, Linf_order=(1.16, 1.56)),
        # L1 0.00437, 0.00131, 0.000372, 0.000102; 1.81 | L2 0.00767, 0.00258, 0.000827, 0.000262; 1.63 | Linf 0.027, 0.0116, 0.00482, 0.00196; 1.26
        "mom": dict(L1={128: 0.00023, 256: 5.1e-05, 512: 1.3e-05, 1024: 3e-06}, L1_order=(1.97, 2.37),
            L2={128: 0.00026, 256: 5.8e-05, 512: 1.4e-05, 1024: 3.5e-06}, L2_order=(1.96, 2.36),
            Linf={128: 0.00038, 256: 0.00011, 512: 3.5e-05, 1024: 9.5e-06}, Linf_order=(1.66, 2.06)),
        # L1 0.000175, 3.93e-05, 9.63e-06, 2.35e-06; 2.07 | L2 0.000197, 4.45e-05, 1.1e-05, 2.7e-06; 2.06 | Linf 0.000291, 8.67e-05, 2.69e-05, 7.28e-06; 1.76
        "E": dict(L1={128: 0.0019, 256: 0.00042, 512: 0.00011, 1024: 2.7e-05}, L1_order=(1.94, 2.34),
            L2={128: 0.003, 256: 0.00063, 512: 0.00015, 1024: 3.6e-05}, L2_order=(2.02, 2.42),
            Linf={128: 0.0075, 256: 0.0017, 512: 0.0004, 1024: 9.4e-05}, Linf_order=(2.01, 2.41)),
        # L1 0.00149, 0.000326, 8.35e-05, 2.1e-05; 2.04 | L2 0.00228, 0.000481, 0.000113, 2.77e-05; 2.12 | Linf 0.00578, 0.00134, 0.00031, 7.22e-05; 2.11
    },
    "hllc-mc": {
        "rho": dict(L1={128: 6e-05, 256: 1.7e-05, 512: 4.5e-06, 1024: 1.1e-06}, L1_order=(1.81, 2.21),
            L2={128: 7.2e-05, 256: 1.9e-05, 512: 5.1e-06, 1024: 1.3e-06}, L2_order=(1.83, 2.23),
            Linf={128: 0.00016, 256: 3.8e-05, 512: 9.6e-06, 1024: 2.5e-06}, Linf_order=(1.90, 2.30)),
        # L1 4.65e-05, 1.31e-05, 3.44e-06, 8.83e-07; 1.91 | L2 5.5e-05, 1.5e-05, 3.9e-06, 9.94e-07; 1.93 | Linf 0.000125, 2.93e-05, 7.4e-06, 1.94e-06; 2.00
        "mom": dict(L1={128: 0.00014, 256: 3.4e-05, 512: 8.4e-06, 1024: 2.1e-06}, L1_order=(1.93, 2.33),
            L2={128: 0.00017, 256: 4.1e-05, 512: 1e-05, 1024: 2.5e-06}, L2_order=(1.92, 2.32),
            Linf={128: 0.00046, 256: 0.0001, 512: 2.4e-05, 1024: 5.8e-06}, Linf_order=(2.00, 2.40)),
        # L1 0.000109, 2.62e-05, 6.43e-06, 1.59e-06; 2.03 | L2 0.00013, 3.14e-05, 7.75e-06, 1.93e-06; 2.02 | Linf 0.000354, 7.93e-05, 1.86e-05, 4.49e-06; 2.10
        "E": dict(L1={128: 0.00027, 256: 6.6e-05, 512: 1.6e-05, 1024: 4.1e-06}, L1_order=(1.91, 2.31),
            L2={128: 0.0003, 256: 7.5e-05, 512: 1.9e-05, 1024: 4.7e-06}, L2_order=(1.91, 2.31),
            Linf={128: 0.00057, 256: 0.00013, 512: 3.5e-05, 1024: 9e-06}, Linf_order=(1.89, 2.29)),
        # L1 0.000206, 5.07e-05, 1.26e-05, 3.13e-06; 2.01 | L2 0.000234, 5.79e-05, 1.44e-05, 3.59e-06; 2.01 | Linf 0.000438, 0.000103, 2.7e-05, 6.93e-06; 1.99
    },
}

# Total energy E + rho phi in the static potential: not conserved exactly (the source kicks with the analytic g at
# cell centres, not the discrete gradient of phi), a truncation drift; mass in evolve_staged; momentum is not
# conserved (the external force). |d(E + rho phi)| / (E + rho phi) at T, ~1.3x measured, orders -0.1 .. +0.3.
ENERGY = {"hllc-minmod": {128: 2.9e-04, 256: 7.1e-05, 512: 1.7e-05, 1024: 4.4e-06},
          # 2.21e-04, 5.42e-05, 1.34e-05, 3.35e-06; 2.01
          "hll-minmod": {128: 9.3e-04, 256: 2.2e-04, 512: 5.3e-05, 1024: 1.3e-05},
          # 7.19e-04, 1.68e-04, 4.08e-05, 1.01e-05; 2.05
          "hllc-mc": {128: 9.7e-06, 256: 1.2e-06, 512: 1.4e-07, 1024: 1.8e-08}}
          # 7.48e-06, 9.06e-07, 1.11e-07, 1.38e-08; 3.03
ENERGY_ORDER = {"hllc-minmod": (1.91, 2.31), "hll-minmod": (1.95, 2.35), "hllc-mc": (2.93, 3.33)}


def steady(n, q=5):
    """Exact cell averages of rho = p = exp(-sin 2 pi x), u = 0 (Gauss-Legendre, q points)."""
    gx, gw = np.polynomial.legendre.leggauss(q)
    x = ((np.arange(n)[:, None] + 0.5 + 0.5 * gx[None, :]) / n)
    rho = (0.5 * gw[None, :] * np.exp(-np.sin(2 * np.pi * x))).sum(1)
    return np.stack([rho, np.zeros(n), rho / (G - 1.0)])


def cfg_for(n, scheme, sign=1.0):
    solver, limiter = SCHEMES[scheme]
    return accuracy_cfg(test_problem="Soundwave", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0, BOUNDARY=0, gamma=G,
                        riemann_solver=solver, LIMITER=limiter, external_potential="sin", external_phi0=sign)


@functools.lru_cache(maxsize=None)
def errors(n, scheme, sign=1.0):
    """{field: (L1, L2, Linf)} of U(T) - U(0)."""
    U0 = steady(n)
    U = evolve_staged(cfg_for(n, scheme, sign), U0, T)
    phi = sign * np.sin(2 * np.pi * (np.arange(n) + 0.5) / n)
    etot = lambda V: float((V[2] + V[0] * phi).sum())                          # noqa: E731
    out = {f: (float(np.abs(e).mean()), float(np.sqrt((e * e).mean())), float(np.abs(e).max()))
           for f, e in zip(FIELDS, U - U0)}
    out["budget"] = abs(etot(U) / etot(U0) - 1.0)                               # |d(E + rho phi)| / (E + rho phi)
    return out


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_isothermal_atmosphere(scheme):
    """L1, L2, Linf of rho, rho u, E against the steady state at every N and their orders; mass exact."""
    b = BOUNDS[scheme]
    res = [errors(n, scheme) for n in NS]
    for fl in FIELDS:
        for j, nm in enumerate(NORMS):
            vals = [r[fl][j] for r in res]
            for n, v in zip(NS, vals):
                check(f"atmosphere {scheme} N={n} {fl} {nm}", v, b[fl][nm][n])
            order = -np.polyfit(np.log(NS), np.log(vals), 1)[0]
            check(f"atmosphere {scheme} {fl} {nm} order", order, b[fl][f"{nm}_order"][0], lower=True)
            check(f"atmosphere {scheme} {fl} {nm} order", order, b[fl][f"{nm}_order"][1])
    budgets = [r["budget"] for r in res]
    for n, v in zip(NS, budgets):
        check(f"atmosphere {scheme} N={n} |d(E + rho phi)|/(E + rho phi)", v, ENERGY[scheme][n])
    order = -np.polyfit(np.log(NS), np.log(budgets), 1)[0]
    check(f"atmosphere {scheme} energy order", order, ENERGY_ORDER[scheme][0], lower=True)
    check(f"atmosphere {scheme} energy order", order, ENERGY_ORDER[scheme][1])


def test_bound_catches_a_flipped_potential():
    """Teeth: with phi -> -phi the atmosphere falls over (measured rho L1 1.28 at 256)."""
    assert errors(256, "hllc-minmod", sign=-1.0)["rho"][0] > BOUNDS["hllc-minmod"]["rho"]["L1"][256]
