"""Evrard collapse energy budget (Evrard 1988; as Springel 2010 sec. 9.1): M = R = G = 1, rho = M / (2 pi R^2 r),
u = 0.05, gamma 5/3, here in a periodic box L = 4R. A cell-centred gravity source does not conserve energy
exactly: |E(t) - E(0)| / |E(0)| and its decrease with N (AREPO: ~50 % at 24464 cells, <1e-3 conservative)."""
import functools

import numpy as np
import pytest

from luna.ic_generation import evrard3d
from tests._helpers import accuracy_cfg, check, evolve_staged, gpu_only, self_gravity_energies

L, G, GAM = 4.0, 1.0, 5.0 / 3.0
TIMES = tuple(0.25 * k for k in range(1, 13))           # to t = 3, as Springel's Fig. 42
# {(scheme, N): |dE / E0| at TIMES}, 1.3x measured. First order (cusp, then the accretion shock):
# order at t = 3 measured 0.96 (hllc), 0.89 (hll). hllc-mc 32 went to p < 0 at the bounce until the C8 face fix.
BOUNDS = {
    ("hllc-minmod", 32): (0.016, 0.025, 0.096, 1.7, 3.6, 3.8, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9),
    # 0.0126 0.0192 0.0735 1.31 2.77 2.96 3 3.01 3.02 3.02 3.02 3.02
    ("hllc-minmod", 64): (0.0062, 0.012, 0.18, 0.94, 1.7, 1.8, 1.9, 1.9, 1.9, 2, 2, 2),
    # 0.00476 0.00915 0.137 0.726 1.27 1.41 1.45 1.47 1.49 1.51 1.53 1.55
    ("hll-minmod", 32): (0.025, 0.037, 0.12, 1.9, 3.8, 4, 4, 4.1, 4.1, 4.1, 4.1, 4.1),
    # 0.0195 0.0281 0.0906 1.48 2.91 3.08 3.11 3.12 3.12 3.12 3.12 3.12
    ("hll-minmod", 64): (0.0091, 0.016, 0.2, 1.2, 2, 2.1, 2.1, 2.2, 2.2, 2.2, 2.2, 2.2),
    # 0.00702 0.012 0.156 0.912 1.52 1.63 1.65 1.66 1.67 1.67 1.68 1.68
    ("hllc-mc", 32): (0.012, 0.0086, 0.05, 1.7, 3.1, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2),
    # 0.00863 0.00661 0.038 1.3 2.35 2.39 2.39 2.39 2.39 2.39 2.39 2.39 (CPU FP64, after the C8 face fix)
    ("hllc-mc", 64): (0.0031, 0.0027, 0.16, 0.8, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.4, 1.4),
    # 0.00237 0.00205 0.121 0.617 0.969 1 1.01 1.02 1.02 1.04 1.06 1.08
}
# max_d |dP_d| / sqrt(M E_gas): periodic, sum rho D L^-1 rho = 0 for the exact 7-point inverse (D antisymmetric and
# commuting with L), so the drift is the warm solve's tolerance (1e-4 rms residual) acting on the asymmetric part.
# Minmod stays mirror-symmetric: round-off (measured <= 4.6e-15, A100 and CPU). hllc-mc 64 loses the symmetry at
# the bounce: 4.3e-8 at t = 1.5 (A100 = CPU), 1.6e-12 with gravity_warm_tol 1e-9, so the solve, not a bug.
P_TOL = {("hllc-minmod", 32): 1e-14, ("hllc-minmod", 64): 1e-14, ("hll-minmod", 32): 1e-14,
         ("hll-minmod", 64): 1e-14, ("hllc-mc", 32): 1e-14, ("hllc-mc", 64): 6e-8}
ORDER = {"hllc-minmod": (0.86, 1.26), "hll-minmod": (0.79, 1.19)}   # of the t = 3 error, 32 -> 64


# hllc-mc's predicted faces go non-physical at the bounce (audit C8): it allows the first-order face state and
# asserts the count, FACES_MC[N] = (lo, hi), exact (CPU-8 job 31665570 and A100 job 31663810 agree at 32 and 64).
FACES_MC = {32: (5520, 5520), 64: (12216, 12216), 128: (94368, 94368), 256: (944208, 944208)}


def cfg_for(n, scheme="hllc-mc"):
    solver, limiter = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1)}[scheme]
    return accuracy_cfg(test_problem="Evrard-Collapse-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=L, Ly=L, Lz=L,
                        BOUNDARY=0, gamma=GAM, riemann_solver=solver, LIMITER=limiter, SELF_GRAVITY=True, gravity_energy="kick",
                        GRAVITATIONAL_G=G, problem=evrard3d.Params(M=1.0, R=1.0, u=0.05),
                        allow_first_order_faces=scheme == "hllc-mc")      # MC: C8 at the bounce, FACES_MC


def energies(U, n):
    """(kinetic, thermal, potential): W = 1/2 sum rho phi dV, phi from an FFT solve of the 7-point Laplacian."""
    return self_gravity_energies(U, L / n, G)


def run_budget(n, scheme, times=TIMES):
    """[(t, E_kin, E_th, W, |E - E0| / |E0|, max_d |dP_d| / sqrt(M E_gas))] at `times`."""
    from luna.ic_generation.ic import initial_conditions
    cfg = cfg_for(n, scheme)
    U0 = np.asarray(initial_conditions(cfg))
    e0 = sum(energies(U0, n))
    pscale = np.sqrt(U0[0].sum() * U0[4].sum())
    out = []
    faces = FACES_MC.get(n) if scheme == "hllc-mc" else None
    for t, U in zip(times, evolve_staged(cfg, U0, list(times), faces=faces)):
        k, th, w = energies(U, n)
        dp = max(abs(U[d].sum() - U0[d].sum()) for d in (1, 2, 3)) / pscale
        out.append((t, k, th, w, abs(k + th + w - e0) / abs(e0), dp))
    return out


budget = functools.lru_cache(maxsize=None)(run_budget)


@pytest.mark.parametrize("scheme,n", list(BOUNDS), ids=[f"{sc}-{n}" for sc, n in BOUNDS])
def test_evrard_energy(scheme, n):
    """|E(t) - E0| / |E0| at every output time; mass to round-off (in evolve_staged)."""
    for (t, *_, err, dp), b in zip(budget(n, scheme), BOUNDS[scheme, n]):
        check(f"evrard {scheme} N={n} t={t} |dE/E0|", err, b)
        check(f"evrard {scheme} N={n} t={t} |dP|/sqrt(ME)", dp, P_TOL[scheme, n])


@pytest.mark.parametrize("scheme", list(ORDER))
def test_evrard_energy_order(scheme):
    """The t = 3 error falls at first order, 32 -> 64."""
    order = np.log2(budget(32, scheme)[-1][4] / budget(64, scheme)[-1][4])
    check(f"evrard {scheme} order", order, ORDER[scheme][0], lower=True)
    check(f"evrard {scheme} order", order, ORDER[scheme][1])


def test_bound_catches_an_uncentred_kick(monkeypatch):
    """Teeth: energy work from the old momentum only (measured 0.0334 at t = 0.5, 32)."""
    import jax.numpy as jnp
    import luna.gravity.backend as backend

    def kick(U, g, s, dt, extra=None):
        gx, gy, gz = g[0] * s, g[1] * s, g[2] * s
        E = U[-1] + dt * (U[1] * gx + U[2] * gy + U[3] * gz)
        return jnp.stack([U[0], U[1] + dt * U[0] * gx, U[2] + dt * U[0] * gy, U[3] + dt * U[0] * gz, E])
    monkeypatch.setattr(backend, "kick_gas_body", kick)
    assert run_budget(32, "hllc-minmod", times=TIMES[:2])[1][4] > BOUNDS["hllc-minmod", 32][1]


@gpu_only
@pytest.mark.parametrize("scheme", ("hllc-minmod", "hll-minmod", "hllc-mc"))
def test_evrard_energy_gpu(scheme):
    """64, 128, 256 on a GPU: the t = 3 error falls with N (bounds once measured on viper)."""
    errs = [budget(n, scheme)[-1][4] for n in (64, 128, 256)]
    print(f"MEASURED evrard {scheme} t=3 |dE/E0| 64,128,256 = {errs}")
    assert errs[0] > errs[1] > errs[2]
