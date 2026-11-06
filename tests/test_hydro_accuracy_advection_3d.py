"""A density wave advected uniformly along the 3D diagonal for one period: the exact solution is the initial state."""
import jax
import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve, gpu_only

# {(limiter, solver): {N: (L1, L2, Linf)}} per AMP, ~1.3x measured (FP64, default dt; CPU-8 and MI300A equal at 32, 64).
# Measured 32-256: minmod L1 0.0836, 0.0372, 0.0102, 0.00284; MC 0.0226, 0.00549, 0.00127, 0.000288;
# van Leer 0.0455, 0.0114, 0.00270, 0.000620; HLL minmod 0.0876, 0.0398, 0.0111, 0.00308.
BOUNDS = {
    (0, "hllc"): {32: (0.11, 0.14, 0.27), 64: (0.049, 0.058, 0.115), 128: (0.0134, 0.0183, 0.049), 256: (0.0037, 0.0059, 0.02)},
    (1, "hllc"): {32: (0.03, 0.038, 0.087), 64: (0.0072, 0.0104, 0.03), 128: (0.00165, 0.0028, 0.01), 256: (0.00038, 0.00076, 0.0034)},
    (2, "hllc"): {32: (0.06, 0.07, 0.135), 64: (0.015, 0.021, 0.052), 128: (0.0036, 0.006, 0.019), 256: (0.00081, 0.0018, 0.007)},
    (0, "hll"): {32: (0.114, 0.147, 0.283), 64: (0.052, 0.062, 0.126), 128: (0.0144, 0.02, 0.054), 256: (0.004, 0.0063, 0.022)},
}
# 128 -> 256 orders, L1 then L2, measured -0.1 .. +0.3.
ORDERS = {(0, "hllc"): ((1.75, 2.15), (1.54, 1.94)), (1, "hllc"): ((2.03, 2.43), (1.79, 2.19)),
          (2, "hllc"): ((2.02, 2.42), (1.71, 2.11)), (0, "hll"): ((1.74, 2.14), (1.54, 1.94))}
ROUND_OFF = 1e-13                               # p and v stay uniform, and rho keeps the wave's x <-> y <-> z symmetry
IDS = [f"{'minmod mc vanleer'.split()[lim]}-{s}" for lim, s in BOUNDS]

GAMMA = 5.0 / 3.0
AMP = 0.2
P0 = 1.0 / GAMMA                                # c_s = 1 at rho = 1
V = 1.0                                         # u = v = w = V: the wave is back after t = 1


def initial(n):
    """U on n^3 cells, rho = 1 + AMP sin(2 pi (x + y + z)) point-sampled, p and v uniform."""
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = 1.0 + AMP * np.sin(2.0 * np.pi * (X + Y + Z))
    return np.stack([rho, rho * V, rho * V, rho * V, P0 / (GAMMA - 1.0) + 1.5 * rho * V * V])


def advect(n, limiter=0, solver="hllc", order=1, mesh_shape=None):
    """(U0, U) after one period."""
    cfg = accuracy_cfg(test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=1.0, Ly=1.0, Lz=1.0,
                       BOUNDARY=0, gamma=GAMMA, ORDER=order, LIMITER=limiter, riemann_solver=solver)
    U0 = initial(n)
    return U0, np.asarray(evolve(cfg, U0, 1.0, mesh_shape=mesh_shape)[0])


def errors(U0, U):
    """L1, L2, Linf of rho - rho_0 per AMP, and max |p / P0 - 1|, max |u_i / V - 1|."""
    e = np.abs(U[0] - U0[0]) / AMP
    vel = U[1:4] / U[0]
    p = (GAMMA - 1.0) * (U[4] - 0.5 * U[0] * (vel ** 2).sum(0))
    return dict(L1=float(e.mean()), L2=float(np.sqrt((e ** 2).mean())), Linf=float(e.max()),
                p=float(np.abs(p / P0 - 1.0).max()), v=float(np.abs(vel / V - 1.0).max()))


def checked(n, limiter, solver):
    """Run, check every bound at n, the uniform p and v and the axes; return the errors."""
    tag = f"advect3d {limiter} {solver} N={n}"
    U0, U = advect(n, limiter, solver)
    e = errors(U0, U)
    for k, bound in zip(("L1", "L2", "Linf"), BOUNDS[limiter, solver][n]):
        check(f"{tag} {k}", e[k], bound)
    check(f"{tag} max |p / p0 - 1|", e["p"], ROUND_OFF)
    check(f"{tag} max |v / V - 1|", e["v"], ROUND_OFF)
    for axes in ((1, 0, 2), (2, 1, 0)):
        check(f"{tag} rho swapped {axes}", float(np.abs(U[0] - U[0].transpose(axes)).max()), ROUND_OFF)
    return e


@pytest.mark.parametrize("limiter,solver", list(BOUNDS), ids=IDS)
def test_advection_cpu(limiter, solver):
    """N = 32, 64: L1, L2, Linf against the exact solution; p, v uniform and the axes alike to round-off."""
    for n in (32, 64):
        checked(n, limiter, solver)


@gpu_only
@pytest.mark.parametrize("limiter,solver", list(BOUNDS), ids=IDS)
def test_advection_gpu(limiter, solver):
    """N = 128, 256: the same, and the L1 and L2 orders."""
    a, b = checked(128, limiter, solver), checked(256, limiter, solver)
    for k, (lo, hi) in zip(("L1", "L2"), ORDERS[limiter, solver]):
        o = float(np.log2(a[k] / b[k]))
        check(f"advect3d {limiter} {solver} {k} order", o, lo, lower=True)
        check(f"advect3d {limiter} {solver} {k} order", o, hi)


@pytest.mark.skipif(jax.device_count() < 2, reason="needs 2 devices")
def test_two_devices_are_bitwise_one():
    """Split along x, the state after one period equals one device's in every bit."""
    np.testing.assert_array_equal(advect(32, mesh_shape=(2, 1, 1))[1], advect(32)[1])


def test_bound_catches_first_order():
    """Godunov (L1 0.433 at N = 32) fails the minmod bound: the ladder has teeth."""
    with pytest.raises(AssertionError, match="L1"):
        e = errors(*advect(32, order=0))
        check("advect3d first order L1", e["L1"], BOUNDS[0, "hllc"][32][0])
