"""Validate solve_poisson_fmg_bidiff: forward (jvp/jacfwd) AND reverse
(vjp/grad) mode, vs central FD, and reverse-consistency with the validated
custom_vjp solver (solve_poisson_fmg_diff)."""

import pytest
pytestmark = pytest.mark.slow

import jax
import jax.numpy as jnp


from luna.gravity.poisson_diff import solve_poisson_fmg_diff, solve_poisson_fmg_bidiff

GRID = 32
DX = 1.0 / GRID
FOURPIG = 1.0
BC = 0   # periodic


def L_of_rho(rho, solver):
    phi = solver(rho, DX, FOURPIG, BC)
    return 0.5 * jnp.sum(phi ** 2)


def main():
    key = jax.random.PRNGKey(0)
    rho = jax.random.normal(key, (GRID, GRID, GRID), dtype=jnp.float64)
    rho = rho - jnp.mean(rho)   # solvable RHS for periodic BC

    f_bi = lambda r: L_of_rho(r, solve_poisson_fmg_bidiff)
    f_vjp = lambda r: L_of_rho(r, solve_poisson_fmg_diff)

    # reverse-mode gradients
    g_rev_bi = jax.grad(f_bi)(rho)
    g_rev_vjp = jax.grad(f_vjp)(rho)

    # forward-mode gradient (this is what fails on the custom_vjp solver)
    # grad via jvp: build full grad by jvp against basis is expensive; instead
    # check a directional derivative matches reverse + FD.
    key2 = jax.random.PRNGKey(1)
    v = jax.random.normal(key2, rho.shape, dtype=jnp.float64)
    v = v - jnp.mean(v)
    _, dir_fwd = jax.jvp(f_bi, (rho,), (v,))           # forward-mode directional deriv
    dir_rev = float(jnp.sum(g_rev_bi * v))             # reverse-mode directional deriv

    # central FD directional derivative
    h = 1e-6
    dir_fd = float((f_bi(rho + h * v) - f_bi(rho - h * v)) / (2 * h))

    print("Poisson bidiff — forward & reverse vs FD (periodic, 32^3)")
    print(f"  directional deriv:  jvp(fwd)={float(dir_fwd):.8e}")
    print(f"                      grad(rev)={dir_rev:.8e}")
    print(f"                      FD        ={dir_fd:.8e}")
    rel_fwd = abs(dir_fwd - dir_fd) / abs(dir_fd)
    rel_rev = abs(dir_rev - dir_fd) / abs(dir_fd)
    print(f"  forward vs FD rel = {rel_fwd:.2e}")
    print(f"  reverse vs FD rel = {rel_rev:.2e}")

    # reverse-mode consistency: bidiff vs the validated custom_vjp solver
    rel_cons = float(jnp.linalg.norm(g_rev_bi - g_rev_vjp) /
                     (jnp.linalg.norm(g_rev_vjp) + 1e-30))
    print(f"  reverse grad: bidiff vs custom_vjp rel = {rel_cons:.2e}")

    # jacfwd actually runs through bidiff (the headline: it RAISES on the
    # custom_vjp solver). Few-input map: 2 scalars scaling two density modes.
    def few_in(p):
        r = rho * (1.0 + p[0]) + v * p[1]
        phi = solve_poisson_fmg_bidiff(r, DX, FOURPIG, BC)
        return jnp.array([0.5 * jnp.sum(phi ** 2), jnp.sum(phi)])
    Jf = jax.jacfwd(few_in)(jnp.zeros(2))
    print(f"  jacfwd ran through bidiff: J shape {Jf.shape}, "
          f"finite={bool(jnp.all(jnp.isfinite(Jf)))}")

    assert rel_fwd < 1e-4, "forward-mode wrong"
    assert rel_rev < 5e-3, "reverse-mode wrong"   # FMG residual-tolerance limited
    assert rel_cons < 1e-3, "reverse inconsistent with custom_vjp"
    print("\nALL POISSON BIDIFF CHECKS PASSED ✓")


def test_poisson_bidiff_fwd_rev():
    main()


if __name__ == "__main__":
    main()
