"""Linear advection of a scalar by a prescribed, divergence-free 2D velocity: LUNA's MUSCL-Hancock
(the same limited slopes, one cell-centred predictor, one upwind flux per face) without the Euler
dynamics, for advection benchmarks such as the rotating slotted cylinder."""

import jax
import jax.numpy as jnp

from luna.hydro.slopes import compute_slopes_from_padded
from luna.time_integration.integrators import cell_dt

BOUNDARIES = ("zero", "periodic")


def make_advection_2d(n, lo, hi, velocity, limiter=0, cfl=0.8, boundary="zero"):
    """(compute_dt(u), step(u, dt)) on an n x n grid over [lo, hi]^2; u has shape (n, n).

    `velocity(x, y) -> (vx, vy)` must be divergence-free (the predictor drops u div v) with vx
    independent of x and vy of y, so a face value is that face's average. "zero" holds the
    ghosts at 0 (a Dirichlet boundary), "periodic" wraps.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary={boundary!r}: one of {BOUNDARIES}")
    h = (hi - lo) / n
    c = lo + (jnp.arange(-1, n + 1) + 0.5) * h          # centres of the interior + one ghost ring
    f = lo + jnp.arange(n + 1) * h                       # faces of the interior
    X, Y = jnp.meshgrid(c, c, indexing="ij")
    vx_c, vy_c = velocity(X, Y)
    vx_f = velocity(f[:, None], c[None, 1:-1])[0]        # x-faces (n+1, n)
    vy_f = velocity(c[1:-1, None], f[None, :])[1]        # y-faces (n, n+1)
    mode = "constant" if boundary == "zero" else "wrap"

    def compute_dt(u):
        return jnp.min(cell_dt([vx_c[1:-1, 1:-1], vy_c[1:-1, 1:-1]], 0.0, (h, h), cfl))

    def step(u, dt):
        up = jnp.pad(u, 2, mode=mode)[None]                         # two ghosts
        dux, duy, _ = compute_slopes_from_padded(up, limiter)       # interior + one ring
        uc = up[:, 1:-1, 1:-1]
        half = uc - 0.5 * dt / h * (vx_c * dux + vy_c * duy)        # Hancock predictor
        xl, xr = (half + 0.5 * dux)[0, :-1, 1:-1], (half - 0.5 * dux)[0, 1:, 1:-1]
        yl, yr = (half + 0.5 * duy)[0, 1:-1, :-1], (half - 0.5 * duy)[0, 1:-1, 1:]
        fx = vx_f * jnp.where(vx_f > 0, xl, xr)                     # upwind: the exact Riemann flux
        fy = vy_f * jnp.where(vy_f > 0, yl, yr)
        return u - dt / h * (fx[1:] - fx[:-1] + fy[:, 1:] - fy[:, :-1])

    return jax.jit(compute_dt), jax.jit(step)
