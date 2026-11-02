"""Slab-geometry M2 radiation: maximum-entropy closure for m3, differentiated implicitly.

psi(mu) = exp(a0 + a1 mu + a2 mu^2); multipliers by damped Newton on a convex potential.
See docs/RADIATION.md.
"""
import jax
import jax.numpy as jnp
import numpy as np

# Gauss-Legendre on [-1, 1]
_N_QUAD = 64
_MU = jnp.asarray(np.polynomial.legendre.leggauss(_N_QUAD)[0])
_W = jnp.asarray(np.polynomial.legendre.leggauss(_N_QUAD)[1])

# multipliers diverge on the realizable boundary f^2 <= g <= 1: stay a hair inside
REALIZABLE_EPS = 1.0e-6


def normalised_moments(ab):
    """Return ((m1, m2, m3)/m0, log m0) of exp(a mu + b mu^2), max-shifted against overflow."""
    z = ab[0] * _MU + ab[1] * _MU * _MU
    zmax = jnp.max(z)
    w = jnp.exp(z - zmax) * _W
    s = jnp.sum(w)
    return (jnp.stack([jnp.sum(_MU * w), jnp.sum(_MU ** 2 * w),
                       jnp.sum(_MU ** 3 * w)]) / s, zmax + jnp.log(s))


def _residual(ab, fg):
    """Return grad Psi: normalised (m1, m2) minus the target (f, g)."""
    return normalised_moments(ab)[0][:2] - fg


def _potential(ab, fg):
    """Psi(a, b) = log m0 - a f - b g, convex, minimised at the solution."""
    return normalised_moments(ab)[1] - jnp.dot(ab, fg)


def clamp(f, g):
    """Push (f, g) a hair inside the realizable set f^2 <= g <= 1."""
    f = jnp.clip(f, -1.0 + REALIZABLE_EPS, 1.0 - REALIZABLE_EPS)
    return f, jnp.clip(g, f * f + REALIZABLE_EPS, 1.0 - REALIZABLE_EPS)


_BACKTRACK = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])


def solve_multipliers(fg, iters=30):
    """Return multipliers (a, b) by fixed-count damped Newton on Psi from the isotropic start."""
    def body(ab, _):
        r = _residual(ab, fg)
        J = jax.jacfwd(_residual)(ab, fg)
        d = jnp.linalg.solve(J + 1e-13 * jnp.eye(2), r)
        cand = ab[None, :] - _BACKTRACK[:, None] * d[None, :]
        vals = jax.vmap(lambda c: _potential(c, fg))(cand)
        vals = jnp.where(jnp.isfinite(vals), vals, jnp.inf)
        return cand[jnp.argmin(vals)], None

    ab, _ = jax.lax.scan(body, jnp.zeros(2), None, length=iters)
    return ab


def _third(ab):
    return normalised_moments(ab)[0][2]


@jax.custom_jvp
def closure(fg):
    """m3/m0 of the maximum-entropy distribution with these (f, g)."""
    return _third(solve_multipliers(fg))


@closure.defjvp
def _closure_jvp(primals, tangents):
    """Implicit derivative dh = (dh/da) (dr/da)^-1 d(f,g); a JVP so JAX transposes it for reverse mode."""
    (fg,), (dfg,) = primals, tangents
    ab = solve_multipliers(fg)
    J = jax.jacfwd(_residual)(ab, fg)          # dr/da, symmetric positive
    dh_da = jax.grad(_third)(ab)
    return _third(ab), jnp.dot(dh_da, jnp.linalg.solve(J, dfg))


def third_moment(U, chunk=0):
    """Return m3 of U = (m0, m1, m2) for a cell or a grid.

    `chunk`: solve in blocks via lax.map (works around a ROCm segfault with many solves).
    """
    m0 = jnp.maximum(U[0], 1e-300)
    f, g = clamp(U[1] / m0, U[2] / m0)
    fg = jnp.stack([f.reshape(-1), g.reshape(-1)], axis=1)
    n = fg.shape[0]
    if chunk and n > chunk:
        nb = -(-n // chunk)
        pad = nb * chunk - n
        if pad:
            fg = jnp.concatenate([fg, jnp.zeros((pad, 2))])
        h = jax.lax.map(jax.vmap(closure), fg.reshape(nb, chunk, 2))
        h = h.reshape(-1)[:n]
    else:
        h = jax.vmap(closure)(fg)
    return m0 * h.reshape(f.shape)


def flux(U, chat, closure_fn=None):
    """Return the flux c (m1, m2, m3) along x; `closure_fn(U) -> m3` defaults to the exact solve."""
    m3 = (third_moment if closure_fn is None else closure_fn)(U)
    return chat * jnp.stack([U[1], U[2], m3])


def _minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a),
                                                          jnp.abs(b)), 0.0)


def realizable(U):
    """Clip the state to m0 >= 0, f^2 <= g <= 1."""
    m0 = jnp.maximum(U[0], 0.0)
    safe = jnp.maximum(m0, 1e-300)
    f, g = clamp(U[1] / safe, U[2] / safe)
    return jnp.stack([m0, m0 * f, m0 * g])


def transport_step(U, dt, dx, *, chat, boundary=1, order=2, closure_fn=None):
    """One explicit GLF step of the slab M2 system. U: (3, N)."""
    g = 2 if order == 2 else 1
    mode = "wrap" if boundary == 0 else "edge"
    C = jnp.pad(U, [(0, 0), (g, g)], mode=mode)
    if order == 2:
        slope = _minmod(C[:, 1:-1] - C[:, :-2], C[:, 2:] - C[:, 1:-1])
        Uc = C[:, 1:-1]
        UL = (Uc + 0.5 * slope)[:, :-1]
        UR = (Uc - 0.5 * slope)[:, 1:]
    else:
        UL, UR = C[:, :-1], C[:, 1:]
    face = (0.5 * (flux(UL, chat, closure_fn) + flux(UR, chat, closure_fn))
            - 0.5 * chat * (UR - UL))
    return U - dt / dx * (face[:, 1:] - face[:, :-1])


def step(U, dt, dx, *, chat, boundary=1, order=2, closure_fn=None):
    """One SSP-RK2 step, clipped to the realizable set."""
    kw = dict(chat=chat, boundary=boundary, order=order, closure_fn=closure_fn)
    U1 = transport_step(U, dt, dx, **kw)
    if order == 2:
        U1 = 0.5 * (U + transport_step(U1, dt, dx, **kw))
    return realizable(U1)


def entropy(fg):
    """Return the convex dual h(f, g) = a.(f, g) - log m0(a); its gradient is the multiplier map."""
    ab = solve_multipliers(fg)
    return jnp.dot(ab, fg) - normalised_moments(ab)[1]
