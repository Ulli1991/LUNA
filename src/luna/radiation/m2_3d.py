"""Ten-moment 3-D M2 with the exact maximum-entropy closure (reference and training target).

psi(n) = exp(a0 + a.n + n.B.n), eight multipliers per cell by damped Newton.
See docs/RADIATION.md.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

# sphere quadrature: Gauss-Legendre in cos(theta) times uniform in phi
_NT, _NP = 24, 48


def _quadrature(nt=_NT, npp=_NP):
    """Return sphere nodes n (3, Q) and weights w (Q,)."""
    mu, wmu = np.polynomial.legendre.leggauss(nt)
    phi = (np.arange(npp) + 0.5) * 2.0 * np.pi / npp
    st = np.sqrt(1.0 - mu ** 2)
    n = np.stack([np.outer(st, np.cos(phi)).ravel(),
                  np.outer(st, np.sin(phi)).ravel(),
                  np.outer(mu, np.ones(npp)).ravel()])
    w = np.outer(wmu, np.full(npp, 2.0 * np.pi / npp)).ravel()
    return n, w


# numpy, not jnp: built inside a trace (the stage imports this module lazily) jnp tables are tracers
_N, _W = _quadrature()


def basis(n):
    """Return the eight basis functions v_k(n) the multipliers act on; n: (3, Q)."""
    x, y, z = n[0], n[1], n[2]
    xp = np if isinstance(n, np.ndarray) else jnp
    return xp.stack([x, y, z,
                     x * x - y * y, 3.0 * z * z - 1.0,
                     x * y, x * z, y * z])


_V = basis(_N)                                   # (8, Q)


def normalised_moments(a):
    """Return (<v> (8,), <n n n> (3,3,3), log m0) under psi = exp(a.v)."""
    z = a @ _V
    zmax = jnp.max(z)
    w = jnp.exp(z - zmax) * _W
    s = jnp.sum(w)
    v = (_V @ w) / s
    nnn = jnp.einsum("iq,jq,kq,q->ijk", _N, _N, _N, w) / s
    return v, nnn, zmax + jnp.log(s)


_SYM = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
EPS = 1.0e-300


def pack(m0, m1, m2):
    """Pack m0 (...), m1 (3, ...), m2 (3, 3, ...) into the (10, ...) state."""
    return jnp.concatenate([m0[None], m1,
                            jnp.stack([m2[i, j] for i, j in _SYM])])


def tensor(U):
    """Return the symmetric second moment as (3, 3, ...)."""
    c = {(i, j): U[4 + k] for k, (i, j) in enumerate(_SYM)}
    return jnp.stack([jnp.stack([c[(min(i, j), max(i, j))] for j in range(3)])
                      for i in range(3)])


def density_scale(c):
    """Return max(m0, EPS, |c_1|/2, ..., |c_9|/2) of ten components: m0 wherever the state is realizable.

    Realizable means |c_k| <= m0, so the halving keeps m0 (bitwise) on states realizable to round-off,
    and every component over the scale stays within [-2, 2]: m1/m0 of an unfloored state overflows.
    """
    big = jnp.abs(c[1])
    for k in range(2, 10):
        big = jnp.maximum(big, jnp.abs(c[k]))
    return jnp.maximum(jnp.maximum(c[0], EPS), 0.5 * big)


def normalised(c):
    """Return the eight <v_k> of ten components c (a list), normalised by `density_scale`."""
    s = density_scale(c)
    t = [c[4 + k] / s for k in range(6)]
    return [c[1 + i] / s for i in range(3)] + [t[0] - t[1], 3.0 * t[2] - 1.0, t[3], t[4], t[5]]


def targets(U):
    """Return the eight normalised targets <v_k> from the ten-component state."""
    return jnp.stack(normalised([U[k] for k in range(10)]))


def _residual(a, u):
    return normalised_moments(a)[0] - u


def _potential(a, u):
    return normalised_moments(a)[2] - jnp.dot(a, u)


_BACKTRACK = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])


def solve_multipliers(u, iters=40):
    """Damped Newton on the convex potential, from the isotropic start."""
    def body(a, _):
        r = _residual(a, u)
        J = jax.jacfwd(_residual)(a, u)
        d = jnp.linalg.solve(J + 1e-12 * jnp.eye(8), r)
        cand = a[None, :] - _BACKTRACK[:, None] * d[None, :]
        vals = jax.vmap(lambda c: _potential(c, u))(cand)
        vals = jnp.where(jnp.isfinite(vals), vals, jnp.inf)
        return cand[jnp.argmin(vals)], None

    a, _ = jax.lax.scan(body, jnp.zeros(8), None, length=iters)
    return a


def closure_one(u, iters=40):
    """m3_ijk / m0 for one state's normalised moments."""
    return normalised_moments(solve_multipliers(u, iters))[1]


def third_moment(U, iters=40, chunk=0):
    """Return m3_ijk for a field by the entropy solve in every cell.

    `chunk`: solve in blocks via lax.map (works around a ROCm segfault on large programs).
    """
    m0 = jnp.maximum(U[0], 1e-300)
    u = targets(U)                                        # (8, ...)
    flat = u.reshape(8, -1).T                             # (N, 8)
    solve = jax.vmap(lambda p: closure_one(p, iters))
    if chunk and flat.shape[0] > chunk:
        n = flat.shape[0]
        pad = (-n) % chunk
        padded = jnp.concatenate([flat, jnp.zeros((pad, 8))]) if pad else flat
        blocks = padded.reshape(-1, chunk, 8)
        q = jax.lax.map(solve, blocks).reshape(-1, 3, 3, 3)[:n]
    else:
        q = solve(flat)                                   # (N, 3, 3, 3)
    q = jnp.moveaxis(q, 0, 3).reshape(3, 3, 3, *m0.shape)
    return q * m0


def realizable(U):
    """Project to m0 >= 0, |m1| <= m0, and m2/m0 with spectrum in [0, 1] summing to 1."""
    m0 = jnp.maximum(U[0], 0.0)
    safe = jnp.maximum(m0, EPS)
    m1 = U[1:4]
    n1 = jnp.sqrt(jnp.sum(m1 ** 2, axis=0) + EPS) / safe
    m1 = m1 * jnp.minimum(1.0, 1.0 / jnp.maximum(n1, EPS))

    T = jnp.moveaxis(tensor(U) / safe, (0, 1), (-2, -1))     # (..., 3, 3)
    T = 0.5 * (T + jnp.swapaxes(T, -1, -2))
    P = _clip_spectrum(jax.lax.stop_gradient(T))
    # stopped correction: eigh's gradient is NaN at degenerate (isotropic) cells
    T = T + jax.lax.stop_gradient(P - T)
    return pack(m0, m1, jnp.moveaxis(T, (-2, -1), (0, 1)) * safe)


def _clip_spectrum(T):
    """Return the symmetric matrix with eigenvalues clipped to [0, 1] and renormalised to sum 1."""
    w, V = jnp.linalg.eigh(T)
    w = jnp.maximum(w, 0.0)
    s = jnp.sum(w, axis=-1, keepdims=True)
    w = jnp.where(s > 0.0, w / jnp.maximum(s, EPS), 1.0 / 3.0)
    return jnp.einsum("...ij,...j,...kj->...ik", V, w, V)


def flux(U, d, chat, closure_fn=None):
    """Return the flux of the ten moments along axis d: c (m1_d, m2_di, m3_dij)."""
    m2 = tensor(U)
    m3 = (third_moment if closure_fn is None else closure_fn)(U)

    return chat * jnp.concatenate([
        U[1 + d][None], m2[d],
        jnp.stack([m3[d, i, j] for i, j in _SYM])])


def _minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a),
                                                          jnp.abs(b)), 0.0)


def axis_update(U, d, dt, dx, *, chat, boundary=1, closure_fn=None,
                diffusion=1.0):
    """Return -dt/dx times the minmod-MUSCL GLF flux difference along axis d."""
    ax = 1 + d
    mode = "wrap" if boundary == 0 else "edge"
    pad = [(0, 0)] + [((2, 2) if a == ax else (0, 0)) for a in (1, 2, 3)]
    C = jnp.pad(U, pad, mode=mode)

    def sl(a, b):
        return tuple(slice(None) if k != ax else
                     slice(a, b if b != 0 else None) for k in range(4))

    s = _minmod(C[sl(1, -1)] - C[sl(0, -2)], C[sl(2, 0)] - C[sl(1, -1)])
    Uc = C[sl(1, -1)]
    L, R = (Uc + 0.5 * s)[sl(0, -1)], (Uc - 0.5 * s)[sl(1, 0)]
    face = (0.5 * (flux(L, d, chat, closure_fn) + flux(R, d, chat, closure_fn))
            - 0.5 * diffusion * chat * (R - L))
    tr = tuple(slice(None) if k != ax else slice(1, None) for k in range(4))
    tl = tuple(slice(None) if k != ax else slice(0, -1) for k in range(4))
    return -dt / dx * (face[tr] - face[tl])


def transport_step(U, dt, dx, *, chat, boundary=1, closure_fn=None,
                   diffusion=1.0):
    """One unsplit minmod-MUSCL + GLF step over the three axes."""
    kw = dict(chat=chat, boundary=boundary, closure_fn=closure_fn,
              diffusion=diffusion)
    return U + sum(axis_update(U, d, dt, dx, **kw) for d in range(3))


def step(U, dt, dx, *, chat, boundary=1, closure_fn=None, diffusion=1.0):
    """One SSP-RK2 step, projected to the realizable set; `diffusion` scales the LF dissipation."""
    kw = dict(chat=chat, boundary=boundary, closure_fn=closure_fn,
              diffusion=diffusion)
    U1 = transport_step(U, dt, dx, **kw)
    U1 = 0.5 * (U + transport_step(U1, dt, dx, **kw))
    return realizable(U1)


def make_split_step(dt, dx, *, chat, boundary=1, closure_fn=None,
                    diffusion=1.0):
    """Return `step` as a function of U with each axis jitted separately (ROCm segfault workaround)."""
    kw = dict(chat=chat, boundary=boundary, closure_fn=closure_fn,
              diffusion=diffusion)
    upd = [jax.jit(functools.partial(axis_update, d=d, dt=dt, dx=dx, **kw))
           for d in range(3)]
    fin = jax.jit(realizable)

    def transport(U):
        return U + sum(f(U) for f in upd)

    def one(U):
        return fin(0.5 * (U + transport(transport(U))))

    return one


def beam(E, direction):
    """Return the ten moments of a beam: m1 = E u, m2 = E u(x)u."""
    u = jnp.asarray(direction, dtype=jnp.float64)
    u = u / jnp.linalg.norm(u)
    m1 = jnp.stack([E * u[i] for i in range(3)])
    m2 = jnp.stack([jnp.stack([E * u[i] * u[j] for j in range(3)])
                    for i in range(3)])
    return pack(E, m1, m2)


def add(*states):
    """Superpose states by adding their moments."""
    out = states[0]
    for v in states[1:]:
        out = out + v
    return out
