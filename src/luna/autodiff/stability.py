"""Matrix-free linear stability of a rollout from its tangent propagator M = dU(T)/dU0.

See docs/AUTODIFF.md, "Linear stability".
"""
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from luna.hydro.conversions import cons_to_prim


def global_sum(like):
    """Return a global sum for arrays laid out like `like` (shard_map psum when sharded)."""
    sh = getattr(like, "sharding", None)
    if not (isinstance(sh, NamedSharding) and sh.mesh.size > 1):
        return jnp.sum
    from luna.utils.shmap import shard_map
    axes = []
    for a in sh.spec:
        axes += [] if a is None else ([a] if isinstance(a, str) else list(a))
    return shard_map(lambda x: jax.lax.psum(jnp.sum(x), tuple(axes)),
                     mesh=sh.mesh, in_specs=(sh.spec,), out_specs=P())


def chu_norm(U, gamma):
    """Linear maps (to_z, from_z) between dU and z, with sum(z**2) = 2 x Chu (1965) energy."""
    W = cons_to_prim(U, gamma)
    rho, v, p = W[0], W[1:-1], W[-1]
    ke = 0.5 * jnp.sum(v * v, axis=0)
    a_rho, a_v, a_s = jnp.sqrt(p) / rho, jnp.sqrt(rho), jnp.sqrt(p / (gamma - 1.0))

    def to_z(dU):
        drho = dU[0]
        dv = (dU[1:-1] - v * drho) / rho
        dp = (gamma - 1.0) * (dU[-1] - jnp.sum(v * dU[1:-1], axis=0) + ke * drho)
        return jnp.concatenate([(a_rho * drho)[None], a_v * dv,
                                (a_s * (dp / p - drho / rho))[None]])

    def from_z(z):
        drho = z[0] / a_rho
        dv = z[1:-1] / a_v
        dp = p * (z[-1] / a_s + drho / rho)
        dm = rho * dv + v * drho
        dE = dp / (gamma - 1.0) + ke * drho + rho * jnp.sum(v * dv, axis=0)
        return jnp.concatenate([drho[None], dm, dE[None]])

    return to_z, from_z


def random_like(U, seed=0):
    """Unit-variance normal array with U's shape, dtype and sharding."""
    sh = getattr(U, "sharding", None)
    gen = jax.jit(lambda k: jax.random.normal(k, U.shape, U.dtype),
                  out_shardings=sh if isinstance(sh, NamedSharding) else None)
    return gen(jax.random.PRNGKey(seed))


def _tangent_map(prop, gamma):
    """B(z; U0, UT) = to_z_T(prop(U0 + from_z_0(z))), M in Chu-norm coordinates."""
    # States are arguments, not closure constants, which would be baked into the program.
    def B(z, U0, UT):
        _, from_in = chu_norm(U0, gamma)
        to_out, _ = chu_norm(UT, gamma)
        return to_out(prop(U0 + from_in(z)))
    return B


def _orthonormalize(V, total):
    """Orthonormal basis of span(V) by Cholesky-QR twice."""
    for _ in range(2):
        G = np.array([[float(total(a * b)) for b in V] for a in V])
        Li = np.linalg.inv(np.linalg.cholesky(G))
        V = [sum(Li[i, j] * V[j] for j in range(i + 1)) for i in range(len(V))]
    return V


def _combine(V, c):
    return sum(float(ci) * v for ci, v in zip(c, V))


def _subspace_iteration(apply, U0, n_vec, n_iter, rtol, seed, symmetric, log, tag):
    """Block power iteration with Rayleigh-Ritz: (Ritz values by modulus, vectors, top history)."""
    total = global_sum(U0)
    Q = _orthonormalize([random_like(U0, seed + i) for i in range(n_vec)], total)
    hist = []
    for it in range(n_iter):
        W = [apply(q) for q in Q]
        H = np.array([[float(total(q * w)) for w in W] for q in Q])
        theta, Y = np.linalg.eigh(0.5 * (H + H.T)) if symmetric else np.linalg.eig(H)
        order = np.argsort(-np.abs(theta))
        theta, Y = theta[order], Y[:, order]
        hist.append(float(np.abs(theta[0])))
        log(f"  {tag} iter {it:3d}: " + " ".join(f"{abs(t):.10e}" for t in theta))
        done = it > 0 and abs(hist[-1] - hist[-2]) <= rtol * hist[-1]
        if done or it == n_iter - 1:
            break
        Q = _orthonormalize(W, total)
    vecs = [_combine(Q, np.real(Y[:, i])) for i in range(n_vec)]
    return theta, vecs, hist


def optimal_perturbation(prop, U0, gamma, n_iter=40, rtol=1e-8, seed=0, n_vec=4, log=print):
    """Leading Chu-norm singular values of M via B^T B: dict(s, singular_values, z_in, z_out, history)."""
    B = _tangent_map(prop, gamma)
    total = global_sum(U0)
    UT = jax.jit(prop)(U0)

    @jax.jit
    def jvp(z, U0, UT):
        return jax.jvp(lambda z: B(z, U0, UT), (jnp.zeros_like(z),), (z,))[1]

    @jax.jit
    def btb(z, U0, UT):
        f = lambda z: B(z, U0, UT)
        zero = jnp.zeros_like(z)
        return jax.vjp(f, zero)[1](jax.jvp(f, (zero,), (z,))[1])[0]

    theta, vecs, hist = _subspace_iteration(lambda q: btb(q, U0, UT), U0, n_vec, n_iter,
                                            rtol, seed, True, log, "svd s^2")
    sv = np.sqrt(np.abs(theta))
    z_in = vecs[0] / jnp.sqrt(total(vecs[0] ** 2))
    z_out = jvp(z_in, U0, UT) / sv[0]
    return dict(s=float(sv[0]), singular_values=sv, z_in=z_in, z_out=z_out,
                history=[float(np.sqrt(h)) for h in hist])


def leading_mode(prop, U0, gamma, n_iter=40, rtol=1e-8, seed=0, n_vec=4, log=print):
    """Leading eigenvalues of M (JVP only; for a steady base state): dict(lam, eigenvalues, z, history)."""
    B = _tangent_map(prop, gamma)
    total = global_sum(U0)
    UT = jax.jit(prop)(U0)

    @jax.jit
    def jvp(z, U0, UT):
        return jax.jvp(lambda z: B(z, U0, UT), (jnp.zeros_like(z),), (z,))[1]

    theta, vecs, hist = _subspace_iteration(lambda q: jvp(q, U0, UT), U0, n_vec, n_iter,
                                            rtol, seed, False, log, "eig |lambda|")
    z = vecs[0] / jnp.sqrt(total(vecs[0] ** 2))
    return dict(lam=float(np.abs(theta[0])), eigenvalues=theta, z=z, history=hist)


def nonlinear_growth(prop, U0, gamma, z, eps, n_apply):
    """Nonlinear Chu-norm response to U0 + eps from_z(z) after each of n_apply steps, over eps |z|."""
    from_z = chu_norm(U0, gamma)[1]
    total = global_sum(U0)
    step = jax.jit(prop)
    norm = jax.jit(lambda a, b, U0: jnp.sqrt(total(chu_norm(U0, gamma)[0](a - b) ** 2)))
    z_norm = float(jnp.sqrt(total(z * z)))
    Ub, Up = U0, U0 + eps * from_z(z)
    amp = []
    for _ in range(n_apply):
        Ub, Up = step(Ub), step(Up)
        amp.append(float(norm(Up, Ub, U0)) / (eps * z_norm))
    return amp


def tangent_vs_difference(prop, U0, gamma, z, hs):
    """Relative Chu-norm error |D_h - M z| / |M z| of the tangent against a difference, per h."""
    to_z, from_z = chu_norm(U0, gamma)
    total = global_sum(U0)
    step = jax.jit(prop)
    UT = step(U0)
    Mz = jax.jit(lambda U, z: jax.jvp(prop, (U,), (from_z(z),))[1])(U0, z)
    norm = jax.jit(lambda d, U0: jnp.sqrt(total(chu_norm(U0, gamma)[0](d) ** 2)))
    ref = float(norm(Mz, U0))
    return [float(norm((step(U0 + h * from_z(z)) - UT) / h - Mz, U0)) / ref for h in hs]


def growth_rate(factor, horizon):
    """E-folding rate ln(factor) / horizon."""
    return math.log(factor) / horizon
