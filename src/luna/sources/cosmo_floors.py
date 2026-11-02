"""Stability floors of a gas run, applied once per step; the grid's density floor conserves mass.

`floor_core` is layout-generic: it serves the grid here and the AMR levels (luna/amr/cosmo.py).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from luna.units import KB_OVER_MP_KMS2_PER_K


def temp_coeff(a, T, mu):
    """Super-comoving pressure per density for a temperature T: a^2 kB T / (mu m_p)."""
    return a * a * KB_OVER_MP_KMS2_PER_K * T / mu if T > 0.0 else 0.0


def jeans_coeff(a, coeff_fac, n_cells, dx, rho_bar):
    """Truelove-Jeans floor coefficient cj(a) in P~ >= cj rho~^2 for n_cells Jeans cells."""
    if n_cells <= 0.0:
        return 0.0
    return coeff_fac * a * (n_cells * dx) ** 2 / (4.0 * np.pi ** 2 * rho_bar)


def floor_core(rho, mom, E, ct, ctmax, cj, *, gamma, rho_floor, floor_buffer, v_max, T_max,
               p_floor, sum_axis):
    """Apply density floor, |v~| ceiling, T ceiling and pressure floor; return (rho, mom, E).

    `mom` has its component axis at `sum_axis`; P~ >= max(p_floor, ct rho~, cj rho~^2).
    """
    if rho_floor > 0.0:
        bad = rho < rho_floor * floor_buffer
        eint_fl = jnp.maximum(ct * rho_floor / (gamma - 1.0), 1e-300)
        rho = jnp.where(bad, rho_floor, rho)
        mom = jnp.where(bad, 0.0, mom)
        E = jnp.where(bad, eint_fl, E)
    ke = 0.5 * jnp.sum(mom * mom, axis=sum_axis, keepdims=True) / rho
    eint = E - ke
    if v_max > 0.0:
        v2 = jnp.sum(mom * mom, axis=sum_axis, keepdims=True) / (rho * rho)
        scl = jnp.where(v2 > v_max * v_max, v_max / jnp.sqrt(v2 + 1e-300), 1.0)
        mom = mom * scl
        ke = 0.5 * jnp.sum(mom * mom, axis=sum_axis, keepdims=True) / rho
    if T_max > 0.0:
        eint = jnp.minimum(eint, ctmax * rho / (gamma - 1.0))
    P = (gamma - 1.0) * eint
    Pmin = jnp.maximum(jnp.maximum(p_floor, ct * rho), cj * rho * rho)
    eint = jnp.maximum(P, Pmin) / (gamma - 1.0)
    E = ke + jnp.maximum(eint, 1e-300)
    return rho, mom, E


def conservative_density_floor(rho, mom, E, ct, *, gamma, rho_floor, floor_buffer, boxsum, total):
    """Lift cells under the floor with mass from their 3x3x3 neighbours, any shortfall from the whole grid."""
    thr = rho_floor * floor_buffer
    bad = rho < thr

    def lift(args):
        rho, mom, E = args
        rho_n = jnp.maximum(rho, rho_floor)
        D = jnp.where(bad, rho_n - rho, 0.0)                 # what each floored cell needs
        S = jnp.where(bad, 0.0, rho - thr)                   # what each other cell can give
        T = boxsum(S)
        q = jnp.where(T > 0.0, D / jnp.where(T > 0.0, T, 1.0), 0.0)
        Q = boxsum(q)                                        # asked of each donor, per unit S
        s = jnp.where(Q > 1.0, 1.0 / jnp.where(Q > 1.0, Q, 1.0), 1.0)
        taken = S * jnp.minimum(Q, 1.0)
        short = D - q * boxsum(S * s)                        # left after the neighbours gave
        left = S - taken
        pool = total(left)
        phi = jnp.clip(total(short) / jnp.where(pool > 0.0, pool, 1.0), 0.0, 1.0)
        keep = 1.0 - (taken + phi * left) / jnp.where(bad, 1.0, rho)
        eint_fl = jnp.maximum(ct * rho_n / (gamma - 1.0), 1e-300)
        return (jnp.where(bad, rho_n, rho - (taken + phi * left)),
                jnp.where(bad, 0.0, mom * keep), jnp.where(bad, eint_fl, E * keep))

    return jax.lax.cond(total(bad.astype(rho.dtype)) > 0.0, lift, lambda a: a, (rho, mom, E))


def _local_ghosts(f, axis, periodic):
    """One-cell ghost slabs of `f` along `axis` from its own far side (periodic) or zeros."""
    lo = jax.lax.slice_in_dim(f, f.shape[axis] - 1, f.shape[axis], axis=axis)
    hi = jax.lax.slice_in_dim(f, 0, 1, axis=axis)
    return (lo, hi) if periodic else (jnp.zeros_like(lo), jnp.zeros_like(hi))


def _mesh_ghosts(f, axis, name, n_proc, periodic):
    """One-cell ghost slabs along a sharded `axis`, zero outside a non-periodic domain."""
    from luna.utils.halo_exchange_shardmap import _axis_ghosts
    lo, hi = _axis_ghosts(f, axis=axis, axis_name=name, n_proc=n_proc, periodic=True)
    if periodic:
        return lo, hi
    idx = jax.lax.axis_index(name)
    return jnp.where(idx == 0, 0.0, lo), jnp.where(idx == n_proc - 1, 0.0, hi)


def make_boxsum(mesh, periodic, ndim=3):
    """Return (boxsum, total): the 3x3x3 neighbourhood sum of a (1, ...) field and the grid sum."""
    names = tuple(mesh.axis_names) if mesh is not None and mesh.size > 1 else ()

    def ghosts(f, d):
        name = names[d] if d < len(names) else None
        if name is not None and int(mesh.shape[name]) > 1:
            return _mesh_ghosts(f, d + 1, name, int(mesh.shape[name]), periodic)
        return _local_ghosts(f, d + 1, periodic)

    def boxsum(f):
        for d in range(ndim):
            lo, hi = ghosts(f, d)
            g = jnp.concatenate([lo, f, hi], axis=d + 1)
            n = f.shape[d + 1]
            f = (jax.lax.slice_in_dim(g, 0, n, axis=d + 1) + f
                 + jax.lax.slice_in_dim(g, 2, n + 2, axis=d + 1))
        return f

    def total(f):
        s = jnp.sum(f)
        return jax.lax.psum(s, names) if names else s

    return boxsum, total


def floor_body(U, ct, ctmax, cj, *, rho_floor=0.0, floor_buffer=1.0, boxsum=None, total=None,
               **knobs):
    """Apply the floor stack to a (5, ...) conservative grid; the density floor conserves mass."""
    rho, mom, E = U[0:1], U[1:4], U[4:5]
    if rho_floor > 0.0:
        rho, mom, E = conservative_density_floor(
            rho, mom, E, ct, gamma=knobs["gamma"], rho_floor=rho_floor,
            floor_buffer=floor_buffer, boxsum=boxsum, total=total)
    rho, mom, E = floor_core(rho, mom, E, ct, ctmax, cj, sum_axis=0, rho_floor=0.0,
                             floor_buffer=floor_buffer, **knobs)
    return jnp.concatenate([rho, mom, E], axis=0)


def elementwise_program(mesh, body, n_scalars):
    """Return `body(U, *scalars)` jitted, as one shard_map over `mesh` when it spans several devices."""
    # One shard_map program: an eager op between shard_map kernels wedges the ROCm runtime.
    if mesh is None or mesh.size == 1:
        return jax.jit(body)
    from jax.sharding import PartitionSpec as P
    from luna.utils.shmap import shard_map
    axes = tuple(mesh.axis_names)
    in_specs = (P(None, *axes),) + (P(),) * n_scalars
    return jax.jit(shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=P(None, *axes)))


class CosmoFloors:
    """Floor stack as a once-per-step projection `floors(U, a)`."""

    def __init__(self, *, gamma, coeff_fac, dx, rho_bar, p_floor, jeans_ncell,
                 T_floor, T_max, v_max, mu, rho_floor_frac, floor_buffer, mesh=None,
                 periodic=True):
        self.gamma = float(gamma)
        self.coeff_fac = float(coeff_fac)
        self.dx = float(dx)
        self.rho_bar = float(rho_bar)
        self.p_floor = float(p_floor)
        self.jeans_N = float(jeans_ncell)
        self.T_floor = float(T_floor)
        self.T_max = float(T_max)
        self.v_max = float(v_max)
        self.mu = float(mu)
        self.rho_floor = float(rho_floor_frac) * self.rho_bar
        self.floor_buffer = float(floor_buffer)
        self.density_on = self.rho_floor > 0.0
        self.pressure_on = self.p_floor > 0.0 or self.jeans_N > 0.0 or self.T_floor > 0.0
        self.active = self.density_on or self.pressure_on
        boxsum, total = make_boxsum(mesh, periodic)
        self._floors = elementwise_program(mesh, partial(
            floor_body, gamma=self.gamma, rho_floor=self.rho_floor,
            floor_buffer=self.floor_buffer, v_max=self.v_max, T_max=self.T_max,
            p_floor=self.p_floor, boxsum=boxsum, total=total), 3)

    def jeans_coeff(self, a):
        return jeans_coeff(a, self.coeff_fac, self.jeans_N, self.dx, self.rho_bar)

    def temp_coeff(self, a):
        return temp_coeff(a, self.T_floor, self.mu)

    def tempmax_coeff(self, a):
        return temp_coeff(a, self.T_max, self.mu)

    def __call__(self, U, a):
        if not self.active:
            return U
        return self._floors(U, self.temp_coeff(a), self.tempmax_coeff(a), self.jeans_coeff(a))

    def describe(self):
        return (f"floors: rho={self.rho_floor:.3e} p={self.p_floor:.3e} T={self.T_floor} "
                f"Tmax={self.T_max} NJ={self.jeans_N} vmax={self.v_max}")
