"""Hydro-on-blocks: LUNA's MUSCL-Hancock solver applied per AMR block via ``vmap``.

Each block consumes a 1-ghost ``W_pad`` sliced from the wider halo; all boundary
physics lives in the halo fill (``boundary=1`` / ``exact_wrap=False``), and the
solver never reads the NaN corner ghosts. The plain ghost-based path does not
telescope across block seams: :func:`conservative_block_divergence` fixes
same-level seams, :mod:`amr.reflux` fixes coarse-fine faces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hydro.conversions import cons_to_prim, prim_to_cons
from hydro.muscl_hancock import (
    divergence_from_face_fluxes,
    muscl_hancock_boundary_states,
    muscl_hancock_divergence,
    muscl_hancock_face_fluxes,
    riemann_face_flux,
)


def _inner_ghost_index(ndim, g):
    """Index tuple that strips the outer ``g-1`` ghost layers of a block halo,
    leaving interior + a single ghost layer (the neighbour's edge cell) — the
    1-ghost ``W_pad`` the LUNA solver expects.
    """
    lo = g - 1
    sl = slice(lo, -lo) if lo > 0 else slice(None)
    return (slice(None), slice(None)) + (sl,) * ndim


def inv_kw(cfg, level):
    """``dict(inv_dx=..., inv_dy=..., inv_dz=...)`` from ``cfg.level_dx(level)``
    (1.0 for axes beyond ``cfg.ndim``) -- the kwargs every LUNA solver entry
    point takes."""
    dx = cfg.level_dx(level)
    ndim = cfg.ndim
    return dict(inv_dx=1.0 / dx[0],
                inv_dy=1.0 / dx[1] if ndim >= 2 else 1.0,
                inv_dz=1.0 / dx[2] if ndim == 3 else 1.0)


def cons_to_prim_blocks(U_blocks, gamma):
    """Per-block conservative -> primitive. ``U_blocks`` is
    ``(n_slots, n_vars, *block_shape)``; the NaN sentinel slot maps to NaN."""
    return jax.vmap(lambda U: cons_to_prim(U, gamma))(U_blocks)


def prim_to_cons_blocks(W_blocks, gamma):
    """Per-block primitive -> conservative (inverse of :func:`cons_to_prim_blocks`)."""
    return jax.vmap(lambda W: prim_to_cons(W, gamma))(W_blocks)


def block_divergence(
    W_ghosted,
    dt,
    level,
    cfg,
    *,
    gamma,
    limiter=1,
    riemann_solver="hllc",
    order=2,
):
    """Per-block hydro flux divergence ``dU/dt = -div.F`` from ghost-filled
    primitive blocks ``W_ghosted`` (n_slots, n_vars, *(block_shape + 2*n_ghost));
    ``dt`` enters the MUSCL-Hancock predictor only, ``level`` sets dx. Returns
    (n_slots, n_vars, *block_shape)."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    W_pad = W_ghosted[_inner_ghost_index(ndim, g)]      # interior + 1 ghost
    ik = inv_kw(cfg, level)

    def one_block(Wp):
        return muscl_hancock_divergence(
            Wp,
            dt,
            gamma=gamma,
            limiter=limiter,
            riemann_solver=riemann_solver,
            spatial_dim=ndim,
            **ik,
            # boundary != 0 so the 2D path reads the ghost face rather than
            # self-wrapping; exact_wrap stays False so the 3D path does too.
            boundary=1,
            order=order,
            exact_wrap=False,
        )

    return jax.vmap(one_block)(W_pad)


def block_face_fluxes(
    W_ghosted,
    dt,
    level,
    cfg,
    *,
    gamma,
    limiter=1,
    riemann_solver="hllc",
):
    """Per-block MUSCL-Hancock face fluxes: like :func:`block_divergence` but
    returns the per-axis tuple ``(Fx, Fy[, Fz])``, each
    ``(n_slots, n_vars, ..., block_shape[d] + 1, ...)``; the block-boundary
    faces are the coarse-fine interface fluxes consumed by :mod:`amr.reflux`."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    W_pad = W_ghosted[_inner_ghost_index(ndim, g)]
    ik = inv_kw(cfg, level)

    def one_block(Wp):
        return muscl_hancock_face_fluxes(
            Wp, dt, gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
            spatial_dim=ndim, **ik, boundary=1, exact_wrap=False,
        )

    return jax.vmap(one_block)(W_pad)


def block_boundary_states(W_ghosted, dt, level, cfg, *, gamma, limiter=1):
    """Per-block reconstructed boundary face states for same-level seam fluxes:
    per-axis tuple ``((L_0, R_0), ...)``, each ``(n_slots, n_vars, *transverse)``
    -- the left/right reconstructions at the block's first/last interior cell
    (see :func:`hydro.muscl_hancock.muscl_hancock_boundary_states`)."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    W_pad = W_ghosted[_inner_ghost_index(ndim, g)]
    ik = inv_kw(cfg, level)

    def one_block(Wp):
        return muscl_hancock_boundary_states(
            Wp, dt, gamma=gamma, limiter=limiter, spatial_dim=ndim, **ik,
        )

    return jax.vmap(one_block)(W_pad)


def _set_boundary_face(F_d, d, idx, new_face, mask, ndim):
    """Overwrite a block's boundary face flux (``idx`` 0 = -d, -1 = +d; face axis
    at array position ``2+d`` because of the leading (n_slots, n_vars) axes) with
    ``new_face`` where ``mask`` (per-block) is set, else keep the original."""
    ax = 2 + d
    sl = [slice(None)] * F_d.ndim
    sl[ax] = idx
    sl = tuple(sl)
    orig = F_d[sl]                                  # (n_slots, n_vars, *transverse)
    m = mask.reshape((mask.shape[0],) + (1,) * (orig.ndim - 1))
    return F_d.at[sl].set(jnp.where(m, new_face, orig))


def conservative_block_divergence(
    W_ghosted,
    dt,
    level,
    cfg,
    neighbors,
    *,
    gamma,
    limiter=1,
    riemann_solver="hllc",
    return_face_fluxes=False,
    coords=None,
):
    """Per-block flux divergence with **consistent same-level seam fluxes**:
    each block-block face shares ONE Riemann flux (this block's ``R_d`` vs the
    neighbour's ``L_d``), so conserved variables telescope across the seam to
    machine precision. Coarse-fine faces keep the ghost-based flux (fixed by
    :mod:`amr.reflux`). With ``coords`` and ``cfg.boundary == 1``, outflow
    domain faces use flux = HLLC(pred, pred) = f(pred) — the BC applied at the
    predicted-state level, avoiding the O(dt) wall mass pump of the ghost-based
    flux (Sod AMR benchmark: drho/rho ~ 5e-3 by t=0.2)."""
    ndim = cfg.ndim
    n_slots = neighbors.shape[0]
    sentinel = n_slots - 1

    fluxes = list(block_face_fluxes(
        W_ghosted, dt, level, cfg, gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
    ))
    states = block_boundary_states(W_ghosted, dt, level, cfg, gamma=gamma, limiter=limiter)

    def _seam(a, b, d):
        return jax.vmap(
            lambda x, y: riemann_face_flux(x, y, axis=d, gamma=gamma, riemann_solver=riemann_solver)
        )(a, b)

    bs = cfg.block_shape
    r = cfg.refinement_ratio
    for d in range(ndim):
        Ld, Rd = states[d]                          # (n_slots, n_vars, *transverse)
        nbr_plus = neighbors[:, 2 * d]
        nbr_minus = neighbors[:, 2 * d + 1]
        # +d seam: this block's R_d vs the +d neighbour's L_d.
        seam_plus = _seam(Rd, Ld[nbr_plus], d)
        # -d seam: the -d neighbour's R_d vs this block's L_d (== that neighbour's
        # own +d seam, so both sides use the identical flux).
        seam_minus = _seam(Rd[nbr_minus], Ld, d)

        F_d = fluxes[d]
        F_d = _set_boundary_face(F_d, d, -1, seam_plus, nbr_plus != sentinel, ndim)
        F_d = _set_boundary_face(F_d, d, 0, seam_minus, nbr_minus != sentinel, ndim)

        if coords is not None and cfg.boundary == 1:
            nb_d = (cfg.base_grid[d] // bs[d]) * (r ** level)   # blocks along d
            at_hi = (coords[:, d] == nb_d - 1) & (nbr_plus == sentinel)
            at_lo = (coords[:, d] == 0) & (nbr_minus == sentinel)
            F_d = _set_boundary_face(F_d, d, -1, _seam(Rd, Rd, d), at_hi, ndim)
            F_d = _set_boundary_face(F_d, d, 0, _seam(Ld, Ld, d), at_lo, ndim)
        fluxes[d] = F_d

    ik = inv_kw(cfg, level)

    # fluxes carry a leading block axis; assemble the divergence per block.
    def _div(*fs):
        return divergence_from_face_fluxes(fs, spatial_dim=ndim, **ik)

    dU = jax.vmap(_div)(*fluxes)
    if return_face_fluxes:
        # the exact per-axis face fluxes the update used (seam-corrected) --
        # consumed by amr.reflux for coarse-fine flux correction.
        return dU, tuple(fluxes)
    return dU


def advance_blocks_conservative(
    U_blocks,
    fill_ghosts,
    neighbors,
    dt,
    level,
    cfg,
    *,
    gamma,
    limiter=1,
    riemann_solver="hllc",
):
    """Like :func:`advance_blocks` but with consistent same-level seam fluxes
    (:func:`conservative_block_divergence`)."""
    W_blocks = cons_to_prim_blocks(U_blocks, gamma)
    W_ghosted = fill_ghosts(W_blocks)
    dU = conservative_block_divergence(
        W_ghosted, dt, level, cfg, neighbors,
        gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
    )
    return U_blocks + dt * dU


def advance_blocks(
    U_blocks,
    fill_ghosts,
    dt,
    level,
    cfg,
    *,
    gamma,
    limiter=1,
    riemann_solver="hllc",
    order=2,
):
    """Advance conservative blocks one step: ``U <- U + dt * dU``.

    ``fill_ghosts(W_blocks) -> W_ghosted`` encodes the level's halo policy.
    The NaN sentinel slot advances to NaN and is ignored downstream."""
    W_blocks = cons_to_prim_blocks(U_blocks, gamma)
    W_ghosted = fill_ghosts(W_blocks)
    dU = block_divergence(
        W_ghosted, dt, level, cfg,
        gamma=gamma, limiter=limiter, riemann_solver=riemann_solver, order=order,
    )
    return U_blocks + dt * dU
