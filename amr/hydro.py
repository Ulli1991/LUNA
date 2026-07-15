"""Hydro-on-blocks: graft LUNA's MUSCL-Hancock solver onto AMR blocks.

The block layer is the AMR analogue of ``shard_map``'s shards: each block holds
its interior cells plus a ghost halo filled from neighbours (same-level), from a
prolonged coarse parent (coarse-fine faces), or from a physical BC. We therefore
reuse :func:`hydro.muscl_hancock.muscl_hancock_divergence` *verbatim*, once per
block via ``vmap``, in exactly the mode LUNA already runs across shard seams:

  * ``exact_wrap=False`` (3D) / ``boundary != 0`` (2D) so the solver takes its
    block-boundary face state from the **ghost cell** rather than self-wrapping.
    All boundary physics thus lives in the halo fill, never in the solver.

LUNA's production stepper pads with a **single** ghost cell and accepts a
half-order, non-telescoping flux at that outer face (see the shard-seam comment
in ``hydro/muscl_hancock.py``). The block solver inherits the identical
contract: it consumes a 1-ghost ``W_pad`` sliced from the (wider) block halo, so
the inner ghost is the neighbour's edge cell — the per-block analogue of the
``ppermute`` halo exchange. The slope and face-BC helpers only read axis-aligned
face ghosts (their transverse indices are ``1:-1``), never the corner cells that
:func:`amr.halo.fill_ghost_same_level` leaves NaN, so the wider halo's unfilled
corners are safe.

The plain ghost-based path (:func:`block_divergence`) has each block compute its
own flux at a shared face (from its reconstructed edge state vs. the ghost), so
conserved quantities do *not* telescope across block seams — the same seam
non-conservation LUNA tolerates at shard boundaries, now at every block face.
This is resolved by two mechanisms, both implemented here / in :mod:`amr.reflux`:

  * **same-level seams**: :func:`conservative_block_divergence` /
    :func:`advance_blocks_conservative` share ONE Riemann flux per block-block
    face (this block's ``R_d`` + the neighbour's ``L_d``, exchanged on the slot
    axis), so mass and energy telescope to machine precision on a level.
  * **coarse-fine seams**: Berger-Colella reflux (:mod:`amr.reflux`, default on
    in the driver) corrects the coarse cell outside a fine patch by the
    transverse-averaged fine subface flux.

Faces with no same-level neighbour keep the ghost-based flux and rely on reflux
(coarse-fine) or the physical BC (domain edge).
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
    """Per-block hydro flux divergence ``dU/dt = -div.F`` from a ghost-filled
    primitive halo.

    Parameters
    ----------
    W_ghosted : (n_slots, n_vars, *(block_shape + 2*n_ghost))
        Primitive blocks with ghosts already filled (same-level + coarse + BC).
    dt : scalar
        Time step (enters the MUSCL-Hancock predictor only).
    level : int
        Refinement level; sets ``inv_dx`` via ``cfg.level_dx(level)``.

    Returns
    -------
    (n_slots, n_vars, *block_shape) flux divergence per block.
    """
    ndim = cfg.ndim
    g = cfg.n_ghost
    W_pad = W_ghosted[_inner_ghost_index(ndim, g)]      # interior + 1 ghost

    dx = cfg.level_dx(level)
    inv_dx = 1.0 / dx[0]
    inv_dy = 1.0 / dx[1] if ndim >= 2 else 1.0
    inv_dz = 1.0 / dx[2] if ndim == 3 else 1.0          # unused when ndim==2

    def one_block(Wp):
        return muscl_hancock_divergence(
            Wp,
            dt,
            gamma=gamma,
            limiter=limiter,
            riemann_solver=riemann_solver,
            spatial_dim=ndim,
            inv_dx=inv_dx,
            inv_dy=inv_dy,
            inv_dz=inv_dz,
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
    """Per-block MUSCL-Hancock face fluxes (for coarse-fine reflux).

    Same 1-ghost slice + per-block ``vmap`` as :func:`block_divergence`, but
    returns the per-axis face-flux arrays instead of the divergence. Output is a
    tuple ``(Fx, Fy[, Fz])`` with ``Fd`` of shape
    ``(n_slots, n_vars, ..., block_shape[d] + 1, ...)``. The block-boundary
    faces (``Fd[:, :, 0]`` / ``Fd[:, :, -1]``) are the coarse-fine interface
    fluxes consumed by :mod:`amr.reflux`.
    """
    ndim = cfg.ndim
    g = cfg.n_ghost
    W_pad = W_ghosted[_inner_ghost_index(ndim, g)]

    dx = cfg.level_dx(level)
    inv_dx = 1.0 / dx[0]
    inv_dy = 1.0 / dx[1] if ndim >= 2 else 1.0
    inv_dz = 1.0 / dx[2] if ndim == 3 else 1.0

    def one_block(Wp):
        return muscl_hancock_face_fluxes(
            Wp, dt, gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
            spatial_dim=ndim, inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
            boundary=1, exact_wrap=False,
        )

    return jax.vmap(one_block)(W_pad)


def block_boundary_states(W_ghosted, dt, level, cfg, *, gamma, limiter=1):
    """Per-block reconstructed boundary face states (for same-level seam fluxes).

    Returns a per-axis tuple ``((L_0, R_0), ...)`` with each slab stacked over
    blocks: ``(n_slots, n_vars, *transverse)``. ``L_d`` / ``R_d`` are the left /
    right reconstructions at the block's first / last interior cell along axis
    ``d`` (see :func:`hydro.muscl_hancock.muscl_hancock_boundary_states`)."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    W_pad = W_ghosted[_inner_ghost_index(ndim, g)]
    dx = cfg.level_dx(level)
    inv_dx = 1.0 / dx[0]
    inv_dy = 1.0 / dx[1] if ndim >= 2 else 1.0
    inv_dz = 1.0 / dx[2] if ndim == 3 else 1.0

    def one_block(Wp):
        return muscl_hancock_boundary_states(
            Wp, dt, gamma=gamma, limiter=limiter, spatial_dim=ndim,
            inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
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
    """Per-block flux divergence with **consistent same-level seam fluxes**.

    Each block-block face shares ONE Riemann flux, computed from both blocks'
    boundary reconstructions (this block's ``R_d`` and the neighbour's ``L_d``),
    so the flux that leaves one block equals the flux that enters its neighbour
    and the conserved variables telescope across the seam (machine-precision
    conservation on a single multi-block level). Faces with no same-level
    neighbour (coarse-fine interfaces) keep the ghost-based boundary flux from
    :func:`block_face_fluxes` — coarse-fine conservation is restored separately
    by :mod:`amr.reflux`.

    PHYSICAL (outflow) domain faces are special-cased when ``coords`` is given
    and ``cfg.boundary == 1``: the ghost-based flux there pairs the interior's
    half-step PREDICTED state against the ghost's UN-predicted copy (with 2
    ghost layers the predictor cannot be evaluated on the outermost ghost), an
    O(dt) inconsistency that pumps mass through the wall (Sod AMR benchmark:
    F_y[rho] = 2.3e-3 through a v=0 boundary; drho/rho ~ 5e-3 by t=0.2). The
    consistent zero-gradient flux applies the BC at the predicted-state level:
    flux = HLLC(pred, pred) = f(pred) — exactly zero mass flux when the
    predicted normal velocity vanishes."""
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

    dx = cfg.level_dx(level)
    inv_dx = 1.0 / dx[0]
    inv_dy = 1.0 / dx[1] if ndim >= 2 else 1.0
    inv_dz = 1.0 / dx[2] if ndim == 3 else 1.0

    # fluxes carry a leading block axis; assemble the divergence per block.
    def _div(*fs):
        return divergence_from_face_fluxes(
            fs, spatial_dim=ndim, inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz
        )

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

    ``fill_ghosts(W_blocks) -> W_ghosted`` is supplied by the caller and encodes
    the level's halo policy (same-level only for the base level; same-level +
    coarse-parent + BC for refined levels). Keeping it a callback lets this core
    advance stay agnostic to the level's neighbour/parent context.

    The NaN sentinel slot advances to NaN and is meant to be ignored downstream.
    """
    W_blocks = cons_to_prim_blocks(U_blocks, gamma)
    W_ghosted = fill_ghosts(W_blocks)
    dU = block_divergence(
        W_ghosted, dt, level, cfg,
        gamma=gamma, limiter=limiter, riemann_solver=riemann_solver, order=order,
    )
    return U_blocks + dt * dU
