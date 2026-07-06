"""Dimension-generic block geometry (pure JAX).

The trailing ``ndim`` axes of every array are spatial; any leading axes
(``n_blocks``, ``n_vars``) are carried through untouched. This is the single
code path the brief asked for: 2D and 3D differ only by ``ndim``/axis tuples,
never by hand-unrolled per-axis code.

Contents:
  * ``morton_encode`` / ``morton_decode`` — block identity & spatial-locality
    ordering (O(N log N) neighbour finding / unaltered-block matching, vs the
    O(N^2) pairwise compare in upstream ``amr.py:356``).
  * ``prolong`` / ``restrict`` — conservative inter-level transfer.
  * block <-> physical-coordinate helpers.
  * ``refinement_grid_mask`` — delta density-contrast criterion + buffer dilation
    (cosmology wants delta, not the gradient-of-density used upstream).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

# Morton bit budget is dtype-limited. LUNA runs float32 / int32 by DEFAULT
# (`setup/distributed_setup.py`: enable_x64=False), so without x64 the widest
# integer is 32 bits and a uint64 code silently truncates. We therefore pick
# dtype + bits-per-dim from the *current* x64 setting at call time (the flag may
# be set after import by initialize_jax_distributed):
#   * x64 ON  -> uint64, 21 bits/dim  (coords < 2**21 ~ 2.1M per dim)
#   * x64 OFF -> uint32; 16 bits/dim in 2D, 10 bits/dim in 3D
#               (3D coords < 1024/dim => base_blocks*2**level < 1024).
# A level with more blocks-per-dim than the budget needs x64; callers can check
# with `morton_capacity(ndim)`.
def _morton_params(ndim):
    if jax.config.jax_enable_x64:
        return jnp.uint64, 21
    return jnp.uint32, (31 if ndim == 1 else 16 if ndim == 2 else 10)


def morton_capacity(ndim):
    """Max representable block-coord per dim for the current x64 setting."""
    _, bits = _morton_params(ndim)
    return 1 << bits


# ---------------------------------------------------------------------------
# Morton (Z-order) codes — block identity + locality-preserving ordering
# ---------------------------------------------------------------------------
def morton_encode(coords, dtype=None, bits=None):
    """Interleave per-dim integer block coords into one Z-order code.

    Parameters
    ----------
    coords : int array (..., ndim)
        Non-negative block coordinates; axis 0 of the last dim is the least
        significant (varies fastest in the interleave).
    dtype, bits : optional
        Override the auto-selected (x64-aware) code dtype and bits-per-dim.
        Useful for testing the uint32 path without toggling global x64.

    Returns
    -------
    uint32/uint64 array (...,) per the active x64 setting.
    """
    coords = jnp.asarray(coords)
    ndim = coords.shape[-1]
    adt, abits = _morton_params(ndim)
    dtype = adt if dtype is None else dtype
    bits = abits if bits is None else bits
    c = coords.astype(dtype)
    code = jnp.zeros(coords.shape[:-1], dtype=dtype)
    one = dtype(1)
    for b in range(bits):
        for d in range(ndim):
            bit = (c[..., d] >> dtype(b)) & one
            code |= bit << dtype(b * ndim + d)
    return code


def morton_decode(code, ndim, dtype=None, bits=None):
    """Inverse of :func:`morton_encode`. Returns int32 coords (..., ndim)."""
    adt, abits = _morton_params(ndim)
    dtype = adt if dtype is None else dtype
    bits = abits if bits is None else bits
    code = jnp.asarray(code).astype(dtype)
    one = dtype(1)
    out = []
    for d in range(ndim):
        cd = jnp.zeros(code.shape, dtype=dtype)
        for b in range(bits):
            bit = (code >> dtype(b * ndim + d)) & one
            cd |= bit << dtype(b)
        out.append(cd.astype(jnp.int32))
    return jnp.stack(out, axis=-1)


# ---------------------------------------------------------------------------
# Conservative inter-level transfer (generic over the trailing ndim axes)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def prolong(data, ndim):
    """Coarse -> fine, piecewise-constant (refinement ratio 2).

    Each coarse cell becomes ``2**ndim`` fine cells with the same value. For a
    density-like quantity this is conservative: total over the children
    (volume (dx/2)**ndim each) equals the parent (volume dx**ndim). Equivalent
    to upstream's ``jnp.kron(data, ones((2,)*ndim))`` but via ``repeat`` (no
    flatten/reshape blow-up).
    """
    out = data
    for ax in range(-ndim, 0):
        out = jnp.repeat(out, 2, axis=ax)
    return out


@partial(jax.jit, static_argnums=(1,))
def prolong_linear(data, ndim):
    """Coarse -> fine, cell-centred (tri)linear interpolation (ratio 2).

    Per axis: ``fine[2i] = 0.75 c[i] + 0.25 c[i-1]``, ``fine[2i+1] = 0.75 c[i]
    + 0.25 c[i+1]`` (edges clamp one-sided). NOT conservative -- meant for
    smooth scalars, e.g. Dirichlet phi ghosts at coarse-fine boundaries, where
    the piecewise-constant :func:`prolong` puts an O(dx_coarse) interpolation
    error (~5% of a k=2pi/L mode) straight into the fine level's boundary
    values."""
    out = data
    nd_total = data.ndim
    for d in range(ndim):
        ax = nd_total - ndim + d
        out = jnp.moveaxis(out, ax, -1)
        lo = jnp.concatenate([out[..., :1], out[..., :-1]], axis=-1)
        hi = jnp.concatenate([out[..., 1:], out[..., -1:]], axis=-1)
        even = 0.75 * out + 0.25 * lo
        odd = 0.75 * out + 0.25 * hi
        inter = jnp.stack([even, odd], axis=-1)
        inter = inter.reshape(out.shape[:-1] + (2 * out.shape[-1],))
        out = jnp.moveaxis(inter, -1, ax)
    return out


@partial(jax.jit, static_argnums=(1,))
def prolong_linear_cons(data, ndim):
    """Coarse -> fine, CONSERVATIVE minmod-limited linear (ratio 2).

    The two children of a parent cell along each axis are ``parent -/+ slope/4``
    with ``slope = minmod(c_i - c_{i-1}, c_{i+1} - c_i)`` -- the child pair mean
    equals the parent exactly (mass/momentum/energy conserving), and minmod
    keeps the interpolation monotone (no new extrema, positivity-safe). Edge
    cells limit to zero slope (piecewise-constant). This is the RAMSES-style
    initialization for NEWLY REFINED cells: piecewise-constant injection there
    plants an O(dx_coarse) staircase right on the feature that triggered
    refinement, which a converging flow then amplifies (measured: 11% density
    error at a Zeldovich pancake peak vs ~1% with limited-linear)."""
    out = data
    nd_total = data.ndim
    for d in range(ndim):
        ax = nd_total - ndim + d
        out = jnp.moveaxis(out, ax, -1)
        dl = out - jnp.concatenate([out[..., :1], out[..., :-1]], axis=-1)
        dr = jnp.concatenate([out[..., 1:], out[..., -1:]], axis=-1) - out
        slope = jnp.where(dl * dr > 0.0,
                          jnp.sign(dl) * jnp.minimum(jnp.abs(dl), jnp.abs(dr)),
                          0.0)
        lo = out - 0.25 * slope
        hi = out + 0.25 * slope
        inter = jnp.stack([lo, hi], axis=-1)
        inter = inter.reshape(out.shape[:-1] + (2 * out.shape[-1],))
        out = jnp.moveaxis(inter, -1, ax)
    return out


@partial(jax.jit, static_argnums=(1,))
def restrict(data, ndim):
    """Fine -> coarse, ``2**ndim`` average (refinement ratio 2).

    Each trailing spatial axis must be even. Generic reshape: split every
    spatial axis ``2m -> (m, 2)`` then mean over the size-2 sub-axes.
    """
    lead = data.shape[:-ndim]
    spatial = data.shape[-ndim:]
    new_shape = lead + tuple(x for s in spatial for x in (s // 2, 2))
    reshaped = data.reshape(new_shape)
    n_lead = len(lead)
    avg_axes = tuple(n_lead + 1 + 2 * i for i in range(ndim))
    return reshaped.mean(axis=avg_axes)


# ---------------------------------------------------------------------------
# Block <-> physical coordinates
# ---------------------------------------------------------------------------
def block_origin(block_coords, level, cfg):
    """Physical lower corner of a block, given its integer coords at ``level``.

    block_coords : int array (..., ndim)
    Returns float array (..., ndim).
    """
    block_coords = jnp.asarray(block_coords)
    dx = jnp.asarray(cfg.level_dx(level))
    block_shape = jnp.asarray(cfg.block_shape)
    block_width = dx * block_shape  # physical extent of one block per dim
    return block_coords * block_width


def block_cell_centers(block_coords, level, cfg):
    """Cell-center physical coordinates for a single block.

    block_coords : int array (ndim,)
    Returns float array (ndim, *block_shape): component d is the d-th coordinate
    of every cell center in the block.
    """
    ndim = cfg.ndim
    dx = cfg.level_dx(level)
    origin = block_origin(block_coords, level, cfg)
    axes = [origin[d] + (jnp.arange(cfg.block_shape[d]) + 0.5) * dx[d] for d in range(ndim)]
    grids = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack(grids, axis=0)


# ---------------------------------------------------------------------------
# Refinement criterion: delta density-contrast + buffer dilation
# ---------------------------------------------------------------------------
def _dilate(mask, buffer, ndim):
    """Grow a boolean/0-1 mask by ``buffer`` cells along each spatial axis.

    Dimension-generic morphological dilation via ``lax.reduce_window`` (max over
    a (2*buffer+1)**ndim window) on the trailing ndim axes. This is the
    intra-block refine margin; cross-block dilation is applied later once block
    topology / halos exist.
    """
    if buffer <= 0:
        return mask
    n_lead = mask.ndim - ndim
    window = (1,) * n_lead + (2 * buffer + 1,) * ndim
    strides = (1,) * mask.ndim
    pad = [(0, 0)] * n_lead + [(buffer, buffer)] * ndim
    return jax.lax.reduce_window(
        mask, -jnp.inf, jax.lax.max, window, strides, pad
    )


def refinement_grid_mask(blk_data, cfg, *, threshold, var_index=0, buffer=None,
                         contrast_mode="value"):
    """Per-cell refinement flag from a density-contrast criterion.

    Parameters
    ----------
    blk_data : array (n_blocks, n_vars, *block_shape)
        Block field data. The NaN sentinel block (last slot) flags nowhere
        because ``NaN > threshold`` is False.
    threshold : float
        Refine where the contrast exceeds this.
    var_index : int
        Which variable channel carries density (0 by default).
    buffer : int or None
        Refine-margin half-width. Defaults to ``cfg.n_ghost``.
    contrast_mode : {"value", "delta"}
        "value": flag cells where ``rho > threshold``.
        "delta": flag where ``rho/mean(rho) - 1 > threshold`` (mean over valid,
        non-NaN cells) — the cosmological density contrast.

    Returns
    -------
    float array (n_blocks, *block_shape) of 0/1 flags.
    """
    if buffer is None:
        buffer = cfg.n_ghost
    ndim = cfg.ndim
    rho = blk_data[:, var_index]  # (n_blocks, *block_shape)

    if contrast_mode == "delta":
        mean_rho = jnp.nanmean(rho)
        field = rho / mean_rho - 1.0
    elif contrast_mode == "value":
        field = rho
    else:
        raise ValueError(f"contrast_mode must be 'value' or 'delta'; got {contrast_mode}")

    flagged = jnp.where(field > threshold, 1.0, 0.0)  # NaN -> 0 (False)
    dilated = _dilate(flagged, buffer, ndim)
    return jnp.sign(dilated)
