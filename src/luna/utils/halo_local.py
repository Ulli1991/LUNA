"""Single-device halo via jnp.pad: the same ghosts as make_halo_shardmap on a 1-device mesh, no Mesh needed.

Boundary codes: 0 periodic, 1 zero-gradient, 2 reflective, 4 diode. See docs/HYDRO_KERNEL_PATHS.md."""

import jax.numpy as jnp


def pad_halo_local(W, boundary, sdim, width=1, axes=None):
    """Pad (Nvar, N...) W with `width` ghosts per side (wrap/edge/mirror, then reflect or diode the normal velocity)."""
    if sdim not in (1, 2, 3) or W.ndim != sdim + 1:
        raise ValueError(f"W of shape {W.shape} does not match spatial_dim={sdim}")
    if boundary not in (0, 1, 2, 4):
        raise NotImplementedError(f"local halo supports boundary 0/1/2/4; got {boundary}")
    if axes is not None and boundary not in (0, 1):
        raise NotImplementedError("per-axis halo: boundary 0/1 only")
    mode = {0: "wrap", 2: "symmetric"}.get(boundary, "edge")   # width 1: symmetric == edge
    padded = range(sdim) if axes is None else axes
    pads = [(0, 0)] + [(width, width) if d in padded else (0, 0) for d in range(sdim)]
    U = jnp.pad(W, pads, mode=mode)
    if boundary in (2, 4):
        for d in range(sdim):
            ax, comp = 1 + d, 1 + d
            lo = [slice(None)] * U.ndim
            hi = [slice(None)] * U.ndim
            lo[0] = hi[0] = comp
            lo[ax], hi[ax] = slice(0, width), slice(-width, None)
            lo, hi = tuple(lo), tuple(hi)
            if boundary == 2:
                U = U.at[lo].set(-U[lo]).at[hi].set(-U[hi])
            else:
                U = U.at[lo].set(jnp.minimum(U[lo], 0.0)).at[hi].set(jnp.maximum(U[hi], 0.0))
    return U
