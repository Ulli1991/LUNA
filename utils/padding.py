import jax.numpy as jnp
from jax import lax

def pad(U, boundary):
    """
    General dimension-agnostic padding with selectable boundary conditions.

    Parameters
    ----------
    U : array
        Shape (Nvar, Nx) for 1D, (Nvar, Nx, Ny) for 2D, (Nvar, Nx, Ny, Nz) for 3D.
    boundary : int
        0 = periodic, 1 = zero-gradient, 2 = reflective

    Returns
    -------
    U_pad : array
        Padded array with 1 ghost cell per side in each spatial dimension.
    """

    ndim = U.ndim - 1  # number of spatial dimensions

    # -------------------------
    # Boundary padding functions
    # -------------------------

    def periodic(U):
        """Apply periodic boundary conditions."""
        if ndim == 1:
            return jnp.concatenate([U[:, -1:], U, U[:, :1]], axis=1)
        elif ndim == 2:
            left  = U[:, -1:, :]
            right = U[:, :1, :]
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = Ux[:, :, -1:]
            top    = Ux[:, :, :1]
            return jnp.concatenate([bottom, Ux, top], axis=2)
        elif ndim == 3:
            left  = U[:, -1:, :, :]
            right = U[:, :1, :, :]
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = Ux[:, :, -1:, :]
            top    = Ux[:, :, :1, :]
            Uy = jnp.concatenate([bottom, Ux, top], axis=2)
            back  = Uy[:, :, :, -1:]
            front = Uy[:, :, :, :1]
            return jnp.concatenate([back, Uy, front], axis=3)

    def zero_grad(U):
        """Apply zero-gradient boundary conditions."""
        if ndim == 1:
            left  = U[:, 0:1]
            right = U[:, -1:]
            return jnp.concatenate([left, U, right], axis=1)
        elif ndim == 2:
            left  = U[:, 0:1, :]
            right = U[:, -1:, :]
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = Ux[:, :, 0:1]
            top    = Ux[:, :, -1:]
            return jnp.concatenate([bottom, Ux, top], axis=2)
        elif ndim == 3:
            left  = U[:, 0:1, :, :]
            right = U[:, -1:, :, :]
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = Ux[:, :, 0:1, :]
            top    = Ux[:, :, -1:, :]
            Uy = jnp.concatenate([bottom, Ux, top], axis=2)
            back  = Uy[:, :, :, 0:1]
            front = Uy[:, :, :, -1:]
            return jnp.concatenate([back, Uy, front], axis=3)

    def reflective(U):
        """Apply reflective boundary conditions."""
        if ndim == 1:
            # For 1D, reflective boundary: replicate first and last cells
            left  = U[:, 1:2]   # Reflect the first interior cell for left boundary
            right = U[:, -2:-1] # Reflect the last interior cell for right boundary
            return jnp.concatenate([left, U, right], axis=1)

        elif ndim == 2:
            # For 2D, reflective boundary: replicate first and last rows/columns
            left  = U[:, 1:2, :]   # Reflect the first interior column for left boundary
            right = U[:, -2:-1, :] # Reflect the last interior column for right boundary
            Ux = jnp.concatenate([left, U, right], axis=1)

            # Reflect the first and last rows
            bottom = Ux[:, :, 1:2]   # Reflect the first interior row for bottom boundary
            top    = Ux[:, :, -2:-1] # Reflect the last interior row for top boundary
            return jnp.concatenate([bottom, Ux, top], axis=2)

        elif ndim == 3:
            # For 3D, reflective boundary: replicate first and last rows/columns/slices
            left  = U[:, 1:2, :, :]   # Reflect the first interior column for left boundary
            right = U[:, -2:-1, :, :] # Reflect the last interior column for right boundary
            Ux = jnp.concatenate([left, U, right], axis=1)

            # Reflect the first and last rows
            bottom = Ux[:, :, 1:2, :]   # Reflect the first interior row for bottom boundary
            top    = Ux[:, :, -2:-1, :] # Reflect the last interior row for top boundary
            Uy = jnp.concatenate([bottom, Ux, top], axis=2)

            # Reflect the first and last slices
            back  = Uy[:, :, :, 1:2]   # Reflect the first interior slice for back boundary
            front = Uy[:, :, :, -2:-1] # Reflect the last interior slice for front boundary
            return jnp.concatenate([back, Uy, front], axis=3)

    # -------------------------
    # Fixed JAX-boolean tests
    # -------------------------
    
    # Debugging output (you can comment these out once you're confident)
    print(f"Boundary value received: {boundary}")

    is_periodic  = jnp.array(boundary == 0)
    is_zero_grad = jnp.array(boundary == 1)
    is_reflective = jnp.array(boundary == 2)

    print(f"Periodic: {is_periodic}, Zero Gradient: {is_zero_grad}, Reflective: {is_reflective}")

    # -------------------------
    # Correct JAX nested cond
    # -------------------------
    
    # Use lax.cond to select the boundary condition
    return lax.cond(
        is_periodic,
        periodic,
        lambda U: lax.cond(is_zero_grad, zero_grad, reflective, U),
        U
    )

