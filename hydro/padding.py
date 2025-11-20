import jax.numpy as jnp
from jax import lax
import jax

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
        """
        Apply reflective boundary conditions.
        For wall boundaries, the normal momentum component must be negated.
        Conservative variables: [rho, rho*u, rho*v, (rho*w), E]
        """
        if ndim == 1:
            # 1D: U has shape (3, Nx) with [rho, rho*u, E]
            # Left boundary: copy interior cell but negate rho*u (index 1)
            left = U[:, 0:1].at[1].set(-U[1, 0:1])
            # Right boundary: copy interior cell but negate rho*u (index 1)
            right = U[:, -1:].at[1].set(-U[1, -1:])
            return jnp.concatenate([left, U, right], axis=1)

        elif ndim == 2:
            # 2D: U has shape (4, Nx, Ny) with [rho, rho*u, rho*v, E]
            # Left/right boundaries (x-direction): negate rho*u (index 1)
            left = U[:, 0:1, :].at[1].set(-U[1, 0:1, :])
            right = U[:, -1:, :].at[1].set(-U[1, -1:, :])
            Ux = jnp.concatenate([left, U, right], axis=1)

            # Bottom/top boundaries (y-direction): negate rho*v (index 2)
            bottom = Ux[:, :, 0:1].at[2].set(-Ux[2, :, 0:1])
            top = Ux[:, :, -1:].at[2].set(-Ux[2, :, -1:])
            return jnp.concatenate([bottom, Ux, top], axis=2)

        elif ndim == 3:
            # 3D: U has shape (5, Nx, Ny, Nz) with [rho, rho*u, rho*v, rho*w, E]
            # Left/right boundaries (x-direction): negate rho*u (index 1)
            left = U[:, 0:1, :, :].at[1].set(-U[1, 0:1, :, :])
            right = U[:, -1:, :, :].at[1].set(-U[1, -1:, :, :])
            Ux = jnp.concatenate([left, U, right], axis=1)

            # Bottom/top boundaries (y-direction): negate rho*v (index 2)
            bottom = Ux[:, :, 0:1, :].at[2].set(-Ux[2, :, 0:1, :])
            top = Ux[:, :, -1:, :].at[2].set(-Ux[2, :, -1:, :])
            Uy = jnp.concatenate([bottom, Ux, top], axis=2)

            # Back/front boundaries (z-direction): negate rho*w (index 3)
            back = Uy[:, :, :, 0:1].at[3].set(-Uy[3, :, :, 0:1])
            front = Uy[:, :, :, -1:].at[3].set(-Uy[3, :, :, -1:])
            return jnp.concatenate([back, Uy, front], axis=3)

    # -------------------------
    # Fixed JAX-boolean tests
    # -------------------------
   
    is_periodic  = jnp.array(boundary == 0)
    is_zero_grad = jnp.array(boundary == 1)
    is_reflective = jnp.array(boundary == 2)

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

