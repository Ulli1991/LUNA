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

    def periodic(U):
        if ndim == 1:
            return jnp.concatenate([U[:, -1:], U, U[:, :1]], axis=1)
        elif ndim == 2:
            left = U[:, -1:, :]
            right = U[:, :1, :]
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = Ux[:, :, -1:]
            top = Ux[:, :, :1]
            return jnp.concatenate([bottom, Ux, top], axis=2)
        elif ndim == 3:
            left = U[:, -1:, :, :]
            right = U[:, :1, :, :]
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = Ux[:, :, -1:, :]
            top = Ux[:, :, :1, :]
            Uy = jnp.concatenate([bottom, Ux, top], axis=2)
            back = Uy[:, :, :, -1:]
            front = Uy[:, :, :, :1]
            return jnp.concatenate([back, Uy, front], axis=3)
        else:
            raise ValueError("Unsupported dimension")

    def zero_grad(U):
        if ndim == 1:
            left = jnp.expand_dims(U[:, 0], axis=1)
            right = jnp.expand_dims(U[:, -1], axis=1)
            return jnp.concatenate([left, U, right], axis=1)
        elif ndim == 2:
            left = jnp.expand_dims(U[:, 0, :], axis=1)
            right = jnp.expand_dims(U[:, -1, :], axis=1)
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = jnp.expand_dims(Ux[:, :, 0], axis=2)
            top = jnp.expand_dims(Ux[:, :, -1], axis=2)
            return jnp.concatenate([bottom, Ux, top], axis=2)
        elif ndim == 3:
            left = jnp.expand_dims(U[:, 0, :, :], axis=1)
            right = jnp.expand_dims(U[:, -1, :, :], axis=1)
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = jnp.expand_dims(Ux[:, :, 0, :], axis=2)
            top = jnp.expand_dims(Ux[:, :, -1, :], axis=2)
            Uy = jnp.concatenate([bottom, Ux, top], axis=2)
            back = jnp.expand_dims(Uy[:, :, :, 0], axis=3)
            front = jnp.expand_dims(Uy[:, :, :, -1], axis=3)
            return jnp.concatenate([back, Uy, front], axis=3)
        else:
            raise ValueError("Unsupported dimension")

    def reflective(U):
        if ndim == 1:
            left = jnp.expand_dims(U[:, 1], axis=1)
            right = jnp.expand_dims(U[:, -2], axis=1)
            return jnp.concatenate([left, U, right], axis=1)
        elif ndim == 2:
            left = jnp.expand_dims(U[:, 1, :], axis=1)
            right = jnp.expand_dims(U[:, -2, :], axis=1)
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = jnp.expand_dims(Ux[:, :, 1], axis=2)
            top = jnp.expand_dims(Ux[:, :, -2], axis=2)
            return jnp.concatenate([bottom, Ux, top], axis=2)
        elif ndim == 3:
            left = jnp.expand_dims(U[:, 1, :, :], axis=1)
            right = jnp.expand_dims(U[:, -2, :, :], axis=1)
            Ux = jnp.concatenate([left, U, right], axis=1)
            bottom = jnp.expand_dims(Ux[:, :, 1, :], axis=2)
            top = jnp.expand_dims(Ux[:, :, -2, :], axis=2)
            Uy = jnp.concatenate([bottom, Ux, top], axis=2)
            back = jnp.expand_dims(Uy[:, :, :, 1], axis=3)
            front = jnp.expand_dims(Uy[:, :, :, -2], axis=3)
            return jnp.concatenate([back, Uy, front], axis=3)
        else:
            raise ValueError("Unsupported dimension")

    # JIT-safe conditional selection
    return lax.cond(boundary == 0, periodic,
                    lambda U: lax.cond(boundary == 1, zero_grad, reflective, U),
                    U)

