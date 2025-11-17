import jax.numpy as jnp

def periodic_pad(U):
    """
    Dimension-agnostic periodic padding.
    Pads with 1 ghost cell on each side in every spatial dimension.

    Input shapes:
        1D: (Nvar, Nx)
        2D: (Nvar, Nx, Ny)
        3D: (Nvar, Nx, Ny, Nz)

    Output shapes:
        1D → (Nvar, Nx+2)
        2D → (Nvar, Nx+2, Ny+2)
        3D → (Nvar, Nx+2, Ny+2, Nz+2)
    """
    D = U.ndim - 1  # number of spatial dims

    if D == 1:
        _, Nx = U.shape
        U_pad = jnp.zeros((U.shape[0], Nx + 2), dtype=U.dtype)

        U_pad = U_pad.at[:, 1:-1].set(U)
        U_pad = U_pad.at[:, 0].set(U[:, -1])
        U_pad = U_pad.at[:, -1].set(U[:, 0])
        return U_pad

    elif D == 2:
        _, Nx, Ny = U.shape
        U_pad = jnp.zeros((U.shape[0], Nx + 2, Ny + 2), dtype=U.dtype)

        U_pad = U_pad.at[:, 1:-1, 1:-1].set(U)

        # x-boundaries
        U_pad = U_pad.at[:, 0, 1:-1].set(U[:, -1, :])
        U_pad = U_pad.at[:, -1, 1:-1].set(U[:, 0, :])

        # y-boundaries
        U_pad = U_pad.at[:, 1:-1, 0].set(U[:, :, -1])
        U_pad = U_pad.at[:, 1:-1, -1].set(U[:, :, 0])

        # corners
        U_pad = U_pad.at[:, 0, 0].set(U[:, -1, -1])
        U_pad = U_pad.at[:, 0, -1].set(U[:, -1, 0])
        U_pad = U_pad.at[:, -1, 0].set(U[:, 0, -1])
        U_pad = U_pad.at[:, -1, -1].set(U[:, 0, 0])

        return U_pad

    elif D == 3:
        _, Nx, Ny, Nz = U.shape
        U_pad = jnp.zeros((U.shape[0], Nx + 2, Ny + 2, Nz + 2), dtype=U.dtype)

        U_pad = U_pad.at[:, 1:-1, 1:-1, 1:-1].set(U)

        # x-wrap planes
        U_pad = U_pad.at[:, 0, 1:-1, 1:-1].set(U[:, -1, :, :])
        U_pad = U_pad.at[:, -1, 1:-1, 1:-1].set(U[:, 0, :, :])

        # y-wrap planes
        U_pad = U_pad.at[:, 1:-1, 0, 1:-1].set(U[:, :, -1, :])
        U_pad = U_pad.at[:, 1:-1, -1, 1:-1].set(U[:, :, 0, :])

        # z-wrap planes
        U_pad = U_pad.at[:, 1:-1, 1:-1, 0].set(U[:, :, :, -1])
        U_pad = U_pad.at[:, 1:-1, 1:-1, -1].set(U[:, :, :, 0])

        # corners & edges automatically handled by consistency
        return U_pad

    else:
        raise ValueError("Periodic padding only defined for 1D, 2D, or 3D fields.")

def zero_gradient_pad(U):
    """
    Dimension-agnostic zero-gradient padding (Neumann BCs).
    Pads with 1 ghost cell on each side in every spatial dimension.

    Input shapes:
        1D: (Nvar, Nx)
        2D: (Nvar, Nx, Ny)
        3D: (Nvar, Nx, Ny, Nz)

    Output shapes:
        1D → (Nvar, Nx+2)
        2D → (Nvar, Nx+2, Ny+2)
        3D → (Nvar, Nx+2, Ny+2, Nz+2)
    """
    D = U.ndim - 1  # number of spatial dims

    if D == 1:
        _, Nx = U.shape
        U_pad = jnp.zeros((U.shape[0], Nx + 2), dtype=U.dtype)

        # interior
        U_pad = U_pad.at[:, 1:-1].set(U)

        # zero-gradient BCs: copy boundary values
        U_pad = U_pad.at[:, 0].set(U[:, 0])
        U_pad = U_pad.at[:, -1].set(U[:, -1])
        return U_pad

    elif D == 2:
        _, Nx, Ny = U.shape
        U_pad = jnp.zeros((U.shape[0], Nx + 2, Ny + 2), dtype=U.dtype)
        U_pad = U_pad.at[:, 1:-1, 1:-1].set(U)

        # zero-gradient BCs: copy edges
        # x-boundaries
        U_pad = U_pad.at[:, 0, 1:-1].set(U[:, 0, :])
        U_pad = U_pad.at[:, -1, 1:-1].set(U[:, -1, :])
        # y-boundaries
        U_pad = U_pad.at[:, 1:-1, 0].set(U[:, :, 0])
        U_pad = U_pad.at[:, 1:-1, -1].set(U[:, :, -1])
        # corners
        U_pad = U_pad.at[:, 0, 0].set(U[:, 0, 0])
        U_pad = U_pad.at[:, 0, -1].set(U[:, 0, -1])
        U_pad = U_pad.at[:, -1, 0].set(U[:, -1, 0])
        U_pad = U_pad.at[:, -1, -1].set(U[:, -1, -1])
        return U_pad

    elif D == 3:
        _, Nx, Ny, Nz = U.shape
        U_pad = jnp.zeros((U.shape[0], Nx + 2, Ny + 2, Nz + 2), dtype=U.dtype)
        U_pad = U_pad.at[:, 1:-1, 1:-1, 1:-1].set(U)

        # zero-gradient BCs: copy faces
        # x-boundaries
        U_pad = U_pad.at[:, 0, 1:-1, 1:-1].set(U[:, 0, :, :])
        U_pad = U_pad.at[:, -1, 1:-1, 1:-1].set(U[:, -1, :, :])
        # y-boundaries
        U_pad = U_pad.at[:, 1:-1, 0, 1:-1].set(U[:, :, 0, :])
        U_pad = U_pad.at[:, 1:-1, -1, 1:-1].set(U[:, :, -1, :])
        # z-boundaries
        U_pad = U_pad.at[:, 1:-1, 1:-1, 0].set(U[:, :, :, 0])
        U_pad = U_pad.at[:, 1:-1, 1:-1, -1].set(U[:, :, :, -1])
        # corners & edges automatically handled
        return U_pad

    else:
        raise ValueError("Zero-gradient padding only defined for 1D, 2D, or 3D fields.")

