import jax.numpy as jnp


def soundwave_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """
    Linear sound wave in 3D.

    Small-amplitude sinusoidal perturbation propagating diagonally
    through a uniform medium. Tests numerical dissipation, convergence,
    and that the 3D pipeline preserves the wave correctly.

    Very lightweight: runs at 32^3 on a single GPU.
    Periodic boundaries required.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Ny, Nz, Lx, Ly, Lz, gamma.
    x_slice, y_slice, z_slice : tuple (start, end), optional
        Generate only this range for distributed initialization

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*vx, rho*vy, rho*vz, E]
        with shape (5, Nx_local, Ny_local, Nz_local)
    """
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    # Determine local ranges
    if x_slice is not None:
        x_start, x_end = x_slice
    else:
        x_start, x_end = 0, Nx

    if y_slice is not None:
        y_start, y_end = y_slice
    else:
        y_start, y_end = 0, Ny

    if z_slice is not None:
        z_start, z_end = z_slice
    else:
        z_start, z_end = 0, Nz

    # Create coordinate arrays for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    y_full = jnp.linspace(0, Ly, Ny, endpoint=False)
    z_full = jnp.linspace(0, Lz, Nz, endpoint=False)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    z_local = z_full[z_start:z_end]
    X, Y, Z = jnp.meshgrid(x_local, y_local, z_local, indexing="ij")

    # Background state
    rho0 = 1.0
    p0 = 1.0
    epsilon = 1e-6

    # Diagonal wavevector: one mode in each direction
    kx = 2.0 * jnp.pi / Lx
    ky = 2.0 * jnp.pi / Ly
    kz = 2.0 * jnp.pi / Lz

    # Phase: wave propagating diagonally
    phase = kx * X + ky * Y + kz * Z

    # Perturbed state
    rho = rho0 * (1.0 + epsilon * jnp.sin(phase))
    p = p0 * (1.0 + epsilon * jnp.sin(phase))

    # Zero mean velocity (standing wave)
    vx = jnp.zeros_like(X)
    vy = jnp.zeros_like(X)
    vz = jnp.zeros_like(X)

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    U = jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)
    return U
