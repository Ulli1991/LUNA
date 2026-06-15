import jax.numpy as jnp


def kelvin_helmholtz(cfg, x_slice=None, y_slice=None):
    """
    Kelvin-Helmholtz instability test problem (2D).

    Sets up two fluid layers with different velocities and densities.
    Small sinusoidal perturbations seed the instability, leading to
    characteristic vortex roll-up along the shear interface.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Ny, Lx, Ly, gamma
    x_slice : tuple (x_start, x_end), optional
        Generate only this x-range for distributed initialization
    y_slice : tuple (y_start, y_end), optional
        Generate only this y-range for distributed initialization

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*ux, rho*uy, E] with shape (4, Nx_local, Ny_local)
    """
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # Determine local ranges
    if x_slice is not None:
        x_start, x_end = x_slice
        Nx_local = x_end - x_start
    else:
        x_start, x_end = 0, Nx
        Nx_local = Nx

    if y_slice is not None:
        y_start, y_end = y_slice
        Ny_local = y_end - y_start
    else:
        y_start, y_end = 0, Ny
        Ny_local = Ny

    # Create coordinate arrays for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    y_full = jnp.linspace(0, Ly, Ny, endpoint=False)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")

    # KH parameters
    U0 = 1.0
    a = 0.05
    A = 0.01
    sigma = 0.2
    Y1, Y2 = 0.5, 1.5
    rho_low, rho_high = 1.0, 2.0
    delta = (rho_high - rho_low) / rho_low

    rho = 1.0 + 0.5 * delta * (jnp.tanh((Y - Y1) / a) - jnp.tanh((Y - Y2) / a))
    ux = -U0 * ((jnp.tanh((Y - Y1) / a) - jnp.tanh((Y - Y2) / a)) - 1.0)
    envelope = jnp.exp(-((Y - Y1) ** 2) / sigma**2) + jnp.exp(-((Y - Y2) ** 2) / sigma**2)

    # Seeding. Default: a single clean sin mode (the convergent KH used by the
    # kh_grid resolution series). With LUNA_KH_NOISE>0: PURE broadband white
    # noise and NO seeded mode -- the diffhydro §8.1 corrector regime, where a
    # turbulent multi-scale shear layer develops that a coarse grid cannot
    # resolve. Reproducible: noise is built on the FULL grid from a fixed seed
    # then sliced to the local shard, so coarse = downsampled(fine) is exact.
    import os
    noise_amp = float(os.environ.get("LUNA_KH_NOISE", "0.0"))
    if noise_amp > 0.0:
        import numpy as _np
        _rng = _np.random.default_rng(1234)
        nloc = jnp.asarray(_rng.standard_normal((Nx, Ny))[x_start:x_end, y_start:y_end])
        uy = noise_amp * nloc * envelope
    else:
        uy = A * jnp.sin(2 * jnp.pi * X) * envelope

    p = jnp.ones((Nx_local, Ny_local)) * 10.0
    E = p / (gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)

    U = jnp.stack([rho, rho * ux, rho * uy, E], axis=0)

    return U
