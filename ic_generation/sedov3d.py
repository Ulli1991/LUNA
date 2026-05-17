import jax.numpy as jnp


# Physical parameters used when cfg.units is enabled.
# SN-style thermal-bomb injection (top-hat sphere) into a uniform warm-ISM
# background.
N_AMBIENT_CGS = 1.0       # particles per cm^3 (mean number density)
MU = 1.0                  # mean molecular weight in proton masses (atomic H)
T_AMBIENT_K = 1.0e4       # K
E_INJECT_ERG = 1.0e51     # supernova energy

PROTON_MASS_CGS = 1.6726219e-24
BOLTZMANN_CGS = 1.380649e-16


def sedov_blast_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """
    Sedov-Taylor blast wave in 3D.

    Two modes, chosen by cfg.units.is_enabled:

    - Physical-units mode (cfg.units enabled): SN-style injection.
      Uniform background at N_AMBIENT_CGS particles/cm^3 and T_AMBIENT_K
      with E_INJECT_ERG deposited into a single central cell.

    - Dimensionless mode (no units): legacy top-hat sphere of radius 6*dx
      with total energy 0.5 in code units (keeps existing test reproducible).

    Parameters
    ----------
    cfg : Config
    x_slice, y_slice, z_slice : tuple (start, end), optional
        Per-process slice for distributed initialization.

    Returns
    -------
    U : (5, Nx_local, Ny_local, Nz_local) array of conservatives.
    """
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    is_global_call = x_slice is None and y_slice is None and z_slice is None
    units_on = getattr(cfg, "units", None) is not None and cfg.units.is_enabled

    # Background state and injected energy, both in code units.
    if units_on:
        rho_phys = N_AMBIENT_CGS * MU * PROTON_MASS_CGS                 # g/cm^3
        p_phys = N_AMBIENT_CGS * BOLTZMANN_CGS * T_AMBIENT_K            # erg/cm^3
        rho_bg = rho_phys / cfg.units.UnitDensity_in_cgs
        p_bg = p_phys / cfg.units.UnitPressure_in_cgs
        E_inject = E_INJECT_ERG / cfg.units.UnitEnergy_in_cgs           # total energy
    else:
        rho_bg = 1.0
        p_bg = 1e-2
        E_inject = 0.5

    if is_global_call:
        print(f"3D Sedov Blast Configuration:")
        print(f"  Domain: [{Lx} x {Ly} x {Lz}]")
        print(f"  Grid: [{Nx} x {Ny} x {Nz}]")
        print(f"  dx={dx:.6e}, dy={dy:.6e}, dz={dz:.6e}")
        print(f"  Total cells: {Nx*Ny*Nz:,}")
        print(f"  Units: {'physical (Arepo)' if units_on else 'dimensionless'}")
        print(f"  rho_bg = {rho_bg:.6e} (code units)")
        print(f"  p_bg   = {p_bg:.6e} (code units)")
        print(f"  E_inject = {E_inject:.6e} (code units)")

    # Local-slice ranges.
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
    Nx_local = x_end - x_start
    Ny_local = y_end - y_start
    Nz_local = z_end - z_start

    # Background fields.
    rho = jnp.full((Nx_local, Ny_local, Nz_local), rho_bg)
    vx = jnp.zeros_like(rho)
    vy = jnp.zeros_like(rho)
    vz = jnp.zeros_like(rho)
    p = jnp.full_like(rho, p_bg)

    if units_on:
        # Small-sphere injection: r_blast = R_INJECT_CELLS * dx. Total E_inject
        # is distributed uniformly inside (top hat). This keeps the injection
        # numerically tractable while still being "a few cells" wide.
        R_INJECT_CELLS = 2.0
        r_blast = R_INJECT_CELLS * max(dx, dy, dz)

        x_full = jnp.linspace(0, Lx, Nx, endpoint=False) + 0.5 * dx
        y_full = jnp.linspace(0, Ly, Ny, endpoint=False) + 0.5 * dy
        z_full = jnp.linspace(0, Lz, Nz, endpoint=False) + 0.5 * dz
        X, Y, Z = jnp.meshgrid(
            x_full[x_start:x_end],
            y_full[y_start:y_end],
            z_full[z_start:z_end],
            indexing="ij",
        )
        r_centre = jnp.sqrt((X - 0.5 * Lx) ** 2
                            + (Y - 0.5 * Ly) ** 2
                            + (Z - 0.5 * Lz) ** 2)

        # Count cells inside the injection sphere using the *global* grid so
        # the energy normalization is identical on every shard.
        x_gc = jnp.linspace(0, Lx, Nx, endpoint=False) + 0.5 * dx
        y_gc = jnp.linspace(0, Ly, Ny, endpoint=False) + 0.5 * dy
        z_gc = jnp.linspace(0, Lz, Nz, endpoint=False) + 0.5 * dz
        Xg, Yg, Zg = jnp.meshgrid(x_gc, y_gc, z_gc, indexing="ij")
        n_inject = int(jnp.sum((jnp.sqrt((Xg - 0.5 * Lx) ** 2
                                         + (Yg - 0.5 * Ly) ** 2
                                         + (Zg - 0.5 * Lz) ** 2) < r_blast)))
        V_inject = n_inject * dx * dy * dz
        p_inject = (gamma - 1.0) * (E_inject / V_inject)

        if is_global_call:
            print(f"  Sedov inject: r_blast = {r_blast / max(dx,dy,dz):.2f} cells "
                  f"= {r_blast:.4e} (code) ; {n_inject} cells; "
                  f"V_inject = {V_inject:.4e}")
            print(f"  p_inject (code) = {p_inject:.4e}")

        p = jnp.where(r_centre < r_blast, p_inject + p_bg, p_bg)
    else:
        # Legacy dimensionless mode: top-hat sphere, r_blast = 6*max(dx,dy,dz).
        x_full = jnp.linspace(0, Lx, Nx, endpoint=False) + 0.5 * dx
        y_full = jnp.linspace(0, Ly, Ny, endpoint=False) + 0.5 * dy
        z_full = jnp.linspace(0, Lz, Nz, endpoint=False) + 0.5 * dz
        X, Y, Z = jnp.meshgrid(
            x_full[x_start:x_end],
            y_full[y_start:y_end],
            z_full[z_start:z_end],
            indexing="ij",
        )
        r = jnp.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2 + (Z - 0.5 * Lz) ** 2)
        r_blast = 6.0 * max(dx, dy, dz)
        V_blast = (4.0 / 3.0) * jnp.pi * r_blast**3
        e_blast = E_inject / V_blast
        p_blast = (gamma - 1.0) * e_blast
        p = jnp.where(r < r_blast, p_blast + p_bg, p_bg)

    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
    U = jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)
    return U
