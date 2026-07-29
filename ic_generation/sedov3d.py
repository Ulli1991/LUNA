import jax.numpy as jnp

from ._grid import local_range

# Physical parameters used when cfg.units is enabled.
# SN-style thermal-bomb injection (top-hat sphere) into a uniform warm-ISM
# background.
N_AMBIENT_CGS = 1.0  # particles per cm^3 (mean number density)
MU = 1.0  # mean molecular weight in proton masses (atomic H)
T_AMBIENT_K = 1.0e4  # K
E_INJECT_ERG = 1.0e51  # supernova energy

PROTON_MASS_CGS = 1.6726219e-24
BOLTZMANN_CGS = 1.380649e-16


def sedov_blast_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """3D Sedov-Taylor blast. cfg.units enabled: SN-style 1e51 erg injection into
    a central 2x2x2 cube on a warm-ISM background; units off: legacy top-hat
    sphere (r=6dx, E=0.5 code units). Returns (5, Nx_l, Ny_l, Nz_l) conservatives.
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
        rho_phys = N_AMBIENT_CGS * MU * PROTON_MASS_CGS  # g/cm^3
        p_phys = N_AMBIENT_CGS * BOLTZMANN_CGS * T_AMBIENT_K  # erg/cm^3
        rho_bg = rho_phys / cfg.units.UnitDensity_in_cgs
        p_bg = p_phys / cfg.units.UnitPressure_in_cgs
        E_inject = E_INJECT_ERG / cfg.units.UnitEnergy_in_cgs  # total energy
    else:
        rho_bg = 1.0
        p_bg = 1e-2
        E_inject = 0.5

    if is_global_call:
        print("3D Sedov Blast Configuration:")
        print(f"  Domain: [{Lx} x {Ly} x {Lz}]")
        print(f"  Grid: [{Nx} x {Ny} x {Nz}]")
        print(f"  dx={dx:.6e}, dy={dy:.6e}, dz={dz:.6e}")
        print(f"  Total cells: {Nx*Ny*Nz:,}")
        print(f"  Units: {'physical (Arepo)' if units_on else 'dimensionless'}")
        print(f"  rho_bg = {rho_bg:.6e} (code units)")
        print(f"  p_bg   = {p_bg:.6e} (code units)")
        print(f"  E_inject = {E_inject:.6e} (code units)")

    # Local-slice ranges.
    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    z_start, z_end = local_range(Nz, z_slice)
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
        # Central 2x2x2 = 8-cell cube injection. For even Nx the eight cells
        # centred on the domain centre have indices {Nx/2-1, Nx/2} on each
        # axis. Total E_inject is split uniformly across these 8 cells.
        ix0, iy0, iz0 = Nx // 2 - 1, Ny // 2 - 1, Nz // 2 - 1

        ii = jnp.arange(x_start, x_end)[:, None, None]
        jj = jnp.arange(y_start, y_end)[None, :, None]
        kk = jnp.arange(z_start, z_end)[None, None, :]
        in_cube = (
            (ii >= ix0)
            & (ii <= ix0 + 1)
            & (jj >= iy0)
            & (jj <= iy0 + 1)
            & (kk >= iz0)
            & (kk <= iz0 + 1)
        )

        n_inject = 8
        V_inject = n_inject * dx * dy * dz
        p_inject = (gamma - 1.0) * (E_inject / V_inject)

        if is_global_call:
            print(
                f"  Sedov inject: central 2x2x2 cube ({n_inject} cells), "
                f"V_inject = {V_inject:.4e}"
            )
            print(f"  p_inject (code) = {p_inject:.4e}")

        p = jnp.where(in_cube, p_inject + p_bg, p_bg)
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
