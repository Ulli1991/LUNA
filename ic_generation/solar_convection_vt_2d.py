"""Velasco-Romero & Teyssier (2025) 2D stellar-convection test.

Three-layer polytropic atmosphere (z runs vertical, gravity points in -z):

    | Zone                | Gamma | rho_0 | P_0  | z-range          |
    |---------------------|-------|-------|------|------------------|
    | Bottom radiative    | 1.20  | 1.81  | 5.94 | [z_0, z_1]       |
    | Convective middle   | 1.67  | 1.17  | 3.53 | [z_1, z_2]       |
    | Top radiative       | 1.01  | 0.09  | 0.05 | [z_2, z_3]       |

With (z_0, z_1, z_2, z_3) = (0, 1.64, 7.77, 11) and g = 1.

Inside each zone the profile is polytropic:
    T(z) = T_0,i * [ 1 - ((Gamma_i - 1)/Gamma_i) * rho_0,i * g / P_0,i * (z - z_{i-1}) ]
    rho(z) = rho_0,i * (T(z)/T_0,i)^(1/(Gamma_i-1))
    P(z)   = P_0,i   * (T(z)/T_0,i)^(Gamma_i/(Gamma_i-1))
where T_0,i = P_0,i / rho_0,i (ideal gas, mean molecular weight absorbed).

Convection driver: a small under-dense Gaussian bubble in the middle layer:
    delta_rho / rho_eq = -0.012 * exp( -r^2 / (2 r_b^2) ),  r < r_b
with r_b = 0.25 centered at (x_b, z_b) = (L/4, z_1 + 0.75).

For LUNA's 2D convention the vertical axis is `y`. The mapping:
    x_VT -> x in LUNA
    z_VT -> y in LUNA   (gravity_z config field is along y in 2D, magnitude g)

The two functions:
    solar_convection_vt_2d_equilibrium(cfg) -> U_eq  (no bubble, used by WB)
    solar_convection_vt_2d(cfg)              -> U   (equilibrium + bubble)
"""
import jax.numpy as jnp
import numpy as np


# V&T layer parameters
_VT = dict(
    z_bounds=(0.0, 1.64, 7.77, 11.0),
    Gamma=(1.20, 1.67, 1.01),
    rho_top=(1.81, 1.17, 0.09),       # rho_0 at the BOTTOM of each layer
    P_top=(5.94, 3.53, 0.05),         # P_0 at the BOTTOM of each layer
)


def _layer_profile_at_y(y, g, gamma_idx):
    """Return (rho, p) at coordinate y in V&T zone gamma_idx (0/1/2)."""
    z_lo = _VT["z_bounds"][gamma_idx]
    Gamma = _VT["Gamma"][gamma_idx]
    rho0 = _VT["rho_top"][gamma_idx]
    P0 = _VT["P_top"][gamma_idx]
    T0 = P0 / rho0
    # T(y) = T0 * (1 - ((Gamma-1)/Gamma) * rho0 * g / P0 * (y - z_lo))
    eta = 1.0 - ((Gamma - 1.0) / Gamma) * rho0 * g / P0 * (y - z_lo)
    eta = jnp.maximum(eta, 1e-8)        # guard against negative; clip to tiny positive
    rho = rho0 * eta ** (1.0 / (Gamma - 1.0))
    p   = P0   * eta ** (Gamma / (Gamma - 1.0))
    return rho, p


def _equilibrium_columns(cfg, y_slice):
    """1D (Ny_local,) arrays of (rho_eq, p_eq) along the vertical axis."""
    Ny = cfg.Ny
    Ly = cfg.Ly
    dy = Ly / Ny
    y_start, y_end = (y_slice if y_slice is not None else (0, Ny))
    y_local = (jnp.arange(y_start, y_end) + 0.5) * dy

    g = -cfg.gravity_z
    if g <= 0:
        raise ValueError("V&T solar IC expects cfg.gravity_z < 0 (downward).")

    z1 = _VT["z_bounds"][1]
    z2 = _VT["z_bounds"][2]
    rho_0, p_0 = _layer_profile_at_y(y_local, g, 0)
    rho_1, p_1 = _layer_profile_at_y(y_local, g, 1)
    rho_2, p_2 = _layer_profile_at_y(y_local, g, 2)
    rho = jnp.where(y_local < z1, rho_0,
          jnp.where(y_local < z2, rho_1, rho_2))
    p   = jnp.where(y_local < z1, p_0,
          jnp.where(y_local < z2, p_1, p_2))
    return rho, p


def solar_convection_vt_2d_equilibrium(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Hydrostatic 3-layer polytropic equilibrium (no bubble), 2D."""
    Nx = cfg.Nx
    x_start, x_end = (x_slice if x_slice is not None else (0, Nx))
    nx_l = x_end - x_start

    rho_y, p_y = _equilibrium_columns(cfg, y_slice)
    ny_l = rho_y.shape[0]

    rho = jnp.broadcast_to(rho_y[None, :], (nx_l, ny_l)).copy()
    p   = jnp.broadcast_to(p_y[None, :],   (nx_l, ny_l)).copy()
    zeros = jnp.zeros_like(rho)
    E = p / (cfg.gamma - 1.0)
    return jnp.stack([rho, zeros, zeros, E], axis=0)


def solar_convection_vt_2d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Equilibrium + V&T buoyant Gaussian bubble in the middle layer."""
    U_eq = solar_convection_vt_2d_equilibrium(cfg, x_slice, y_slice, z_slice)

    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    dx = Lx / Nx
    dy = Ly / Ny
    x_start, x_end = (x_slice if x_slice is not None else (0, Nx))
    y_start, y_end = (y_slice if y_slice is not None else (0, Ny))

    x_local = (jnp.arange(x_start, x_end) + 0.5) * dx
    y_local = (jnp.arange(y_start, y_end) + 0.5) * dy

    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")
    x_b = Lx / 2.0                       # x_VT = L/4 = Lx/2 since Lx = L/2
    z_b = _VT["z_bounds"][1] + 0.75      # z1 + 0.75
    r_b = 0.25
    amp = -0.012
    r2 = (X - x_b) ** 2 + (Y - z_b) ** 2
    bubble = jnp.where(r2 < r_b * r_b,
                       amp * jnp.exp(-r2 / (2.0 * r_b * r_b)),
                       0.0)

    # delta_rho / rho_eq = bubble  =>  rho_new = rho_eq * (1 + bubble)
    rho_new = U_eq[0] * (1.0 + bubble)
    # Keep pressure unchanged (V&T initialise the bubble at constant pressure)
    p_eq = (cfg.gamma - 1.0) * U_eq[3]
    E_new = p_eq / (cfg.gamma - 1.0)      # KE = 0
    return jnp.stack([rho_new,
                      jnp.zeros_like(rho_new),
                      jnp.zeros_like(rho_new),
                      E_new], axis=0)
