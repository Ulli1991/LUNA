"""Well-balanced MUSCL-Hancock integrator for stratified low-Mach flows.

Based on Velasco-Romero & Teyssier 2025 (MNRAS 537, 2387), adapted to FV2.
Supports 2D and 3D. Convention: "vertical" is the LAST spatial axis (y in
2D, z in 3D). Gravity is along that axis with magnitude `gravity_z` (the
field is named gravity_z for both dims for backward compat; negative =
downward).

Algorithm (per step):
  1) primitives:        W   = cons_to_prim(U)
                        W'  = W - W_eq                      (perturbation)
  2) slopes on W' only: dW' = slope_limiter(W'_neighbour - W'_center)
  3) MUSCL-Hancock predictor using Jacobian at FULL state:
        W'_half = W'_c - (dt/2) * A(W_c) · ∇W'
  4) face values:       W_L = W_eq(face) + (W'_half + ½·dW')
                        W_R = W_eq(face) + (W'_half - ½·dW')
     For non-vertical faces W_eq(face) = W_eq(cell); for vertical faces it's
     averaged from adjacent cells.
  5) Riemann flux F at full face state (HLLC or L-HLLC).
  6) Equilibrium-flux subtraction: F -= p_eq · δ_normal on the normal-
     momentum slot of each face flux.
  7) Gravity source on the perturbation:
        d(ρ v_vert) += ρ' · g · dt
        d E          += ρ v_vert · g · dt
"""
import jax
import jax.numpy as jnp


def create_wb_muscl_hancock_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                                 dx, dy, dz, inv_dx, inv_dy, inv_dz,
                                 boundary, spatial_dim, CFL,
                                 U_eq,
                                 gravity_z,
                                 sharding=None,
                                 riemann_solver="hllc"):
    if spatial_dim not in (2, 3):
        raise NotImplementedError(
            "Well-balanced integrator supports 2D and 3D only."
        )

    from hydro.conversions import cons_to_prim
    from hydro.slopes import compute_slopes_from_padded
    if riemann_solver == "lhllc":
        from hydro.fluxes_lhllc import lhllc_flux_primitive as flux_fn
    else:
        from hydro.fluxes import hllc_flux_primitive as flux_fn
    if spatial_dim == 3:
        from hydro.jacobian import apply_primitive_jacobian_3d as apply_jacobian
    else:
        from hydro.jacobian import apply_primitive_jacobian_2d as apply_jacobian

    # Note: W_eq is recomputed inside step_fn each call (cheap) so JAX traces
    # it together with the live U and gives it the matching NamedSharding.
    # Pre-computing at factory time desyncs the sharding spec on multi-GPU
    # runs, which crashes the W_pad - W_eq_pad subtraction inside the JIT.

    # Slot index of the vertical momentum and pressure components
    if spatial_dim == 2:
        # U = (rho, mvx, mvy, E); F = (mass, mvx flux, mvy flux, E flux)
        vert_mom_idx = 2
        energy_idx = 3
        # primitive: (rho, vx, vy, p), pressure idx 3
        p_idx = 3
        vert_vel_idx = 2     # vy
    else:
        # U = (rho, mvx, mvy, mvz, E)
        vert_mom_idx = 3
        energy_idx = 4
        p_idx = 4
        vert_vel_idx = 3     # vz

    @jax.jit
    def compute_dt_fn(U):
        W = cons_to_prim(U, gamma)
        rho = W[0]
        P = W[p_idx]
        cs = jnp.sqrt(gamma * P / rho)
        if spatial_dim == 2:
            vx = W[1]; vy = W[2]
            dt_local = CFL * jnp.minimum(
                dx / (jnp.abs(vx) + cs), dy / (jnp.abs(vy) + cs)
            )
        else:
            vx = W[1]; vy = W[2]; vz = W[3]
            dt_local = CFL * jnp.minimum(
                jnp.minimum(dx / (jnp.abs(vx) + cs),
                            dy / (jnp.abs(vy) + cs)),
                dz / (jnp.abs(vz) + cs),
            )
        return jnp.min(dt_local)

    @jax.jit
    def step_fn(U, dt):
        W = cons_to_prim(U, gamma)
        W_eq = cons_to_prim(U_eq, gamma)
        W_pad = pad_halo_fn(W, boundary, spatial_dim)
        W_eq_pad = pad_halo_fn(W_eq, boundary, spatial_dim)
        W_prime_pad = W_pad - W_eq_pad

        dWx, dWy, dWz = compute_slopes_from_padded(W_prime_pad, limiter)

        # Interior cell-center slicing
        if spatial_dim == 2:
            W_c       = W_pad[:, 1:-1, 1:-1]
            W_eq_c    = W_eq_pad[:, 1:-1, 1:-1]
        else:
            W_c       = W_pad[:, 1:-1, 1:-1, 1:-1]
            W_eq_c    = W_eq_pad[:, 1:-1, 1:-1, 1:-1]
        W_prime_c = W_c - W_eq_c

        # MUSCL-Hancock predictor on perturbation (Jacobian at full W)
        if spatial_dim == 2:
            A_dW = apply_jacobian(W_c, dWx * inv_dx, dWy * inv_dy, gamma)
        else:
            A_dW = apply_jacobian(
                W_c, dWx * inv_dx, dWy * inv_dy, dWz * inv_dz, gamma
            )
        W_prime_half = W_prime_c - 0.5 * dt * A_dW

        # Face-state perturbation reconstructions
        hx = 0.5 * dWx; hy = 0.5 * dWy
        if spatial_dim == 3:
            hz = 0.5 * dWz
        Wp_r_x = W_prime_half + hx
        Wp_l_x = W_prime_half - hx
        Wp_r_y = W_prime_half + hy
        Wp_l_y = W_prime_half - hy
        if spatial_dim == 3:
            Wp_r_z = W_prime_half + hz
            Wp_l_z = W_prime_half - hz

        # Boundary ghost perturbation (periodic wrap or unreconstructed)
        if spatial_dim == 2:
            if boundary == 0:
                WL_x_bc = Wp_r_x[:, -1:, :]
                WR_x_bc = Wp_l_x[:,  0:1, :]
                WL_y_bc = Wp_r_y[:, :, -1:]
                WR_y_bc = Wp_l_y[:, :,  0:1]
            else:
                WL_x_bc = W_prime_pad[:,  0:1, 1:-1]
                WR_x_bc = W_prime_pad[:, -1:,  1:-1]
                WL_y_bc = W_prime_pad[:, 1:-1,  0:1]
                WR_y_bc = W_prime_pad[:, 1:-1, -1:]
        else:
            if boundary == 0:
                WL_x_bc = Wp_r_x[:, -1:, :, :]
                WR_x_bc = Wp_l_x[:,  0:1, :, :]
                WL_y_bc = Wp_r_y[:, :, -1:, :]
                WR_y_bc = Wp_l_y[:, :,  0:1, :]
                WL_z_bc = Wp_r_z[:, :, :, -1:]
                WR_z_bc = Wp_l_z[:, :, :,  0:1]
            else:
                WL_x_bc = W_prime_pad[:,  0:1, 1:-1, 1:-1]
                WR_x_bc = W_prime_pad[:, -1:,  1:-1, 1:-1]
                WL_y_bc = W_prime_pad[:, 1:-1,  0:1, 1:-1]
                WR_y_bc = W_prime_pad[:, 1:-1, -1:,  1:-1]
                WL_z_bc = W_prime_pad[:, 1:-1, 1:-1,  0:1]
                WR_z_bc = W_prime_pad[:, 1:-1, 1:-1, -1:]

        Wp_L_x = jnp.concatenate([WL_x_bc, Wp_r_x], axis=1)
        Wp_R_x = jnp.concatenate([Wp_l_x, WR_x_bc], axis=1)
        Wp_L_y = jnp.concatenate([WL_y_bc, Wp_r_y], axis=2)
        Wp_R_y = jnp.concatenate([Wp_l_y, WR_y_bc], axis=2)
        if spatial_dim == 3:
            Wp_L_z = jnp.concatenate([WL_z_bc, Wp_r_z], axis=3)
            Wp_R_z = jnp.concatenate([Wp_l_z, WR_z_bc], axis=3)

        # W_eq at faces
        if spatial_dim == 2:
            W_eq_face_x = W_eq_pad[:, :-1, 1:-1]    # x face: W_eq has same z, take left cell
            # In 2D vertical = y axis. Vertical-face W_eq is averaged.
            W_eq_face_y = 0.5 * (W_eq_pad[:, 1:-1, :-1] + W_eq_pad[:, 1:-1, 1:])
        else:
            W_eq_face_x = W_eq_pad[:, :-1, 1:-1, 1:-1]
            W_eq_face_y = W_eq_pad[:, 1:-1, :-1, 1:-1]
            # Vertical-face W_eq averaged
            W_eq_face_z = 0.5 * (W_eq_pad[:, 1:-1, 1:-1, :-1]
                               + W_eq_pad[:, 1:-1, 1:-1, 1:])

        # Full-state face values
        W_L_x = W_eq_face_x + Wp_L_x
        W_R_x = W_eq_face_x + Wp_R_x
        W_L_y = W_eq_face_y + Wp_L_y
        W_R_y = W_eq_face_y + Wp_R_y
        if spatial_dim == 3:
            W_L_z = W_eq_face_z + Wp_L_z
            W_R_z = W_eq_face_z + Wp_R_z

        # Riemann fluxes (full state)
        Fx = flux_fn(W_L_x, W_R_x, axis=0, gamma=gamma)
        Fy = flux_fn(W_L_y, W_R_y, axis=1, gamma=gamma)
        if spatial_dim == 3:
            Fz = flux_fn(W_L_z, W_R_z, axis=2, gamma=gamma)

        # Equilibrium-flux subtraction: only normal-momentum slot
        p_eq_face_x = W_eq_face_x[p_idx]
        p_eq_face_y = W_eq_face_y[p_idx]
        Fx = Fx.at[1].add(-p_eq_face_x)            # x-mom from x-face
        Fy = Fy.at[2].add(-p_eq_face_y)            # y-mom from y-face
        if spatial_dim == 3:
            p_eq_face_z = W_eq_face_z[p_idx]
            Fz = Fz.at[3].add(-p_eq_face_z)        # z-mom from z-face

        # Divergence of perturbation flux
        if spatial_dim == 2:
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy
            dU = -(dFdx + dFdy)
        else:
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz
            dU = -(dFdx + dFdy + dFdz)

        # Gravity source on the perturbation
        rho = U[0]
        rho_eq = U_eq[0]
        rho_prime = rho - rho_eq
        mom_vert = U[vert_mom_idx]
        dU = dU.at[vert_mom_idx].add(rho_prime * gravity_z)
        dU = dU.at[energy_idx].add(mom_vert * gravity_z)

        U_new = U + dt * dU
        if sharding is not None:
            U_new = jax.lax.with_sharding_constraint(U_new, sharding)
        return U_new

    return compute_dt_fn, step_fn
