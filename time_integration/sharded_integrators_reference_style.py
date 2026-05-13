"""
MUSCL-Hancock following the EXACT structure of the reference code.

Key: Main timestep function is NOT @jax.jit!
Instead, many small @jax.jit helper functions are called.

This matches: https://github.com/pmocz/jax-euler-benchmarks/blob/main/euler_distributed.py
"""
import jax
import jax.numpy as jnp


def create_muscl_hancock_reference_style(gamma, order, limiter,
                                          dx, dy, dz, inv_dx, inv_dy, inv_dz,
                                          boundary, spatial_dim, CFL):
    """
    Create MUSCL-Hancock timestep function matching reference code structure.

    Reference structure:
    - Main update() is NOT @jax.jit
    - Calls many small @jax.jit helper functions
    - Halo exchange (rolls) happen INSIDE the @jax.jit helpers
    """
    from hydro.conversions import cons_to_prim

    # Small @jax.jit helper for dt computation
    @jax.jit
    def compute_dt_cfl(W):
        """Compute CFL timestep - SMALL @jax.jit function."""
        rho = W[0]
        if spatial_dim == 2:
            vx, vy, P = W[1], W[2], W[3]
            cs = jnp.sqrt(gamma * P / rho)
            return CFL * jnp.min(dx / (jnp.sqrt(vx**2 + vy**2) + cs))
        elif spatial_dim == 1:
            vx, P = W[1], W[2]
            cs = jnp.sqrt(gamma * P / rho)
            return CFL * jnp.min(dx / (jnp.abs(vx) + cs))
        elif spatial_dim == 3:
            vx, vy, vz, P = W[1], W[2], W[3], W[4]
            cs = jnp.sqrt(gamma * P / rho)
            max_speed_x = jnp.abs(vx) + cs
            max_speed_y = jnp.abs(vy) + cs
            max_speed_z = jnp.abs(vz) + cs
            dt_local = CFL * jnp.minimum(
                jnp.minimum(dx / max_speed_x, dy / max_speed_y),
                dz / max_speed_z
            )
            return jnp.min(dt_local)
        else:
            raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")

    # Small @jax.jit helper for primitive conversion
    @jax.jit
    def get_primitive(U):
        """Get primitives - SMALL @jax.jit function."""
        return cons_to_prim(U, gamma)

    # Small @jax.jit helper for slopes (WITH ROLLS INSIDE!)
    @jax.jit
    def get_slopes(W):
        """
        Compute LIMITED slopes using roll-based neighbor access - SMALL @jax.jit function.

        Uses proper slope limiters (minmod/moncen/vanleer) for stability at shocks.
        Rolls happen HERE, inside @jax.jit!
        Returns DIFFERENCES (ΔW), not derivatives (dW/dx).
        """
        from hydro.slopes import slope_limiter

        if spatial_dim == 1:
            # Left and right differences using roll
            dl_x = W - jnp.roll(W, 1, axis=1)
            dr_x = jnp.roll(W, -1, axis=1) - W
            dWx = slope_limiter(dl_x, dr_x, limiter)
            return dWx, None, None

        elif spatial_dim == 2:
            dl_x = W - jnp.roll(W, 1, axis=1)
            dr_x = jnp.roll(W, -1, axis=1) - W
            dWx = slope_limiter(dl_x, dr_x, limiter)

            dl_y = W - jnp.roll(W, 1, axis=2)
            dr_y = jnp.roll(W, -1, axis=2) - W
            dWy = slope_limiter(dl_y, dr_y, limiter)

            return dWx, dWy, None

        elif spatial_dim == 3:
            dl_x = W - jnp.roll(W, 1, axis=1)
            dr_x = jnp.roll(W, -1, axis=1) - W
            dWx = slope_limiter(dl_x, dr_x, limiter)

            dl_y = W - jnp.roll(W, 1, axis=2)
            dr_y = jnp.roll(W, -1, axis=2) - W
            dWy = slope_limiter(dl_y, dr_y, limiter)

            dl_z = W - jnp.roll(W, 1, axis=3)
            dr_z = jnp.roll(W, -1, axis=3) - W
            dWz = slope_limiter(dl_z, dr_z, limiter)

            return dWx, dWy, dWz
        else:
            raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")

    # Small @jax.jit helper for predictor step
    @jax.jit
    def apply_predictor(W, dWx, dWy, dWz, dt):
        """
        MUSCL-Hancock predictor step - SMALL @jax.jit function.

        Evolve primitives half a timestep using characteristic evolution.
        """
        from hydro.jacobian import apply_primitive_jacobian_2d, apply_primitive_jacobian_1d, apply_primitive_jacobian_3d

        if spatial_dim == 2:
            dWx_scaled = dWx * inv_dx
            dWy_scaled = dWy * inv_dy
            A_dW = apply_primitive_jacobian_2d(W, dWx_scaled, dWy_scaled, gamma)
            return W - 0.5 * dt * A_dW
        elif spatial_dim == 1:
            dWx_scaled = dWx * inv_dx
            A_dW = apply_primitive_jacobian_1d(W, dWx_scaled, gamma)
            return W - 0.5 * dt * A_dW
        elif spatial_dim == 3:
            dWx_scaled = dWx * inv_dx
            dWy_scaled = dWy * inv_dy
            dWz_scaled = dWz * inv_dz
            A_dW = apply_primitive_jacobian_3d(W, dWx_scaled, dWy_scaled, dWz_scaled, gamma)
            return W - 0.5 * dt * A_dW
        else:
            raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")

    # Small @jax.jit helper for reconstruction (WITH ROLLS INSIDE!)
    @jax.jit
    def reconstruct_to_faces(W, dWx, dWy, dWz):
        """
        Reconstruct primitives to face centers - SMALL @jax.jit function.

        IMPORTANT: Rolls happen HERE, inside @jax.jit!

        Slopes are DIFFERENCES, so reconstruction is W +/- 0.5*slope.
        For interface i+1/2:
        - W_L[i] = W[i] + 0.5*slope[i] (right extrapolation from cell i)
        - W_R[i] = W[i+1] - 0.5*slope[i+1] (left extrapolation from cell i+1)
        """
        if spatial_dim == 2:
            half_dWx = 0.5 * dWx
            W_XL = W + half_dWx
            W_XR = W - half_dWx
            W_XR = jnp.roll(W_XR, -1, axis=1)

            half_dWy = 0.5 * dWy
            W_YL = W + half_dWy
            W_YR = W - half_dWy
            W_YR = jnp.roll(W_YR, -1, axis=2)

            return W_XL, W_XR, W_YL, W_YR, None, None
        elif spatial_dim == 1:
            half_dWx = 0.5 * dWx
            W_L = W + half_dWx
            W_R = W - half_dWx
            W_R = jnp.roll(W_R, -1, axis=1)

            return W_L, W_R, None, None, None, None
        elif spatial_dim == 3:
            half_dWx = 0.5 * dWx
            W_XL = W + half_dWx
            W_XR = W - half_dWx
            W_XR = jnp.roll(W_XR, -1, axis=1)

            half_dWy = 0.5 * dWy
            W_YL = W + half_dWy
            W_YR = W - half_dWy
            W_YR = jnp.roll(W_YR, -1, axis=2)

            half_dWz = 0.5 * dWz
            W_ZL = W + half_dWz
            W_ZR = W - half_dWz
            W_ZR = jnp.roll(W_ZR, -1, axis=3)

            return W_XL, W_XR, W_YL, W_YR, W_ZL, W_ZR
        else:
            raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")

    # Small @jax.jit helper for flux computation
    @jax.jit
    def compute_fluxes_x(W_L, W_R):
        """Compute HLLC fluxes in x-direction - SMALL @jax.jit function."""
        from hydro.fluxes import hllc_flux_primitive
        return hllc_flux_primitive(W_L, W_R, axis=0, gamma=gamma)

    @jax.jit
    def compute_fluxes_y(W_L, W_R):
        """Compute HLLC fluxes in y-direction - SMALL @jax.jit function."""
        from hydro.fluxes import hllc_flux_primitive
        return hllc_flux_primitive(W_L, W_R, axis=1, gamma=gamma)

    @jax.jit
    def compute_fluxes_z(W_L, W_R):
        """Compute HLLC fluxes in z-direction - SMALL @jax.jit function."""
        from hydro.fluxes import hllc_flux_primitive
        return hllc_flux_primitive(W_L, W_R, axis=2, gamma=gamma)

    # Small @jax.jit helper for applying fluxes (WITH ROLLS INSIDE!)
    @jax.jit
    def apply_flux_update(U, Fx, Fy, Fz, dt):
        """
        Apply fluxes to update conservatives - SMALL @jax.jit function.

        IMPORTANT: Rolls happen HERE, inside @jax.jit!

        NOTE: Reference code uses dt*dx because they store total quantities (Mass = rho*dx²).
        We store densities, so we use dt/dx.
        """
        if spatial_dim == 2:
            U_new = U - dt * inv_dx * Fx
            U_new = U_new + dt * inv_dx * jnp.roll(Fx, 1, axis=1)

            U_new = U_new - dt * inv_dy * Fy
            U_new = U_new + dt * inv_dy * jnp.roll(Fy, 1, axis=2)

            return U_new
        elif spatial_dim == 1:
            U_new = U - dt * inv_dx * Fx
            U_new = U_new + dt * inv_dx * jnp.roll(Fx, 1, axis=1)
            return U_new
        elif spatial_dim == 3:
            U_new = U - dt * inv_dx * Fx
            U_new = U_new + dt * inv_dx * jnp.roll(Fx, 1, axis=1)

            U_new = U_new - dt * inv_dy * Fy
            U_new = U_new + dt * inv_dy * jnp.roll(Fy, 1, axis=2)

            U_new = U_new - dt * inv_dz * Fz
            U_new = U_new + dt * inv_dz * jnp.roll(Fz, 1, axis=3)

            return U_new
        else:
            raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")

    # Main timestep orchestrator (NOT @jax.jit! Like reference!)
    def time_step_reference_style(U, t, t_snapshot, t_end):
        """
        Main timestep function - NOT @jax.jit decorated!

        Calls many small @jax.jit helper functions.
        This matches the reference code structure exactly.
        """
        # 1. Get primitives (small @jax.jit)
        W = get_primitive(U)

        # 2. Compute dt (small @jax.jit with global reduction)
        dt = compute_dt_cfl(W)

        # Adjust for end time
        dt = jnp.where((t_end - t < 10 * dt) & (t + dt > t_end), t_end - t, dt)

        if order == 0:
            # First-order: compute fluxes at all interfaces
            Fx = compute_fluxes_x(W, jnp.roll(W, -1, axis=1))
            if spatial_dim >= 2:
                Fy = compute_fluxes_y(W, jnp.roll(W, -1, axis=2))
            else:
                Fy = None
            if spatial_dim == 3:
                Fz = compute_fluxes_z(W, jnp.roll(W, -1, axis=3))
            else:
                Fz = None
        else:
            # Second-order: MUSCL-Hancock
            # 3. Compute slopes (small @jax.jit with rolls!)
            dWx, dWy, dWz = get_slopes(W)

            # 4. Predictor step (small @jax.jit)
            W_half = apply_predictor(
                W, dWx,
                dWy if dWy is not None else dWx * 0,
                dWz if dWz is not None else dWx * 0,
                dt
            )

            # 5. Reconstruct to faces (small @jax.jit with rolls!)
            W_XL, W_XR, W_YL, W_YR, W_ZL, W_ZR = reconstruct_to_faces(
                W_half, dWx,
                dWy if dWy is not None else dWx,
                dWz if dWz is not None else dWx
            )

            # 6. Compute fluxes (small @jax.jit)
            Fx = compute_fluxes_x(W_XL, W_XR)
            if spatial_dim >= 2:
                Fy = compute_fluxes_y(W_YL, W_YR)
            else:
                Fy = None
            if spatial_dim == 3:
                Fz = compute_fluxes_z(W_ZL, W_ZR)
            else:
                Fz = None

        # 7. Apply fluxes (small @jax.jit with rolls!)
        U_new = apply_flux_update(
            U, Fx,
            Fy if Fy is not None else Fx * 0,
            Fz if Fz is not None else Fx * 0,
            dt
        )

        return U_new, dt

    return time_step_reference_style
