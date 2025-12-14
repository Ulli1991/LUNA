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
        else:
            raise NotImplementedError("3D not yet implemented")

    # Small @jax.jit helper for primitive conversion
    @jax.jit
    def get_primitive(U):
        """Get primitives - SMALL @jax.jit function."""
        return cons_to_prim(U, gamma)

    # Small @jax.jit helper for slopes (WITH ROLLS INSIDE!)
    @jax.jit
    def get_slopes(W):
        """
        Compute slopes (differences, not derivatives) - SMALL @jax.jit function.

        IMPORTANT: Rolls happen HERE, inside @jax.jit!
        Returns DIFFERENCES (ΔW), not derivatives (dW/dx).
        """
        if spatial_dim == 2:
            # For each variable, compute slopes as central differences
            Nvar = W.shape[0]
            dWx = jnp.zeros_like(W)
            dWy = jnp.zeros_like(W)

            for i in range(Nvar):
                # Central differences with rolls - return DIFFERENCES not derivatives!
                dWx = dWx.at[i].set(
                    (jnp.roll(W[i], -1, axis=0) - jnp.roll(W[i], 1, axis=0)) / 2
                )
                dWy = dWy.at[i].set(
                    (jnp.roll(W[i], -1, axis=1) - jnp.roll(W[i], 1, axis=1)) / 2
                )

            return dWx, dWy
        elif spatial_dim == 1:
            Nvar = W.shape[0]
            dWx = jnp.zeros_like(W)

            for i in range(Nvar):
                dWx = dWx.at[i].set(
                    (jnp.roll(W[i], -1, axis=0) - jnp.roll(W[i], 1, axis=0)) / 2
                )

            return dWx, None
        else:
            raise NotImplementedError("3D not yet implemented")

    # Small @jax.jit helper for predictor step
    @jax.jit
    def apply_predictor(W, dWx, dWy, dt):
        """
        MUSCL-Hancock predictor step - SMALL @jax.jit function.

        Evolve primitives half a timestep using characteristic evolution.
        """
        from hydro.jacobian import apply_primitive_jacobian_2d, apply_primitive_jacobian_1d

        if spatial_dim == 2:
            Wc = W
            dWx_scaled = dWx * inv_dx
            dWy_scaled = dWy * inv_dy
            A_dW = apply_primitive_jacobian_2d(Wc, dWx_scaled, dWy_scaled, gamma)
            return Wc - 0.5 * dt * A_dW
        elif spatial_dim == 1:
            Wc = W
            dWx_scaled = dWx * inv_dx
            A_dW = apply_primitive_jacobian_1d(Wc, dWx_scaled, gamma)
            return Wc - 0.5 * dt * A_dW
        else:
            raise NotImplementedError("3D not yet implemented")

    # Small @jax.jit helper for reconstruction (WITH ROLLS INSIDE!)
    @jax.jit
    def reconstruct_to_faces(W, dWx, dWy):
        """
        Reconstruct primitives to face centers - SMALL @jax.jit function.

        IMPORTANT: Rolls happen HERE, inside @jax.jit!

        Slopes are DIFFERENCES, so reconstruction is W +/- 0.5*slope.
        For interface i+1/2:
        - W_L[i] = W[i] + 0.5*slope[i] (right extrapolation from cell i)
        - W_R[i] = W[i+1] - 0.5*slope[i+1] (left extrapolation from cell i+1)
        """
        if spatial_dim == 2:
            # X-faces
            half_dWx = 0.5 * dWx
            W_XL = W + half_dWx  # Right extrapolation
            W_XR = W - half_dWx  # Left extrapolation
            W_XR = jnp.roll(W_XR, -1, axis=1)  # Shift to get neighbor's value

            # Y-faces
            half_dWy = 0.5 * dWy
            W_YL = W + half_dWy  # Top extrapolation
            W_YR = W - half_dWy  # Bottom extrapolation
            W_YR = jnp.roll(W_YR, -1, axis=2)  # Shift to get neighbor's value

            return W_XL, W_XR, W_YL, W_YR
        elif spatial_dim == 1:
            half_dWx = 0.5 * dWx
            W_L = W + half_dWx  # Right extrapolation
            W_R = W - half_dWx  # Left extrapolation
            W_R = jnp.roll(W_R, -1, axis=1)  # Shift to get neighbor's value

            return W_L, W_R, None, None
        else:
            raise NotImplementedError("3D not yet implemented")

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

    # Small @jax.jit helper for applying fluxes (WITH ROLLS INSIDE!)
    @jax.jit
    def apply_flux_update(U, Fx, Fy, dt):
        """
        Apply fluxes to update conservatives - SMALL @jax.jit function.

        IMPORTANT: Rolls happen HERE, inside @jax.jit!

        NOTE: Reference code uses dt*dx because they store total quantities (Mass = rho*dx²).
        We store densities, so we use dt/dx.
        """
        if spatial_dim == 2:
            # X-direction update
            U_new = U - dt * inv_dx * Fx
            U_new = U_new + dt * inv_dx * jnp.roll(Fx, 1, axis=1)  # Roll like reference!

            # Y-direction update
            U_new = U_new - dt * inv_dy * Fy
            U_new = U_new + dt * inv_dy * jnp.roll(Fy, 1, axis=2)  # Roll like reference!

            return U_new
        elif spatial_dim == 1:
            U_new = U - dt * inv_dx * Fx
            U_new = U_new + dt * inv_dx * jnp.roll(Fx, 1, axis=1)
            return U_new
        else:
            raise NotImplementedError("3D not yet implemented")

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
            # Flux arrays must have same shape as U for apply_flux_update
            if spatial_dim == 2:
                # X-direction fluxes (right face of each cell)
                Fx = compute_fluxes_x(W, jnp.roll(W, -1, axis=1))
                # Y-direction fluxes (top face of each cell)
                Fy = compute_fluxes_y(W, jnp.roll(W, -1, axis=2))
            else:
                Fx = compute_fluxes_x(W, jnp.roll(W, -1, axis=1))
                Fy = None
        else:
            # Second-order: MUSCL-Hancock
            # 3. Compute slopes (small @jax.jit with rolls!)
            dWx, dWy = get_slopes(W)

            # 4. Predictor step (small @jax.jit)
            W_half = apply_predictor(W, dWx, dWy if dWy is not None else dWx * 0, dt)

            # 5. Reconstruct to faces (small @jax.jit with rolls!)
            W_XL, W_XR, W_YL, W_YR = reconstruct_to_faces(W_half, dWx, dWy)

            # 6. Compute fluxes (small @jax.jit) - must match U shape!
            if spatial_dim == 2:
                # Compute fluxes at all cell interfaces
                Fx = compute_fluxes_x(W_XL, W_XR)
                Fy = compute_fluxes_y(W_YL, W_YR)
            else:
                Fx = compute_fluxes_x(W_XL, W_XR)
                Fy = None

        # 7. Apply fluxes (small @jax.jit with rolls!)
        U_new = apply_flux_update(U, Fx, Fy if Fy is not None else Fx * 0, dt)

        return U_new, dt

    return time_step_reference_style
