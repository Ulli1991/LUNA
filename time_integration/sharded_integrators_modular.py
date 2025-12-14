"""
Modular time integration with separate JIT functions.

This follows the reference implementation pattern:
- Main time_step function is NOT @jax.jit decorated
- Individual operations are separate @jax.jit functions
- Allows JAX to better overlap communication/computation and optimize collectives

Key difference from sharded_integrators.py:
- OLD: One big @jax.jit function with all operations
- NEW: Regular Python function calling multiple @jax.jit subfunctions
"""
import jax
import jax.numpy as jnp


def create_muscl_hancock_modular_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                                       dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim, CFL):
    """
    Create MUSCL-Hancock stepper with MODULAR JIT compilation.

    Returns a regular Python function that orchestrates multiple JIT-compiled operations.
    This allows JAX to asynchronously dispatch and potentially overlap operations.
    """
    from hydro.conversions import cons_to_prim

    # Separate JIT function for dt computation
    @jax.jit
    def compute_dt(U):
        """Compute timestep - separate JIT function."""
        W = cons_to_prim(U, gamma)
        rho = W[0]

        if spatial_dim == 1:
            vx = W[1]
            P = W[2]
            cs = jnp.sqrt(gamma * P / rho)
            max_speed = jnp.abs(vx) + cs
            dt_local = CFL * dx / max_speed
        elif spatial_dim == 2:
            vx = W[1]
            vy = W[2]
            P = W[3]
            cs = jnp.sqrt(gamma * P / rho)
            max_speed_x = jnp.abs(vx) + cs
            max_speed_y = jnp.abs(vy) + cs
            dt_local = CFL * jnp.minimum(dx / max_speed_x, dy / max_speed_y)
        else:
            raise NotImplementedError("3D not yet implemented")

        return jnp.min(dt_local)

    # Separate JIT function for dt adjustment
    @jax.jit
    def adjust_dt(dt, t, t_end):
        """Adjust dt to hit t_end exactly - separate JIT function."""
        return jnp.where(
            (t_end - t < 10 * dt) & (t + dt > t_end),
            t_end - t,
            dt
        )

    # Separate JIT function for cons→prim conversion
    @jax.jit
    def convert_to_prim(U):
        """Convert conservatives to primitives - separate JIT function."""
        return cons_to_prim(U, gamma)

    # Separate JIT function for halo exchange
    @jax.jit
    def apply_halo_exchange_prim(W):
        """
        Apply halo exchange on PRIMITIVES - separate JIT function.

        OPTIMIZATION: Exchange primitives instead of conservatives to avoid
        double conversion (U→W for dt, then U_pad→W_pad for flux).
        """
        return pad_halo_fn(W, boundary, spatial_dim)

    # Separate JIT function for flux computation and update
    @jax.jit
    def compute_flux_and_update_from_prim(U, W_pad, dt):
        """
        Compute fluxes from primitive W_pad and update state.

        Takes W_pad (primitives) directly, avoiding the U_pad→W_pad conversion.
        Supports both first-order and second-order (with MUSCL-Hancock predictor).
        """
        from hydro.fluxes import hllc_flux_primitive
        from hydro.slopes import compute_slopes_from_padded
        from hydro.jacobian import apply_primitive_jacobian_1d, apply_primitive_jacobian_2d
        import jax.lax as lax

        # First-order path
        def first_order_flux(W_pad):
            if spatial_dim == 2:
                W_L_x = W_pad[:, :-1, 1:-1]
                W_R_x = W_pad[:, 1:, 1:-1]
                Fx = hllc_flux_primitive(W_L_x, W_R_x, axis=0, gamma=gamma)
                dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

                W_L_y = W_pad[:, 1:-1, :-1]
                W_R_y = W_pad[:, 1:-1, 1:]
                Fy = hllc_flux_primitive(W_L_y, W_R_y, axis=1, gamma=gamma)
                dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

                return -(dFdx + dFdy)
            elif spatial_dim == 1:
                W_L = W_pad[:, :-1]
                W_R = W_pad[:, 1:]
                Fx = hllc_flux_primitive(W_L, W_R, axis=0, gamma=gamma)
                dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
                return -dFdx
            else:
                raise NotImplementedError("3D not yet implemented")

        # Second-order path with MUSCL-Hancock predictor
        def second_order_flux(W_pad):
            dWx, dWy, dWz = compute_slopes_from_padded(W_pad, limiter)

            if spatial_dim == 2:
                Wc = W_pad[:, 1:-1, 1:-1]

                # MUSCL-Hancock predictor step
                dWx_scaled = dWx * inv_dx
                dWy_scaled = dWy * inv_dy
                A_dW = apply_primitive_jacobian_2d(Wc, dWx_scaled, dWy_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW

                half_dWx = 0.5 * dWx
                half_dWy = 0.5 * dWy

                # X-direction reconstruction
                WL_x = jnp.concatenate([W_pad[:, 0:1, 1:-1], W_half + half_dWx], axis=1)
                WR_x = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:, 1:-1]], axis=1)
                Fx = hllc_flux_primitive(WL_x, WR_x, axis=0, gamma=gamma)
                dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

                # Y-direction reconstruction
                WL_y = jnp.concatenate([W_pad[:, 1:-1, 0:1], W_half + half_dWy], axis=2)
                WR_y = jnp.concatenate([W_half - half_dWy, W_pad[:, 1:-1, -1:]], axis=2)
                Fy = hllc_flux_primitive(WL_y, WR_y, axis=1, gamma=gamma)
                dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

                return -(dFdx + dFdy)

            elif spatial_dim == 1:
                Wc = W_pad[:, 1:-1]

                # MUSCL-Hancock predictor step
                dWx_scaled = dWx * inv_dx
                A_dW = apply_primitive_jacobian_1d(Wc, dWx_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW

                half_dWx = 0.5 * dWx
                WL = jnp.concatenate([W_pad[:, 0:1], W_half + half_dWx], axis=1)
                WR = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:]], axis=1)
                Fx = hllc_flux_primitive(WL, WR, axis=0, gamma=gamma)
                dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
                return -dFdx
            else:
                raise NotImplementedError("3D not yet implemented")

        # Select based on order
        is_first_order = jnp.array(order == 0)
        dU = lax.cond(is_first_order, first_order_flux, second_order_flux, W_pad)

        return U + dt * dU

    # Main orchestrator function (NOT @jax.jit!)
    def time_step_muscl_hancock_modular(U, t, t_snapshot, t_end):
        """
        MUSCL-Hancock time step with MODULAR JIT compilation.

        This is a regular Python function that calls separate JIT-compiled operations.
        JAX can asynchronously dispatch these and potentially overlap them.

        OPTIMIZED FLOW (avoids double conversion):
        1. U → W (convert once)
        2. Compute dt from W
        3. Pad W → W_pad (exchange primitives, not conservatives!)
        4. Compute fluxes directly from W_pad (no second conversion!)
        5. Update U
        """
        # Step 1: Convert to primitives (JIT function)
        # Do this ONCE at the start
        W = convert_to_prim(U)

        # Step 2: Compute dt from primitives (JIT function with global reduction)
        dt = compute_dt(W)

        # Step 3: Adjust dt if needed (JIT function, scalar operation)
        dt = adjust_dt(dt, t, t_end)

        # Step 4: Halo exchange on PRIMITIVES (JIT function with communication)
        # This exchanges W (primitives) instead of U (conservatives)
        # Can potentially overlap with dt computation above!
        W_pad = apply_halo_exchange_prim(W)

        # Step 5: Flux computation and update from primitives (JIT function)
        # Uses W_pad directly, avoiding the U_pad→W_pad conversion!
        U_new = compute_flux_and_update_from_prim(U, W_pad, dt)

        return U_new, dt

    return time_step_muscl_hancock_modular
