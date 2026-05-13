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
import os
import jax
import jax.numpy as jnp


def create_muscl_hancock_modular_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                                       dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim, CFL,
                                       sharding=None):
    """
    Create MUSCL-Hancock stepper with MODULAR JIT compilation.

    Returns a regular Python function that orchestrates multiple JIT-compiled operations.
    This allows JAX to asynchronously dispatch and potentially overlap operations.
    """
    from hydro.conversions import cons_to_prim

    # Separate JIT function for dt computation
    @jax.jit
    def compute_dt(W):
        """Compute timestep from primitives - separate JIT function."""
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
        elif spatial_dim == 3:
            vx = W[1]
            vy = W[2]
            vz = W[3]
            P = W[4]
            cs = jnp.sqrt(gamma * P / rho)
            max_speed_x = jnp.abs(vx) + cs
            max_speed_y = jnp.abs(vy) + cs
            max_speed_z = jnp.abs(vz) + cs
            dt_local = CFL * jnp.minimum(
                jnp.minimum(dx / max_speed_x, dy / max_speed_y),
                dz / max_speed_z
            )
        else:
            raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")

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
            elif spatial_dim == 3:
                W_L_x = W_pad[:, :-1, 1:-1, 1:-1]
                W_R_x = W_pad[:, 1:, 1:-1, 1:-1]
                Fx = hllc_flux_primitive(W_L_x, W_R_x, axis=0, gamma=gamma)
                dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

                W_L_y = W_pad[:, 1:-1, :-1, 1:-1]
                W_R_y = W_pad[:, 1:-1, 1:, 1:-1]
                Fy = hllc_flux_primitive(W_L_y, W_R_y, axis=1, gamma=gamma)
                dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

                W_L_z = W_pad[:, 1:-1, 1:-1, :-1]
                W_R_z = W_pad[:, 1:-1, 1:-1, 1:]
                Fz = hllc_flux_primitive(W_L_z, W_R_z, axis=2, gamma=gamma)
                dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz

                return -(dFdx + dFdy + dFdz)
            else:
                raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")

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

            else:  # 3D
                from hydro.jacobian import apply_primitive_jacobian_3d

                Wc = W_pad[:, 1:-1, 1:-1, 1:-1]

                # MUSCL-Hancock predictor step
                dWx_scaled = dWx * inv_dx
                dWy_scaled = dWy * inv_dy
                dWz_scaled = dWz * inv_dz
                A_dW = apply_primitive_jacobian_3d(Wc, dWx_scaled, dWy_scaled, dWz_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW

                half_dWx = 0.5 * dWx
                half_dWy = 0.5 * dWy
                half_dWz = 0.5 * dWz

                # X-direction reconstruction
                WL_x = jnp.concatenate([W_pad[:, 0:1, 1:-1, 1:-1], W_half + half_dWx], axis=1)
                WR_x = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:, 1:-1, 1:-1]], axis=1)
                Fx = hllc_flux_primitive(WL_x, WR_x, axis=0, gamma=gamma)
                dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

                # Y-direction reconstruction
                WL_y = jnp.concatenate([W_pad[:, 1:-1, 0:1, 1:-1], W_half + half_dWy], axis=2)
                WR_y = jnp.concatenate([W_half - half_dWy, W_pad[:, 1:-1, -1:, 1:-1]], axis=2)
                Fy = hllc_flux_primitive(WL_y, WR_y, axis=1, gamma=gamma)
                dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

                # Z-direction reconstruction
                WL_z = jnp.concatenate([W_pad[:, 1:-1, 1:-1, 0:1], W_half + half_dWz], axis=3)
                WR_z = jnp.concatenate([W_half - half_dWz, W_pad[:, 1:-1, 1:-1, -1:]], axis=3)
                Fz = hllc_flux_primitive(WL_z, WR_z, axis=2, gamma=gamma)
                dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz

                return -(dFdx + dFdy + dFdz)

        # Select based on order (static Python branching — order is a closure constant)
        if order == 0:
            dU = first_order_flux(W_pad)
        else:
            dU = second_order_flux(W_pad)

        U_new = U + dt * dU

        # Enforce output sharding matches input to prevent sharding drift
        if sharding is not None:
            U_new = jax.lax.with_sharding_constraint(U_new, sharding)

        return U_new

    # Debug flag: set to True to trace which JIT call hangs (multi-GPU debugging)
    _debug_sync = bool(os.environ.get('LUNA_DEBUG_SYNC', ''))

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
        import sys

        if _debug_sync and jax.process_index() == 0:
            print(f"[DEBUG] Starting time step, U.sharding={U.sharding}", flush=True, file=sys.stderr)

        # Step 1: Convert to primitives (JIT function)
        W = convert_to_prim(U)

        if _debug_sync:
            jax.block_until_ready(W)
            if jax.process_index() == 0:
                print(f"[DEBUG] ✓ convert_to_prim done, W.sharding={W.sharding}", flush=True, file=sys.stderr)

        # Step 2: Compute dt from primitives (JIT function with global reduction)
        dt = compute_dt(W)

        if _debug_sync:
            jax.block_until_ready(dt)
            if jax.process_index() == 0:
                print(f"[DEBUG] ✓ compute_dt done, dt={dt}", flush=True, file=sys.stderr)

        # Step 3: Adjust dt if needed (JIT function, scalar operation)
        dt = adjust_dt(dt, t, t_end)

        if _debug_sync:
            jax.block_until_ready(dt)
            if jax.process_index() == 0:
                print(f"[DEBUG] ✓ adjust_dt done, dt={dt}", flush=True, file=sys.stderr)

        # Step 4: Halo exchange on PRIMITIVES (JIT function with communication)
        W_pad = apply_halo_exchange_prim(W)

        if _debug_sync:
            jax.block_until_ready(W_pad)
            if jax.process_index() == 0:
                print(f"[DEBUG] ✓ halo exchange done, W_pad.sharding={W_pad.sharding}", flush=True, file=sys.stderr)

        # Step 5: Flux computation and update from primitives (JIT function)
        U_new = compute_flux_and_update_from_prim(U, W_pad, dt)

        if _debug_sync:
            jax.block_until_ready(U_new)
            if jax.process_index() == 0:
                print(f"[DEBUG] ✓ flux+update done, U_new.sharding={U_new.sharding}", flush=True, file=sys.stderr)

        # Force correct sharding to prevent drift between time steps
        if sharding is not None:
            U_new = jax.device_put(U_new, sharding)

            if _debug_sync:
                jax.block_until_ready(U_new)
                if jax.process_index() == 0:
                    print(f"[DEBUG] ✓ resharding done, U_new.sharding={U_new.sharding}", flush=True, file=sys.stderr)

        return U_new, dt

    return time_step_muscl_hancock_modular
