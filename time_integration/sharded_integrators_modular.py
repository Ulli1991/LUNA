"""
Modular time integration for LUNA (the only integrator module in the code).

Pattern:
- The user-facing step function is NOT @jax.jit decorated.
- Individual operations are small @jax.jit functions called in sequence.
- Halo exchanges live inside the JITs so JAX can overlap their collectives
  with the surrounding compute.

This is the path that runs across multi-GPU multi-node setups.

Available integrators (all share the same Python-orchestration pattern):
- RK2            via create_rk2_modular_step           (TIME_SOLVER=0)
- MUSCL-Hancock  via create_muscl_hancock_modular_step (TIME_SOLVER=1)
- RK4            via create_rk4_modular_step           (TIME_SOLVER=2)

MUSCL-Hancock additionally exchanges PRIMITIVES rather than conservatives,
which saves one cons→prim on the padded array per step. RK2/RK4 do
multiple flux evaluations per step and reuse the standard conservative
halo exchange (flux_divergence_from_padded does cons→prim internally).
"""
import os
import jax
import jax.numpy as jnp

from hydro.conversions import cons_to_prim


# ---------------------------------------------------------------------------
# Shared helpers (used by all integrators below)
# ---------------------------------------------------------------------------
def _make_compute_dt_fn(gamma, CFL, spatial_dim, dx, dy, dz):
    """Factory: returns jit'd compute_dt(U) -> dt_cfl using primitives."""
    @jax.jit
    def compute_dt_from_U(U):
        W = cons_to_prim(U, gamma)
        rho = W[0]
        if spatial_dim == 1:
            vx = W[1]; P = W[2]
            cs = jnp.sqrt(gamma * P / rho)
            dt_local = CFL * dx / (jnp.abs(vx) + cs)
        elif spatial_dim == 2:
            vx = W[1]; vy = W[2]; P = W[3]
            cs = jnp.sqrt(gamma * P / rho)
            dt_local = CFL * jnp.minimum(dx / (jnp.abs(vx) + cs),
                                          dy / (jnp.abs(vy) + cs))
        elif spatial_dim == 3:
            vx = W[1]; vy = W[2]; vz = W[3]; P = W[4]
            cs = jnp.sqrt(gamma * P / rho)
            dt_local = CFL * jnp.minimum(
                jnp.minimum(dx / (jnp.abs(vx) + cs),
                            dy / (jnp.abs(vy) + cs)),
                dz / (jnp.abs(vz) + cs),
            )
        else:
            raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")
        return jnp.min(dt_local)
    return compute_dt_from_U


def _make_flux_stage(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                     dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim):
    """Factory: returns jit'd stage(U) -> dU = -∇·F(U).

    One JIT containing both the halo exchange (with collectives) and the
    flux divergence — the unit of work that gets scheduled by the Python
    orchestrator in RK2/RK4.
    """
    @jax.jit
    def stage(U):
        U_pad = pad_halo_fn(U, boundary, spatial_dim)
        return flux_div_fn(U_pad, gamma, order, limiter,
                           dx, dy, dz, inv_dx, inv_dy, inv_dz)
    return stage


# ---------------------------------------------------------------------------
# RK2
# ---------------------------------------------------------------------------
def create_rk2_modular_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                             dx, dy, dz, inv_dx, inv_dy, inv_dz,
                             boundary, spatial_dim, CFL, sharding=None):
    """RK2 stepper following the modular pattern.

    Returns (compute_dt_fn, step_fn) where
        compute_dt_fn(U) -> dt_cfl   (pure CFL, runner does the clamping)
        step_fn(U, dt)   -> U_new
    """
    compute_dt_fn = _make_compute_dt_fn(gamma, CFL, spatial_dim, dx, dy, dz)
    stage = _make_flux_stage(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                              dx, dy, dz, inv_dx, inv_dy, inv_dz,
                              boundary, spatial_dim)

    def rk2_step(U, dt):
        dU1 = stage(U)
        U1 = U + dt * dU1
        dU2 = stage(U1)
        U_new = U + 0.5 * dt * (dU1 + dU2)
        if sharding is not None:
            U_new = jax.device_put(U_new, sharding)
        return U_new

    return compute_dt_fn, rk2_step


# ---------------------------------------------------------------------------
# RK4
# ---------------------------------------------------------------------------
def create_rk4_modular_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                             dx, dy, dz, inv_dx, inv_dy, inv_dz,
                             boundary, spatial_dim, CFL, sharding=None):
    """RK4 stepper following the modular pattern."""
    compute_dt_fn = _make_compute_dt_fn(gamma, CFL, spatial_dim, dx, dy, dz)
    stage = _make_flux_stage(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                              dx, dy, dz, inv_dx, inv_dy, inv_dz,
                              boundary, spatial_dim)

    def rk4_step(U, dt):
        k1 = stage(U)
        k2 = stage(U + 0.5 * dt * k1)
        k3 = stage(U + 0.5 * dt * k2)
        k4 = stage(U + dt * k3)
        U_new = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if sharding is not None:
            U_new = jax.device_put(U_new, sharding)
        return U_new

    return compute_dt_fn, rk4_step


# ---------------------------------------------------------------------------
# MUSCL-Hancock
# ---------------------------------------------------------------------------
def create_muscl_hancock_modular_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                                       dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim, CFL,
                                       sharding=None):
    """MUSCL-Hancock stepper with MODULAR JIT compilation.

    Returns (compute_dt_fn, step_fn) where
        compute_dt_fn(U) -> dt_cfl  (pure CFL, no t_end or snapshot clamping)
        step_fn(U, dt)   -> U_new   (pure step using caller-supplied dt)

    The runner is responsible for all time-target arithmetic.

    Note: flux_div_fn is accepted for signature symmetry with RK2/RK4 but
    is not used — MUSCL-Hancock inlines its own flux+predictor below to
    fuse the cons→prim, slope, predictor, reconstruct, and HLLC stages.
    """
    compute_dt_fn = _make_compute_dt_fn(gamma, CFL, spatial_dim, dx, dy, dz)

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
    def time_step_muscl_hancock_modular(U, dt):
        """
        MUSCL-Hancock time step with MODULAR JIT compilation. Pure: caller
        supplies dt.

        FLOW:
        1. U → W (convert once)
        2. Pad W → W_pad (exchange primitives, not conservatives)
        3. Compute fluxes directly from W_pad
        4. Update U
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

        # Step 2: Halo exchange on PRIMITIVES (JIT function with communication)
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

        return U_new

    return compute_dt_fn, time_step_muscl_hancock_modular
