"""
MUSCL-Hancock time-step factory for LUNA (sole supported integrator).

Pattern:
- The user-facing step function is NOT @jax.jit decorated.
- Individual operations are small @jax.jit functions called in sequence.
- Halo exchanges live inside the JITs so JAX can overlap their collectives
  with the surrounding compute.

The primitive halo exchange (apply_halo_exchange_prim) saves one cons→prim
on the padded array per step versus the conservative halo path.
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



# ---------------------------------------------------------------------------
# MUSCL-Hancock
# ---------------------------------------------------------------------------
def create_muscl_hancock_modular_step(pad_halo_fn, gamma, order, limiter,
                                       dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim, CFL,
                                       sharding=None, riemann_solver="hllc"):
    """MUSCL-Hancock stepper with MODULAR JIT compilation.

    Returns (compute_dt_fn, step_fn) where
        compute_dt_fn(U) -> dt_cfl  (pure CFL, no t_end or snapshot clamping)
        step_fn(U, dt)   -> U_new   (pure step using caller-supplied dt)

    The runner is responsible for all time-target arithmetic. Physics
    (predictor + reconstruct + flux divergence) lives in hydro/muscl_hancock.py.
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

    # Pure-physics MUSCL-Hancock + first-order flux divergence both live in
    # hydro/muscl_hancock.py. This factory just JIT-wraps the call.
    from hydro.muscl_hancock import muscl_hancock_divergence

    @jax.jit
    def compute_flux_and_update_from_prim(U, W_pad, dt):
        """Apply the MUSCL-Hancock (or first-order) step to U using W_pad."""
        dU = muscl_hancock_divergence(
            W_pad, dt,
            gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
            spatial_dim=spatial_dim,
            inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
            boundary=boundary, order=order,
        )
        U_new = U + dt * dU
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
