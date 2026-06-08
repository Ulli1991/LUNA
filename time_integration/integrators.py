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
            vx = W[1]
            P = W[2]
            cs = jnp.sqrt(gamma * P / rho)
            dt_local = CFL * dx / (jnp.abs(vx) + cs)
        elif spatial_dim == 2:
            vx = W[1]
            vy = W[2]
            P = W[3]
            cs = jnp.sqrt(gamma * P / rho)
            dt_local = CFL * jnp.minimum(dx / (jnp.abs(vx) + cs), dy / (jnp.abs(vy) + cs))
        elif spatial_dim == 3:
            vx = W[1]
            vy = W[2]
            vz = W[3]
            P = W[4]
            cs = jnp.sqrt(gamma * P / rho)
            dt_local = CFL * jnp.minimum(
                jnp.minimum(dx / (jnp.abs(vx) + cs), dy / (jnp.abs(vy) + cs)),
                dz / (jnp.abs(vz) + cs),
            )
        else:
            raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")
        return jnp.min(dt_local)

    return compute_dt_from_U


# ---------------------------------------------------------------------------
# MUSCL-Hancock
# ---------------------------------------------------------------------------
def create_muscl_hancock_modular_step(
    pad_halo_fn,
    gamma,
    order,
    limiter,
    dx,
    dy,
    dz,
    inv_dx,
    inv_dy,
    inv_dz,
    boundary,
    spatial_dim,
    CFL,
    sharding=None,
    riemann_solver="hllc",
    use_shardmap=False,
    mesh=None,
):
    """MUSCL-Hancock stepper with MODULAR JIT compilation.

    Returns (compute_dt_fn, step_fn) where
        compute_dt_fn(U) -> dt_cfl  (pure CFL, no t_end or snapshot clamping)
        step_fn(U, dt)   -> U_new   (pure step using caller-supplied dt)

    When `use_shardmap=True` the step body is wrapped in
    `jax.experimental.shard_map.shard_map` so each device runs the full
    pipeline (cons_to_prim → halo → MUSCL → update) on its own shard,
    with halos exchanged via explicit `jax.lax.ppermute` inside
    `pad_halo_fn`. This sidesteps XLA's auto-sharding scratch blow-up
    at small device counts. Requires `mesh` to be supplied so the
    shard_map can construct PartitionSpecs.

    The runner is responsible for all time-target arithmetic. Physics
    (predictor + reconstruct + flux divergence) lives in hydro/muscl_hancock.py.
    """
    compute_dt_fn = _make_compute_dt_fn(gamma, CFL, spatial_dim, dx, dy, dz)

    from hydro.muscl_hancock import muscl_hancock_divergence

    # Inner step (semantics-only — same body whether dispatched per-shard
    # via shard_map or globally via auto-sharding).
    def _step_inner(U, dt):
        W = cons_to_prim(U, gamma)
        W_pad = pad_halo_fn(W, boundary, spatial_dim)
        dU = muscl_hancock_divergence(
            W_pad,
            dt,
            gamma=gamma,
            limiter=limiter,
            riemann_solver=riemann_solver,
            spatial_dim=spatial_dim,
            inv_dx=inv_dx,
            inv_dy=inv_dy,
            inv_dz=inv_dz,
            boundary=boundary,
            order=order,
        )
        U_new = U + dt * dU
        if sharding is not None and not use_shardmap:
            U_new = jax.lax.with_sharding_constraint(U_new, sharding)
        return U_new

    if use_shardmap:
        if mesh is None:
            raise ValueError("use_shardmap=True requires `mesh` to be passed in")
        from jax.experimental.shard_map import shard_map
        from jax.sharding import PartitionSpec as P

        if spatial_dim == 3:
            in_spec = P(None, "x", "y", "z")
        elif spatial_dim == 2:
            in_spec = P(None, "x", "y")
        else:
            in_spec = P(None, "x")
        time_step_muscl_hancock_modular = jax.jit(
            shard_map(
                _step_inner,
                mesh=mesh,
                in_specs=(in_spec, P()),
                out_specs=in_spec,
            )
        )
    else:
        time_step_muscl_hancock_modular = jax.jit(_step_inner)

    return compute_dt_fn, time_step_muscl_hancock_modular
