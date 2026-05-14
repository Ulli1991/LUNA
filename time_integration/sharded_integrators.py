"""
Time integration schemes for sharded arrays with automatic sharding.

These factory functions create time stepping functions that work with
sharded arrays distributed across multiple devices, similar to the pmap
version in integrators.py but for automatic sharding.

All sharded steppers share the runner-facing signature
    (U, t, t_snapshot, t_end) -> (U_new, dt)
so they can be swapped in setup/distributed_setup.py without runner changes.
"""
import jax

from hydro.conversions import cons_to_prim


def _compute_dt(U, gamma, CFL, spatial_dim, dx, dy, dz, t, t_end):
    """CFL-limited dt, clamped to land exactly on t_end if close enough.

    Same logic as the MUSCL-Hancock path, factored out so RK2/RK4 can reuse it.
    """
    W = cons_to_prim(U, gamma)
    rho = W[0]
    if spatial_dim == 1:
        vx = W[1]
        P = W[2]
        cs = jax.numpy.sqrt(gamma * P / rho)
        dt_local = CFL * dx / (jax.numpy.abs(vx) + cs)
    elif spatial_dim == 2:
        vx, vy, P = W[1], W[2], W[3]
        cs = jax.numpy.sqrt(gamma * P / rho)
        dt_local = CFL * jax.numpy.minimum(
            dx / (jax.numpy.abs(vx) + cs),
            dy / (jax.numpy.abs(vy) + cs),
        )
    elif spatial_dim == 3:
        vx, vy, vz, P = W[1], W[2], W[3], W[4]
        cs = jax.numpy.sqrt(gamma * P / rho)
        dt_local = CFL * jax.numpy.minimum(
            jax.numpy.minimum(dx / (jax.numpy.abs(vx) + cs),
                              dy / (jax.numpy.abs(vy) + cs)),
            dz / (jax.numpy.abs(vz) + cs),
        )
    else:
        raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")

    dt = jax.numpy.min(dt_local)
    dt = jax.numpy.where(
        (t_end - t < 10 * dt) & (t + dt > t_end),
        t_end - t,
        dt,
    )
    return dt


def create_rk2_sharded_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                             dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim, CFL):
    """Create an RK2 time stepper for sharded arrays.

    Signature: (U, t, t_snapshot, t_end) -> (U_new, dt). t_snapshot is accepted
    for runner-API compatibility but unused (dt is bounded by t_end only).
    """
    @jax.jit(donate_argnums=(0,))
    def time_step_rk2(U, t, t_snapshot, t_end):
        dt = _compute_dt(U, gamma, CFL, spatial_dim, dx, dy, dz, t, t_end)

        U_pad = pad_halo_fn(U, boundary, spatial_dim)
        dU1 = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)
        U1 = U + dt * dU1

        U1_pad = pad_halo_fn(U1, boundary, spatial_dim)
        dU2 = flux_div_fn(U1_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        return U + 0.5 * dt * (dU1 + dU2), dt

    return time_step_rk2


def create_muscl_hancock_sharded_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                                       dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim, CFL):
    """
    Create a MUSCL-Hancock time stepper for sharded arrays with automatic sharding.

    NOW COMBINES dt computation and time step into ONE function to reduce overhead.

    MUSCL-Hancock is a predictor-corrector method where the predictor step
    is built into the flux function itself (using dt for half-timestep evolution).

    Parameters
    ----------
    pad_halo_fn : callable
        Halo exchange function with signature (U, boundary, spatial_dim) -> U_padded
    flux_div_fn : callable
        Flux divergence function (flux_divergence_from_padded) that accepts dt parameter
    gamma : float
        Adiabatic index
    order : int
        Spatial order (0 or 1)
    limiter : int
        Limiter type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing
    boundary : int
        Boundary condition type (captured in closure)
    spatial_dim : int
        Spatial dimension (captured in closure)
    CFL : float
        CFL number for dt computation

    Returns
    -------
    time_step_fn : callable
        MUSCL-Hancock time stepping function with signature (U, t, t_snapshot) -> (U_new, dt)
    """
    @jax.jit(donate_argnums=(0,))
    def time_step_muscl_hancock(U, t, t_snapshot, t_end):
        """MUSCL-Hancock step with dt computation - ONE function call per timestep."""
        dt = _compute_dt(U, gamma, CFL, spatial_dim, dx, dy, dz, t, t_end)

        # Pad with halo exchange
        U_pad = pad_halo_fn(U, boundary, spatial_dim)

        # Compute flux divergence with predictor step (dt is passed!)
        dU = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz, dt)

        # Update state
        return U + dt * dU, dt

    return time_step_muscl_hancock


def create_rk4_sharded_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                             dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim, CFL):
    """Create an RK4 time stepper for sharded arrays.

    Signature: (U, t, t_snapshot, t_end) -> (U_new, dt). t_snapshot is accepted
    for runner-API compatibility but unused (dt is bounded by t_end only).
    """
    @jax.jit(donate_argnums=(0,))
    def time_step_rk4(U, t, t_snapshot, t_end):
        dt = _compute_dt(U, gamma, CFL, spatial_dim, dx, dy, dz, t, t_end)

        U_pad = pad_halo_fn(U, boundary, spatial_dim)
        k1 = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        U1 = U + 0.5 * dt * k1
        U1_pad = pad_halo_fn(U1, boundary, spatial_dim)
        k2 = flux_div_fn(U1_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        U2 = U + 0.5 * dt * k2
        U2_pad = pad_halo_fn(U2, boundary, spatial_dim)
        k3 = flux_div_fn(U2_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        U3 = U + dt * k3
        U3_pad = pad_halo_fn(U3, boundary, spatial_dim)
        k4 = flux_div_fn(U3_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        return U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), dt

    return time_step_rk4


