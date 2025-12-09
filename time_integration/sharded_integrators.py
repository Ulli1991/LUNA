"""
Time integration schemes for sharded arrays using shard_map.

These factory functions create time stepping functions that work with
sharded arrays distributed across multiple devices.
"""
import jax


def create_rk2_stepper(compute_flux_sharded):
    """
    Create RK2 time stepping function for sharded arrays.

    Parameters
    ----------
    compute_flux_sharded : callable
        Sharded flux function with signature (U, dt) -> dU/dt

    Returns
    -------
    time_step_rk2 : callable
        RK2 time stepping function with signature (U, dt) -> U_new
    """
    @jax.jit
    def time_step_rk2(U, dt):
        """RK2 time step with sharded arrays."""
        # Stage 1
        k1 = compute_flux_sharded(U, dt)
        U_temp = U + 0.5 * dt * k1

        # Stage 2
        k2 = compute_flux_sharded(U_temp, dt)
        U_new = U + dt * k2

        return U_new

    return time_step_rk2


def create_muscl_hancock_stepper(compute_flux_sharded):
    """
    Create MUSCL-Hancock time stepping function for sharded arrays.

    The predictor step is built into the flux function when dt is provided.

    Parameters
    ----------
    compute_flux_sharded : callable
        Sharded flux function with signature (U, dt) -> dU/dt
        Must include predictor step when dt is provided

    Returns
    -------
    time_step_muscl_hancock : callable
        MUSCL-Hancock time stepping function with signature (U, dt) -> U_new
    """
    @jax.jit
    def time_step_muscl_hancock(U, dt):
        """
        MUSCL-Hancock predictor-corrector time step with sharded arrays.

        The predictor step is built into the flux function when dt is provided.
        """
        dUdt = compute_flux_sharded(U, dt)
        U_new = U + dt * dUdt
        return U_new

    return time_step_muscl_hancock


def create_rk4_stepper(compute_flux_sharded):
    """
    Create RK4 time stepping function for sharded arrays.

    Parameters
    ----------
    compute_flux_sharded : callable
        Sharded flux function with signature (U, dt) -> dU/dt

    Returns
    -------
    time_step_rk4 : callable
        RK4 time stepping function with signature (U, dt) -> U_new
    """
    @jax.jit
    def time_step_rk4(U, dt):
        """RK4 time step with sharded arrays."""
        # Stage 1
        k1 = compute_flux_sharded(U, dt)

        # Stage 2
        U1 = U + 0.5 * dt * k1
        k2 = compute_flux_sharded(U1, dt)

        # Stage 3
        U2 = U + 0.5 * dt * k2
        k3 = compute_flux_sharded(U2, dt)

        # Stage 4
        U3 = U + dt * k3
        k4 = compute_flux_sharded(U3, dt)

        # Combine stages
        U_new = U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return U_new

    return time_step_rk4


def select_time_stepper(time_solver, compute_flux_sharded):
    """
    Select and create the appropriate time stepping function.

    Parameters
    ----------
    time_solver : int
        Time solver index:
        0 = RK2
        1 = MUSCL-Hancock (Predictor-Corrector)
        2 = RK4
    compute_flux_sharded : callable
        Sharded flux function with signature (U, dt) -> dU/dt

    Returns
    -------
    time_step_fn : callable
        Selected time stepping function with signature (U, dt) -> U_new
    name : str
        Name of the selected time integrator
    """
    if time_solver == 0:
        time_step_fn = create_rk2_stepper(compute_flux_sharded)
        name = "RK2"
    elif time_solver == 1:
        time_step_fn = create_muscl_hancock_stepper(compute_flux_sharded)
        name = "MUSCL-Hancock (Predictor-Corrector)"
    elif time_solver == 2:
        time_step_fn = create_rk4_stepper(compute_flux_sharded)
        name = "RK4"
    else:
        raise ValueError(
            f"Invalid TIME_SOLVER={time_solver}. "
            "Must be 0 (RK2), 1 (MUSCL-Hancock), or 2 (RK4)."
        )

    return time_step_fn, name
