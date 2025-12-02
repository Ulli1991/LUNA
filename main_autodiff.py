"""
Improved autodiff testing with multiple observables and gradient verification.

This version:
1. Provides multiple observable options (not just total energy)
2. Verifies gradients via finite difference comparison
3. Includes test problem recommendations
4. Better diagnostics for gradient quality
"""
# Enable 64-bit precision in JAX for better numerical stability
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from functools import partial
from config import Config
from ic_generation.ic import initial_conditions
from time_integration.integrators import rk2_step, rk4_step, muscl_hancock_step
from hydro.divergence import flux_divergence
from time_integration.cfl import compute_dt
from file_io.write_autodiff_result import write_autodiff_result
from utils.logo import print_KRATOS_banner
import time

# -------------------------
# Autodiff-ready simulation
# -------------------------
def run_simulation(U0, cfg: Config, max_steps: int, dt_schedule=None):
    """Run simulation from initial conditions to final time.

    Parameters
    ----------
    U0 : array
        Initial conditions
    cfg : Config
        Configuration object
    max_steps : int
        Maximum number of steps
    dt_schedule : array or None
        Optional pre-computed timestep schedule. If provided, use these timesteps instead
        of computing them dynamically. This ensures gradient verification works correctly
        by having autodiff and finite difference use identical timesteps.
    """
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy if dy else None
    inv_dz = 1.0 / dz if dz else None

    flux_fn = partial(
        flux_divergence,
        gamma=cfg.gamma,
        order=cfg.ORDER,
        limiter=cfg.LIMITER,
        boundary=cfg.BOUNDARY,
        dx=dx, dy=dy, dz=dz,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
        use_smooth_limiters=cfg.use_smooth_limiters,
        sharpness=cfg.smooth_limiter_sharpness
    )

    time_solver_map = {
        0: rk2_step,
        1: muscl_hancock_step,
        2: rk4_step
    }
    time_step_fn = partial(time_solver_map[cfg.TIME_SOLVER], flux_fn=flux_fn)

    # TIMESTEP STRATEGY:
    # If dt_schedule is provided, use it (frozen timesteps for gradient verification)
    # Otherwise, compute dt dynamically (to generate the schedule)
    def step_fn(carry, step_idx):
        t, U = carry

        if dt_schedule is not None:
            # Use pre-computed timestep (no gradient flow through dt)
            dt = dt_schedule[step_idx]
        else:
            # Compute dt dynamically based on current state
            dt = compute_dt(U, cfg.gamma, cfg.CFL, t, cfg.t_end, dx, dy, dz)

        # Take a timestep
        U_next = time_step_fn(U, dt)
        t_next = t + dt

        return (t_next, U_next), dt  # Return dt to track steps

    (t_final, U_final), dt_history = jax.lax.scan(step_fn, (cfg.t_start, U0), jnp.arange(max_steps))

    # Debug: count actual steps taken
    actual_steps = jnp.sum(dt_history > 1e-14)
    print(f"DEBUG: Actual steps taken = {actual_steps} out of {max_steps} max_steps")
    print(f"DEBUG: Final time = {t_final}, target = {cfg.t_end}")

    return U_final, dt_history

# -------------------------
# Observable functions
# -------------------------

def observable_total_energy(U_final, spatial_dim):
    """
    Sum of total energy (E).

    WARNING: For conservative schemes, this should be nearly constant!
    May give near-zero or uninformative gradients.
    """
    return jnp.sum(U_final[-1])

def observable_kinetic_energy(U_final, spatial_dim):
    """
    Sum of kinetic energy: KE = 0.5 * (mom^2) / rho

    Better than total energy - changes as flow evolves.
    """
    rho = U_final[0]
    if spatial_dim == 1:
        mom_x = U_final[1]
        ke = 0.5 * mom_x**2 / rho
    elif spatial_dim == 2:
        mom_x = U_final[1]
        mom_y = U_final[2]
        ke = 0.5 * (mom_x**2 + mom_y**2) / rho
    else:  # 3D
        mom_x = U_final[1]
        mom_y = U_final[2]
        mom_z = U_final[3]
        ke = 0.5 * (mom_x**2 + mom_y**2 + mom_z**2) / rho

    return jnp.sum(ke)

def observable_momentum_region(U_final, spatial_dim):
    """
    Sum of x-momentum in the central region [1/4, 3/4] of domain.

    Good for shock problems - captures shock passage through region.
    """
    Nx = U_final.shape[1]
    start = Nx // 4
    end = 3 * Nx // 4

    if spatial_dim == 1:
        return jnp.sum(U_final[1, start:end])
    elif spatial_dim == 2:
        Ny = U_final.shape[2]
        return jnp.sum(U_final[1, start:end, :])
    else:  # 3D
        return jnp.sum(U_final[1, start:end, :, :])

def observable_density_at_point(U_final, spatial_dim):
    """
    Density at center of domain.

    Very sensitive to shock position for shock problems.
    For 1D: x = Nx/2
    For 2D: (Nx/2, Ny/2)
    For 3D: (Nx/2, Ny/2, Nz/2)
    """
    Nx = U_final.shape[1]

    if spatial_dim == 1:
        return U_final[0, Nx // 2]
    elif spatial_dim == 2:
        Ny = U_final.shape[2]
        return U_final[0, Nx // 2, Ny // 2]
    else:  # 3D
        Ny = U_final.shape[2]
        Nz = U_final.shape[3]
        return U_final[0, Nx // 2, Ny // 2, Nz // 2]

def observable_pressure_integral(U_final, spatial_dim, gamma=1.4):
    """
    Integral of pressure over domain.

    Pressure = (gamma - 1) * (E - 0.5*rho*v^2)
    Changes significantly in shock problems.
    """
    rho = U_final[0]
    E = U_final[-1]

    if spatial_dim == 1:
        mom_x = U_final[1]
        ke = 0.5 * mom_x**2 / rho
    elif spatial_dim == 2:
        mom_x = U_final[1]
        mom_y = U_final[2]
        ke = 0.5 * (mom_x**2 + mom_y**2) / rho
    else:  # 3D
        mom_x = U_final[1]
        mom_y = U_final[2]
        mom_z = U_final[3]
        ke = 0.5 * (mom_x**2 + mom_y**2 + mom_z**2) / rho

    pressure = (gamma - 1.0) * (E - ke)
    return jnp.sum(pressure)

# Dictionary of available observables
OBSERVABLES = {
    'total_energy': observable_total_energy,
    'kinetic_energy': observable_kinetic_energy,
    'momentum_region': observable_momentum_region,
    'density_at_point': observable_density_at_point,
    'pressure_integral': observable_pressure_integral,
}

# -------------------------
# Finite difference gradient checker
# -------------------------

def finite_difference_gradient(sim_fn, observable_fn, U0, cfg, max_steps, dt_schedule, epsilon=1e-5):
    """
    Compute gradient via finite differences for verification.

    Uses central differences: (f(x+eps) - f(x-eps)) / (2*eps)

    WARNING: This is SLOW - computes for multiple spatial points across all variables!

    Parameters
    ----------
    sim_fn : callable
        Simulation function (should NOT have dt_schedule baked in)
    observable_fn : callable
        Observable function
    U0 : array
        Initial conditions
    cfg : Config
        Configuration
    max_steps : int
        Maximum steps
    dt_schedule : array
        Timestep schedule to use (same as autodiff)
    epsilon : float
        Finite difference step size

    Returns
    -------
    fd_grads : array
        Finite difference gradients, shape (Nvar, num_samples)
    sample_indices : array
        Spatial indices where FD was computed
    """
    print(f"Computing finite difference gradients (epsilon={epsilon})...")

    # Compute FD for ALL spatial points across all variables
    Nvar = U0.shape[0]
    Nx = U0.shape[1]
    sample_indices = jnp.arange(Nx)  # ALL points

    fd_grads = jnp.zeros((Nvar, Nx))

    total_evals = Nvar * Nx
    print(f"Computing {total_evals} finite difference evaluations ({Nvar} vars × {Nx} points)...")
    print("This will take a while...")

    eval_count = 0
    for var_idx in range(Nvar):
        print(f"  Variable {var_idx+1}/{Nvar}...")
        for spatial_idx in range(Nx):
            # Perturb one component
            U_plus = U0.at[var_idx, spatial_idx].add(epsilon)
            U_minus = U0.at[var_idx, spatial_idx].add(-epsilon)

            # Compute observables (extract U_final from tuple)
            # IMPORTANT: Use the SAME dt_schedule for both perturbations
            U_final_plus, _ = sim_fn(U_plus, cfg, max_steps, dt_schedule=dt_schedule)
            U_final_minus, _ = sim_fn(U_minus, cfg, max_steps, dt_schedule=dt_schedule)

            obs_plus = observable_fn(U_final_plus, cfg.SPATIAL_DIMENSION)
            obs_minus = observable_fn(U_final_minus, cfg.SPATIAL_DIMENSION)

            # Central difference
            fd_grad = (obs_plus - obs_minus) / (2.0 * epsilon)
            fd_grads = fd_grads.at[var_idx, spatial_idx].set(fd_grad)

            eval_count += 1
            if eval_count % 100 == 0:
                print(f"    Progress: {eval_count}/{total_evals} ({100*eval_count/total_evals:.1f}%)")

    return fd_grads, sample_indices

# -------------------------
# Main test function
# -------------------------

def main():
    # --- Print logo ---
    print_KRATOS_banner()

    cfg = Config()

    print("="*80)
    print("AUTODIFF DIFFERENTIABILITY TEST")
    print("="*80)
    print()

    # -------------------------
    # Configuration check
    # -------------------------
    print("Configuration:")
    print(f"  Test problem: {cfg.test_problem}")
    print(f"  Spatial dimension: {cfg.SPATIAL_DIMENSION}")
    print(f"  Grid size: Nx={cfg.Nx}, Ny={cfg.Ny}, Nz={cfg.Nz}")
    print(f"  Domain: Lx={cfg.Lx}, Ly={cfg.Ly}, Lz={cfg.Lz}")
    print(f"  Time range: {cfg.t_start:.3f} → {cfg.t_end:.3f}")
    print(f"  Order: {cfg.ORDER} (0=first-order, 1=PLM)")
    print(f"  Limiter: {cfg.LIMITER} (0=minmod, 1=moncen, 2=vanleer)")
    print(f"  Time solver: {cfg.TIME_SOLVER} (0=RK2, 1=Hancock, 2=RK4)")
    print(f"  Smooth limiters: {cfg.use_smooth_limiters}")
    if cfg.use_smooth_limiters:
        print(f"  Sharpness: {cfg.smooth_limiter_sharpness}")
    print()

    # -------------------------
    # Test problem recommendation
    # -------------------------
    print("-" * 80)
    print("TEST PROBLEM RECOMMENDATIONS:")
    print("-" * 80)
    print("""
For differentiability testing:

GOOD (activates limiters, non-trivial gradients):
  - Sod-Shock: Classic shock tube, moderate difficulty
  - Woodward-Colella: Strong shocks, challenging for smooth limiters
  - Shu-Osher: Shock-turbulence interaction

AVOID (too smooth, doesn't test limiters):
  - Soundwave: Linear, limiters barely activate
  - Soundwave-Advected: Same issue

RECOMMENDED OBSERVABLE BY PROBLEM:
  - Sod-Shock: 'density_at_point' or 'momentum_region'
  - Woodward-Colella: 'momentum_region' or 'pressure_integral'
  - Shu-Osher: 'kinetic_energy' or 'momentum_region'
""")
    print("-" * 80)
    print()

    # -------------------------
    # Choose observable
    # -------------------------
    # Auto-select based on test problem
    if "Soundwave" in cfg.test_problem or "Advected" in cfg.test_problem:
        print("WARNING: You're using a smooth test problem!")
        print("         This won't effectively test smooth limiters.")
        print("         Consider using Sod-Shock or Woodward-Colella instead.")
        print()
        observable_name = 'kinetic_energy'
    elif "Sod" in cfg.test_problem:
        observable_name = 'kinetic_energy'  # Whole domain contributes → better gradients
    elif "Woodward" in cfg.test_problem or "Colella" in cfg.test_problem:
        observable_name = 'momentum_region'
    else:
        # Default
        observable_name = 'momentum_region'

    print(f"Selected observable: '{observable_name}'")
    print()

    observable_fn = OBSERVABLES[observable_name]

    # -------------------------
    # Initialize
    # -------------------------
    U0 = initial_conditions(cfg)
    print(f"Initial condition shape: {U0.shape}")
    print()

    # -------------------------
    # Forward simulation
    # -------------------------
    print("="*80)
    print("1. FORWARD SIMULATION")
    print("="*80)

    # Estimate max number of steps needed (dt varies, so this is an upper bound)
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None
    dt_estimate = compute_dt(U0, cfg.gamma, cfg.CFL, cfg.t_start, cfg.t_end, dx, dy, dz)
    dt_conservative = dt_estimate * 0.8

    # Calculate upper bound for steps (dt will vary during simulation)
    max_steps = int(jnp.ceil((cfg.t_end - cfg.t_start) / dt_conservative))
    print(f"Variable timestep mode (like main.py differencing version)")
    print(f"Initial dt estimate: {dt_estimate:.6e}")
    print(f"Max steps (upper bound): {max_steps}")
    print()

    # STEP 1: Run once without schedule to get the timestep sequence
    print("Generating timestep schedule from initial conditions...")
    sim_fn_no_schedule = jax.jit(run_simulation, static_argnames=['cfg', 'max_steps'])
    _, dt_history_template = sim_fn_no_schedule(U0, cfg, max_steps)
    print(f"✓ Timestep schedule generated ({jnp.sum(dt_history_template > 0):.0f} active steps)")
    print()

    # STEP 2: Now use this schedule for all evaluations (autodiff and finite diff)
    # This ensures both methods use identical timesteps
    sim_fn = partial(jax.jit(run_simulation, static_argnames=['cfg', 'max_steps']),
                     dt_schedule=dt_history_template)

    # Warmup
    _ = sim_fn(U0, cfg, max_steps)

    t_start = time.time()
    U_final, dt_history = sim_fn(U0, cfg, max_steps)
    U_final.block_until_ready()
    t_forward = time.time() - t_start

    print(f"✓ Forward simulation completed in {t_forward:.3f}s")
    print(f"  Final state shape: {U_final.shape}")

    obs_value = observable_fn(U_final, cfg.SPATIAL_DIMENSION)
    print(f"  Observable value: {obs_value:.10e}")
    print()

    # -------------------------
    # Autodiff gradient
    # -------------------------
    print("="*80)
    print("2. AUTODIFF GRADIENT COMPUTATION")
    print("="*80)

    # Gradient function: extract U_final from (U_final, dt_history) tuple
    def sim_and_observe(U):
        U_final, _ = sim_fn(U, cfg, max_steps)
        return observable_fn(U_final, cfg.SPATIAL_DIMENSION)

    grad_fn = jax.jit(jax.grad(sim_and_observe))

    # Warmup
    _ = grad_fn(U0)

    t_start = time.time()
    grads_autodiff = grad_fn(U0)
    grads_autodiff.block_until_ready()
    t_grad = time.time() - t_start

    print(f"✓ Autodiff gradient computed in {t_grad:.3f}s")
    print(f"  Gradient shape: {grads_autodiff.shape}")
    print()

    print("Gradient statistics:")
    print(f"  Min:    {jnp.min(grads_autodiff):.6e}")
    print(f"  Max:    {jnp.max(grads_autodiff):.6e}")
    print(f"  Mean:   {jnp.mean(grads_autodiff):.6e}")
    print(f"  Std:    {jnp.std(grads_autodiff):.6e}")
    print(f"  L2norm: {jnp.linalg.norm(grads_autodiff):.6e}")
    print()

    # Check for NaNs or Infs
    n_nans = jnp.sum(jnp.isnan(grads_autodiff))
    n_infs = jnp.sum(jnp.isinf(grads_autodiff))

    if n_nans > 0 or n_infs > 0:
        print(f"⚠ WARNING: Gradient contains {n_nans} NaNs and {n_infs} Infs!")
        print("  This indicates a problem with differentiability.")
    else:
        print("✓ Gradient is finite (no NaNs or Infs)")
    print()

    # -------------------------
    # Finite difference verification
    # -------------------------
    print("="*80)
    print("3. FINITE DIFFERENCE VERIFICATION")
    print("="*80)
    print("Computing finite difference gradients for a subset of points...")
    print("(This is slow - only checking first variable at sampled locations)")
    print()

    # Remove JIT for finite difference (we need multiple evaluations)
    sim_fn_no_jit = run_simulation

    fd_grads, sample_indices = finite_difference_gradient(
        sim_fn_no_jit, observable_fn, U0, cfg, max_steps, dt_history_template, epsilon=1e-5
    )

    # Get corresponding autodiff gradients
    autodiff_samples = grads_autodiff[:, sample_indices]  # Shape: (Nvar, num_samples)

    print(f"✓ Finite difference gradients computed")
    print()

    # Print comparison for each variable
    var_names = ['rho', 'momx', 'E'] if cfg.SPATIAL_DIMENSION == 1 else ['rho', 'momx', 'momy', 'E']
    for var_idx in range(fd_grads.shape[0]):
        print(f"Comparison for variable {var_names[var_idx]}:")
        print(f"{'Index':<8} {'Autodiff':<15} {'FiniteDiff':<15} {'RelError':<12}")
        print("-" * 60)

        # Show first 10 samples for this variable
        num_to_show = min(10, len(sample_indices))
        for i in range(num_to_show):
            idx = sample_indices[i]
            ad_g = autodiff_samples[var_idx, i]
            fd_g = fd_grads[var_idx, i]
            rel_error = abs(ad_g - fd_g) / (abs(fd_g) + 1e-10)
            print(f"{idx:<8} {float(ad_g):<15.6e} {float(fd_g):<15.6e} {float(rel_error):<12.6e}")
        print()

    # Overall statistics across all variables
    rel_errors = jnp.abs(autodiff_samples - fd_grads) / (jnp.abs(fd_grads) + 1e-10)
    max_rel_error = jnp.max(rel_errors)
    mean_rel_error = jnp.mean(rel_errors)

    print(f"Relative error statistics:")
    print(f"  Max:  {float(max_rel_error):.6e}")
    print(f"  Mean: {float(mean_rel_error):.6e}")
    print()

    # Verdict
    if max_rel_error < 1e-3:
        print("✓ EXCELLENT: Autodiff gradients match finite differences (< 1e-3)")
    elif max_rel_error < 1e-2:
        print("✓ GOOD: Autodiff gradients approximately match finite differences (< 1e-2)")
    elif max_rel_error < 1e-1:
        print("⚠ WARNING: Moderate discrepancy (< 1e-1) - check epsilon or limiter sharpness")
    else:
        print("✗ FAILED: Large discrepancy! Gradients may be incorrect.")
    print()

    # -------------------------
    # Summary
    # -------------------------
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test problem: {cfg.test_problem}")
    print(f"Observable: {observable_name}")
    print(f"Smooth limiters: {cfg.use_smooth_limiters} (sharpness={cfg.smooth_limiter_sharpness})")
    print()
    print(f"Forward time: {t_forward:.3f}s")
    print(f"Gradient time: {t_grad:.3f}s")
    print(f"Gradient valid: {n_nans == 0 and n_infs == 0}")
    print(f"FD agreement: max rel error = {float(max_rel_error):.6e}")
    print()

    if cfg.use_smooth_limiters and n_nans == 0 and n_infs == 0 and max_rel_error < 1e-2:
        print("✓ SUCCESS: Solver is differentiable with smooth limiters!")
    else:
        print("⚠ Issues detected - see details above")

    print("="*80)

    # -------------------------
    # Save results to HDF5
    # -------------------------
    print()
    print("Saving results to HDF5...")
    output_file = write_autodiff_result(
        U0=U0,
        U_final=U_final,
        grads=grads_autodiff,
        fd_grads=fd_grads,
        fd_sample_indices=sample_indices,
        observable_value=obs_value,
        observable_name=observable_name,
        cfg=cfg,
        filename="autodiff_result.hdf5"
    )
    print()
    print("To visualize results, run:")
    print(f"  python utils/plot_autodiff_gradients.py {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
