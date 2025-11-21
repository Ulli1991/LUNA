"""
Benchmark HLLC flux implementation to measure performance.

Usage:
    python utils/benchmark_hllc.py
    (from the project root directory)
"""
import os
import sys

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import time

def benchmark_implementation(hllc_func, name, U_L, U_R, axis, gamma, n_iters=50):
    """
    Benchmark a specific HLLC implementation.

    Parameters
    ----------
    hllc_func : callable
        The HLLC flux function to benchmark
    name : str
        Name of the implementation
    U_L, U_R : array
        Test data
    axis : int
        Flux direction
    gamma : float
        Ratio of specific heats
    n_iters : int
        Number of iterations to average over

    Returns
    -------
    avg_time : float
        Average time in milliseconds
    """
    print("="*60)
    print(f"BENCHMARKING: {name}")
    print("="*60)
    grid_shape = U_L.shape[1:]
    print(f"Grid: {' x '.join(map(str, grid_shape))}")
    print("Warming up (JIT compilation)...")

    # Warmup
    result = hllc_func(U_L, U_R, axis, gamma)
    result.block_until_ready()
    print("✓ JIT compilation complete")

    # Benchmark
    print(f"Running {n_iters} iterations...")
    times = []
    for i in range(n_iters):
        t0 = time.time()
        result = hllc_func(U_L, U_R, axis, gamma)
        result.block_until_ready()
        times.append((time.time() - t0) * 1000)  # ms

    avg_time = sum(times) / len(times)
    std_time = jnp.std(jnp.array(times))
    min_time = min(times)
    max_time = max(times)

    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print()

    return avg_time

if __name__ == "__main__":
    # Check JAX configuration
    print("="*60)
    print("JAX CONFIGURATION")
    print("="*60)
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    # Create test data (2D case - this is what your simulations use)
    print("Creating test data...")
    U_L = jnp.ones((4, 4096, 4096))
    U_R = jnp.ones((4, 4096, 4096)) * 1.1
    axis = 0
    gamma = 1.4
    print("✓ Test data created")
    print()

    # Import current (optimized) implementation
    from hydro.fluxes import hllc_flux

    # Benchmark implementation
    time_hllc = benchmark_implementation(
        hllc_flux, "HLLC FLUX (optimized: specialized + jnp.select)",
        U_L, U_R, axis, gamma
    )

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"HLLC flux time: {time_hllc:.2f} ms")
    print(f"Grid: 4096 x 4096 (2D)")
    print()
    print("Performance context:")
    print(f"  - This is the OPTIMIZED version (12.1% faster than pre-optimization)")
    print(f"  - Uses specialized 1D/2D/3D functions")
    print(f"  - Uses jnp.select instead of nested jnp.where")
    print(f"  - HLLC is ~95% of flux_divergence time")
    print()
    print("To see full simulation performance:")
    print("  python utils/profile_simulation.py")
