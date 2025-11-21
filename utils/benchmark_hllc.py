"""
Benchmark HLLC flux implementations to measure performance improvements.

Compares:
1. Current implementation (nested jnp.where)
2. Optimized implementation (specialized 1D/2D/3D + jnp.select)
"""
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

    # Import implementations
    from hydro.fluxes import hllc_flux as hllc_current
    from hydro.fluxes_optimized import hllc_flux as hllc_optimized

    # Benchmark current implementation
    time_current = benchmark_implementation(
        hllc_current, "CURRENT (nested jnp.where)",
        U_L, U_R, axis, gamma
    )

    # Benchmark optimized implementation
    time_optimized = benchmark_implementation(
        hllc_optimized, "OPTIMIZED (specialized + jnp.select)",
        U_L, U_R, axis, gamma
    )

    # Compare results
    print("="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Current implementation:   {time_current:.2f} ms")
    print(f"Optimized implementation: {time_optimized:.2f} ms")
    print()

    speedup = ((time_current - time_optimized) / time_current) * 100
    if time_optimized < time_current:
        print(f"🚀 SPEEDUP: {speedup:.1f}% faster")
        print(f"   ({time_current - time_optimized:.2f} ms saved per flux computation)")
    elif time_optimized > time_current:
        print(f"⚠️  SLOWDOWN: {-speedup:.1f}% slower")
        print(f"   (Current implementation is still faster)")
    else:
        print("No significant difference")

    print()
    print("="*60)
    print("RECOMMENDATION")
    print("="*60)
    if speedup > 1.0:
        print("✓ Replace hydro/fluxes.py with optimized version")
        print("  Expected improvement in full simulation: ~{:.1f}%".format(speedup * 0.6))
        print("  (HLLC is ~60-70% of total runtime)")
    else:
        print("✓ Keep current implementation")
        print("  The optimizations don't provide measurable benefit")
