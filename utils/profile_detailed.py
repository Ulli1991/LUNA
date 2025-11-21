"""
Detailed component-level profiling.

This profiles individual components like:
- Padding
- Slope computation
- Flux computation (HLLC)
- Divergence

Run this to understand which sub-components within flux_divergence are slow.

Usage:
    python utils/profile_detailed.py
    (from the project root directory)
"""

import os
import sys
import time
from functools import partial

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

# JAX configuration
jax.config.update("jax_enable_x64", False)

from config import Config
from hydro.padding import pad
from hydro.slopes import compute_slopes_from_padded
from hydro.fluxes import hllc_flux
from hydro.divergence import flux_divergence


def benchmark_component(func, name, *args, n_iters=100, **kwargs):
    """Benchmark a specific component."""
    print(f"\nBenchmarking: {name}")
    print("-" * 40)

    # Warmup
    result = func(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, tuple):
        for r in result:
            if hasattr(r, 'block_until_ready'):
                r.block_until_ready()

    # Benchmark
    times = []
    for _ in range(n_iters):
        jax.block_until_ready(jnp.array(0.0))  # Sync before
        t0 = time.time()

        result = func(*args, **kwargs)

        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()

        jax.block_until_ready(jnp.array(0.0))  # Sync after
        times.append((time.time() - t0) * 1000)  # ms

    avg = sum(times) / len(times)
    std = jnp.std(jnp.array(times))
    min_t = min(times)
    max_t = max(times)

    print(f"  Average: {avg:.3f} ms")
    print(f"  Std Dev: {std:.3f} ms")
    print(f"  Min:     {min_t:.3f} ms")
    print(f"  Max:     {max_t:.3f} ms")

    return avg


def main():
    print("="*60)
    print("DETAILED COMPONENT PROFILING")
    print("="*60)

    # Load config
    cfg = Config()

    # Initialize
    print("\nSetting up test data...")
    if cfg.SPATIAL_DIMENSION == 1:
        U = jnp.ones((3, 2048))
    elif cfg.SPATIAL_DIMENSION == 2:
        U = jnp.ones((4, 2048, 2048))
    else:
        U = jnp.ones((5, 512, 512, 512))

    print(f"Grid shape: {U.shape}")
    print(f"Boundary type: {['Periodic', 'Zero-gradient', 'Reflective'][cfg.BOUNDARY]}")

    # Precompute constants
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy if dy is not None else None
    inv_dz = 1.0 / dz if dz is not None else None

    print("\n" + "="*60)
    print("COMPONENT BENCHMARKS")
    print("="*60)

    results = {}

    # 1. Padding
    results['padding'] = benchmark_component(
        pad, "Padding (boundary application)",
        U, cfg.BOUNDARY,
        n_iters=100
    )

    # 2. Slope computation (requires padded array)
    U_pad = pad(U, cfg.BOUNDARY)
    results['slopes'] = benchmark_component(
        compute_slopes_from_padded, "Slope computation (PLM)",
        U_pad, cfg.LIMITER,
        n_iters=100
    )

    # 3. HLLC flux (2D case)
    if cfg.SPATIAL_DIMENSION == 2:
        U_L = U_pad[:, :-1, 1:-1]
        U_R = U_pad[:, 1:, 1:-1]
        results['hllc_flux_x'] = benchmark_component(
            hllc_flux, "HLLC flux (x-direction)",
            U_L, U_R, 0, cfg.gamma,
            n_iters=100
        )

        U_B = U_pad[:, 1:-1, :-1]
        U_T = U_pad[:, 1:-1, 1:]
        results['hllc_flux_y'] = benchmark_component(
            hllc_flux, "HLLC flux (y-direction)",
            U_B, U_T, 1, cfg.gamma,
            n_iters=100
        )

    # 4. Full flux divergence
    results['flux_divergence'] = benchmark_component(
        flux_divergence, "Full flux_divergence",
        U, cfg.gamma, cfg.ORDER, cfg.LIMITER, cfg.BOUNDARY,
        dx, dy, dz, inv_dx, inv_dy, inv_dz,
        n_iters=50
    )

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_component = (
        results['padding'] +
        results['slopes'] +
        results.get('hllc_flux_x', 0) +
        results.get('hllc_flux_y', 0)
    )

    total_measured = results['flux_divergence']

    print(f"\nComponent breakdown:")
    print(f"  Padding:           {results['padding']:.3f} ms ({results['padding']/total_component*100:.1f}%)")
    print(f"  Slopes:            {results['slopes']:.3f} ms ({results['slopes']/total_component*100:.1f}%)")

    if 'hllc_flux_x' in results:
        print(f"  HLLC flux (x):     {results['hllc_flux_x']:.3f} ms ({results['hllc_flux_x']/total_component*100:.1f}%)")
        print(f"  HLLC flux (y):     {results['hllc_flux_y']:.3f} ms ({results['hllc_flux_y']/total_component*100:.1f}%)")

    print(f"  ----------------")
    print(f"  Component sum:     {total_component:.3f} ms")
    print(f"  Full divergence:   {total_measured:.3f} ms")

    overhead = total_measured - total_component
    print(f"  Overhead/fusion:   {overhead:.3f} ms ({overhead/total_measured*100:.1f}%)")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    # Find largest component
    components = {
        'Padding': results['padding'],
        'Slopes': results['slopes'],
    }
    if 'hllc_flux_x' in results:
        components['HLLC flux (x)'] = results['hllc_flux_x']
        components['HLLC flux (y)'] = results['hllc_flux_y']

    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)

    print("\nBottlenecks (in order):")
    for i, (name, time_ms) in enumerate(sorted_components, 1):
        pct = time_ms / total_component * 100
        print(f"  {i}. {name}: {pct:.1f}% ({time_ms:.3f} ms)")

    if overhead > 0:
        print(f"\n✓ Negative overhead suggests XLA is fusing operations well")
    else:
        print(f"\n⚠️  Positive overhead suggests room for fusion optimization")


if __name__ == "__main__":
    main()
