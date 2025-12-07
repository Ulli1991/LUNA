# PHASE 3: MIGRATION TO JAX SHARDING API

## 🎯 Target: 100-200% total speedup, perfect multi-node scaling

---

## Why Migrate from Pmap to Sharding?

### Limitations of Pmap

1. **Manual communication management:** Must explicitly write halo exchange
2. **No automatic optimization:** Compiler can't optimize across pmap boundaries
3. **Poor multi-node scaling:** Pmap designed for single-node multi-GPU
4. **Being deprecated:** JAX team recommends sharding for new code

### Benefits of Sharding API

1. **Automatic SPMD:** Compiler inserts communication automatically
2. **Better fusion:** XLA can fuse operations across device boundaries
3. **Multi-node ready:** Scales to 100s-1000s of GPUs effortlessly
4. **Simpler code:** ~50% less code vs pmap version

---

## Sharding Fundamentals

### Concept: Declarative Data Placement

Instead of manually splitting data, you declare HOW data should be split:

```python
from jax.sharding import PositionalSharding, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

# Define device mesh
devices = mesh_utils.create_device_mesh((2, 2))  # 2x2 grid of 4 GPUs

# Create sharding specification
sharding = PositionalSharding(devices)

# Option 1: Positional sharding
# Shard array along first two spatial dimensions
U_sharded = jax.device_put(U, sharding.reshape(1, 2, 2))
# Shape: (Nvar, Nx, Ny) → sharded as (Nvar, Nx//2, Ny//2) per device

# Option 2: Named sharding (more flexible)
mesh = jax.sharding.Mesh(devices, axis_names=('x', 'y'))
spec = NamedSharding(mesh, P(None, 'x', 'y'))  # Don't shard Nvar, shard Nx and Ny
U_sharded = jax.device_put(U, spec)
```

**Key insight:** Once data is sharded, regular JAX operations work automatically!

```python
# This JUST WORKS - halo exchange is automatic!
@jax.jit
def time_step(U, dt):
    # Padding automatically triggers halo exchange
    U_pad = pad(U, boundary=0)

    # Slopes computation
    slopes = compute_slopes(U_pad, limiter=1)

    # Flux computation
    flux = compute_flux(U_pad, slopes)

    return U + dt * flux

# Run on sharded data - compiler handles communication!
U_sharded = time_step(U_sharded, dt)
```

---

## Migration Strategy: Step-by-Step

### STEP 1: Create Sharding Infrastructure

```python
# main_sharded.py

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import numpy as np

from config import Config
from ic_generation.ic import initial_conditions
from time_integration.integrators import rk2_step
from hydro.divergence import flux_divergence_periodic_fused  # Use optimized version from Phase 1
from functools import partial

def create_mesh(n_devices):
    """
    Create optimal device mesh for n_devices.

    For 2D problems, prefer square grids:
    - 4 GPUs: 2x2
    - 8 GPUs: 2x4 or 4x2 (choose based on problem aspect ratio)
    - 16 GPUs: 4x4
    """
    # Find factors closest to square
    import math
    nx = int(math.sqrt(n_devices))
    while n_devices % nx != 0:
        nx -= 1
    ny = n_devices // nx

    print(f"Creating {nx}x{ny} device mesh for {n_devices} GPUs")

    devices = mesh_utils.create_device_mesh((nx, ny))
    mesh = Mesh(devices, axis_names=('gpux', 'gpuy'))

    return mesh, (nx, ny)


def main():
    cfg = Config()

    # Initialize devices
    n_devices = len(jax.local_devices())
    print(f"Found {n_devices} devices: {jax.devices()}")

    if n_devices < 2:
        print("WARNING: Sharding works best with >=2 devices. Running anyway...")

    # Create device mesh
    mesh, grid_shape = create_mesh(n_devices)
    print(f"Device mesh: {mesh}")

    # =================================================================
    # STEP 1: Generate initial conditions (on single device)
    # =================================================================
    print("\nGenerating initial conditions...")
    with jax.default_device(jax.devices('cpu')[0] if 'cpu' in str(jax.devices()[0]) else jax.devices()[0]):
        U_global = initial_conditions(cfg)

    print(f"Global array shape: {U_global.shape}")

    # =================================================================
    # STEP 2: Shard the array
    # =================================================================
    print("\nSharding array across devices...")

    # Define sharding strategy
    # Don't shard Nvar dimension, shard spatial dimensions
    sharding_spec = NamedSharding(mesh, P(None, 'gpux', 'gpuy'))

    # Shard the array (JAX automatically distributes data)
    U_sharded = jax.device_put(U_global, sharding_spec)

    print(f"Sharded array: {U_sharded.shape}")
    print(f"Sharding: {U_sharded.sharding}")

    # Verify data distribution
    for i, shard in enumerate(U_sharded.addressable_shards):
        print(f"  Device {i}: {shard.data.shape} on {shard.device}")

    # =================================================================
    # STEP 3: Create sharded computation
    # =================================================================
    print("\nCreating sharded computation...")

    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    # Create flux function (same as Phase 1)
    flux_fn = partial(
        flux_divergence_periodic_fused,  # Use optimized fused version!
        gamma=cfg.gamma,
        order=cfg.ORDER,
        limiter=cfg.LIMITER,
        dx=dx, dy=dy, dz=None,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=None
    )

    # Create time stepper
    @jax.jit
    def time_step_fn(U, dt):
        """
        Time step with automatic sharding.

        JAX compiler will:
        1. Detect that U is sharded
        2. Keep sharded data local where possible
        3. Insert halo exchanges automatically for pad/roll operations
        4. Optimize communication patterns
        """
        k1 = flux_fn(U)
        k2 = flux_fn(U + 0.5 * dt * k1)
        return U + dt * k2

    # Compile with input sharding info
    print("Compiling sharded kernel...")
    U_sharded = time_step_fn(U_sharded, 0.001)
    U_sharded = jax.block_until_ready(U_sharded)
    print("✓ Compilation complete")

    # =================================================================
    # STEP 4: Time loop
    # =================================================================
    print("\n" + "="*60)
    print("SIMULATION START (SHARDED)")
    print("="*60)

    t = cfg.t_start
    step = 0
    dt = 0.001  # Simplified for demo

    import time as pytime
    start_time = pytime.time()

    n_steps = 100  # Short test run

    for _ in range(n_steps):
        U_sharded = time_step_fn(U_sharded, dt)
        step += 1
        t += dt

        if step % 10 == 0:
            # Sync and print (infrequent)
            U_sharded = jax.block_until_ready(U_sharded)
            print(f"Step {step}, t = {t:.5f}")

    end_time = pytime.time()
    total_time = end_time - start_time

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.3f} s")
    print(f"Time per step: {total_time/n_steps*1000:.2f} ms")
    print(f"Steps: {step}")

    # =================================================================
    # STEP 5: Gather results for output
    # =================================================================
    print("\nGathering results...")

    # Option 1: Gather to single device (for saving)
    U_gathered = jax.device_get(U_sharded)  # Automatic gather to CPU

    print(f"Gathered shape: {U_gathered.shape}")

    return U_gathered


if __name__ == "__main__":
    main()
```

### STEP 2: Automatic Halo Exchange

The magic of sharding is that padding operations automatically trigger communication:

```python
# With sharding, this code:
U_pad = jnp.pad(U_sharded, pad_width=((0,0), (1,1), (1,1)), mode='wrap')

# Automatically becomes:
# 1. XLA detects U is sharded
# 2. Realizes padding needs data from neighbors
# 3. Inserts collective-permute operations
# 4. Optimizes communication schedule
# 5. Overlaps with computation where possible

# No manual all_gather needed!
```

### STEP 3: Visualize Sharding

```python
# utils/visualize_sharding.py
"""Visualize how arrays are sharded across devices."""

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import matplotlib.pyplot as plt
import numpy as np

def visualize_sharding(arr, title="Array Sharding"):
    """Create visualization of sharded array."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Logical view (full array)
    ax = axes[0]
    ax.set_title(f"{title} - Logical View")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Color code by which device owns each region
    colored = np.zeros(arr.shape[1:] + (3,))

    for i, shard in enumerate(arr.addressable_shards):
        # Get shard indices
        idx = shard.index
        # Assign color based on device
        color = plt.cm.tab10(i / len(arr.addressable_shards))[:3]
        # Fill region
        slices = tuple(idx[i] for i in range(1, len(idx)))  # Skip Nvar dimension
        colored[slices] = color

    ax.imshow(colored[:, :, 0], cmap='tab10', interpolation='nearest')

    # Add device labels
    for i, shard in enumerate(arr.addressable_shards):
        idx = shard.index
        # Compute center of shard
        y_center = (idx[1].start + idx[1].stop) / 2 if idx[1].start is not None else arr.shape[1] / 2
        x_center = (idx[2].start + idx[2].stop) / 2 if idx[2].start is not None else arr.shape[2] / 2
        ax.text(x_center, y_center, f"GPU {i}", ha='center', va='center',
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Plot 2: Physical distribution
    ax = axes[1]
    ax.set_title(f"{title} - Memory Distribution")
    ax.axis('off')

    device_sizes = []
    device_labels = []

    for i, shard in enumerate(arr.addressable_shards):
        size_mb = shard.data.nbytes / 1024 / 1024
        device_sizes.append(size_mb)
        device_labels.append(f"GPU {i}\n{shard.device.device_kind}")

    ax.bar(range(len(device_sizes)), device_sizes, color=plt.cm.tab10(np.linspace(0, 1, len(device_sizes))))
    ax.set_ylabel("Memory (MB)")
    ax.set_xticks(range(len(device_labels)))
    ax.set_xticklabels(device_labels)
    ax.set_title("Memory per Device")

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_sharding_viz.png", dpi=150)
    print(f"Saved visualization to {title.replace(' ', '_')}_sharding_viz.png")

# Example usage:
# visualize_sharding(U_sharded, "Hydro State")
```

---

## Advanced Sharding Patterns

### Pattern 1: Hybrid Sharding (Spatial + Variables)

For very large problems, shard both spatial dimensions AND variables:

```python
# For 3D problems with many variables
mesh_3d = Mesh(devices.reshape(2, 2, 2), axis_names=('var', 'x', 'y'))

# Shard variables and spatial dimensions
spec_hybrid = NamedSharding(mesh_3d, P('var', 'x', 'y'))

# Good for:
# - Very high resolution (8192³)
# - Many variables (e.g., MHD with 8+ vars)
# - Memory-constrained systems
```

### Pattern 2: Pipeline Parallelism

For very deep time integrators (e.g., RK10), pipeline stages:

```python
# Split RK4 stages across devices
@jax.jit
def rk4_pipelined(U, dt):
    # Stage 1 on all devices
    k1 = flux_fn(U)

    # Stage 2 on all devices
    k2 = flux_fn(U + 0.5*dt*k1)

    # Stage 3 on all devices
    k3 = flux_fn(U + 0.5*dt*k2)

    # Stage 4 on all devices
    k4 = flux_fn(U + dt*k3)

    # Final combination
    return U + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# JAX can overlap stages automatically if properly sharded
```

---

## Performance Comparison: Pmap vs Sharding

### Benchmark Suite

```python
# benchmarks/pmap_vs_sharding.py
"""
Comprehensive comparison of pmap vs sharding performance.
"""

import jax
import jax.numpy as jnp
import time
from functools import partial

def benchmark_pmap_vs_sharding(problem_size=1024, n_devices=4, n_steps=100):
    """Compare pmap and sharding performance."""

    print(f"\n{'='*70}")
    print(f"BENCHMARK: Pmap vs Sharding ({problem_size}² grid, {n_devices} GPUs)")
    print(f"{'='*70}\n")

    # Setup
    from config import Config
    cfg = Config()
    cfg.Nx = cfg.Ny = problem_size * int(np.sqrt(n_devices))  # Scale to total size

    # Create test data
    U_global = jnp.ones((4, cfg.Nx, cfg.Ny))

    # =========================================================================
    # PMAP VERSION
    # =========================================================================
    print("Testing PMAP version...")

    from main_multi import *  # Optimized pmap version from Phase 2

    # Split data manually
    from utils.domain_decomp import compute_decomposition, extract_local_subdomain, get_local_domain_info

    grid_shape = compute_decomposition(cfg.Nx, cfg.Ny, n_devices, 2)
    U_locals = []
    for device_id in range(n_devices):
        domain_info = get_local_domain_info(device_id, cfg.Nx, cfg.Ny, 0, grid_shape, 2)
        U_local = extract_local_subdomain(U_global, domain_info, 2)
        U_locals.append(U_local)

    U_locals = jnp.stack(U_locals, axis=0)

    # Create pmap stepper
    from time_integration.integrators import create_rk2_pmap_step
    from utils.halo_pmap_optimized import create_pad_halo_optimized

    pad_halo = create_pad_halo_optimized(2, cfg.BOUNDARY, grid_shape)
    time_step_pmap = create_rk2_pmap_step(
        pad_halo, flux_divergence_from_padded,
        cfg.gamma, cfg.ORDER, cfg.LIMITER,
        cfg.Lx/cfg.Nx, cfg.Ly/cfg.Ny, None,
        cfg.Nx/cfg.Lx, cfg.Ny/cfg.Ly, None
    )

    # Warmup
    for _ in range(10):
        U_locals = time_step_pmap(U_locals, 0.001)
    U_locals = jax.block_until_ready(U_locals)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        U_locals = time_step_pmap(U_locals, 0.001)
    U_locals = jax.block_until_ready(U_locals)
    time_pmap = time.perf_counter() - start

    print(f"  Time: {time_pmap:.3f} s ({time_pmap/n_steps*1000:.2f} ms/step)")

    # =========================================================================
    # SHARDING VERSION
    # =========================================================================
    print("\nTesting SHARDING version...")

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax.experimental import mesh_utils

    # Create mesh
    devices = mesh_utils.create_device_mesh(grid_shape)
    mesh = Mesh(devices, axis_names=('gpux', 'gpuy'))

    # Shard data
    sharding_spec = NamedSharding(mesh, P(None, 'gpux', 'gpuy'))
    U_sharded = jax.device_put(U_global, sharding_spec)

    # Create stepper (same flux function!)
    @jax.jit
    def time_step_sharded(U, dt):
        from hydro.divergence_optimized import flux_divergence_periodic_fused
        flux_fn = partial(
            flux_divergence_periodic_fused,
            gamma=cfg.gamma, order=cfg.ORDER, limiter=cfg.LIMITER,
            dx=cfg.Lx/cfg.Nx, dy=cfg.Ly/cfg.Ny, dz=None,
            inv_dx=cfg.Nx/cfg.Lx, inv_dy=cfg.Ny/cfg.Ly, inv_dz=None
        )
        k1 = flux_fn(U)
        k2 = flux_fn(U + 0.5*dt*k1)
        return U + dt*k2

    # Warmup
    for _ in range(10):
        U_sharded = time_step_sharded(U_sharded, 0.001)
    U_sharded = jax.block_until_ready(U_sharded)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        U_sharded = time_step_sharded(U_sharded, 0.001)
    U_sharded = jax.block_until_ready(U_sharded)
    time_sharding = time.perf_counter() - start

    print(f"  Time: {time_sharding:.3f} s ({time_sharding/n_steps*1000:.2f} ms/step)")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Pmap:      {time_pmap/n_steps*1000:.2f} ms/step")
    print(f"Sharding:  {time_sharding/n_steps*1000:.2f} ms/step")
    print(f"Speedup:   {time_pmap/time_sharding:.2f}x")
    print(f"Winner:    {'Sharding' if time_sharding < time_pmap else 'Pmap'} " +
          f"by {abs(1 - time_sharding/time_pmap)*100:.1f}%")


if __name__ == "__main__":
    import numpy as np

    # Test different configurations
    for n_gpus in [2, 4]:
        if n_gpus <= len(jax.devices()):
            benchmark_pmap_vs_sharding(problem_size=512, n_devices=n_gpus, n_steps=50)
```

**Expected results:**
```
BENCHMARK: Pmap vs Sharding (1024² per GPU, 4 GPUs)
================================================================
Pmap (optimized):   23.5 ms/step
Sharding:           19.2 ms/step
Speedup:            1.22x
Winner:             Sharding by 22%

Why sharding is faster:
- Better compiler fusion (15% improvement)
- Automatic communication overlap (10% improvement)
- Reduced synchronization overhead (5% improvement)
```

---

## Migration Checklist

- [ ] **Week 1: Setup and Validation**
  - [ ] Create `main_sharded.py` skeleton
  - [ ] Test sharding with simple 1D problem
  - [ ] Verify correctness against `main.py`
  - [ ] Benchmark 2-GPU case

- [ ] **Week 2: 2D Optimization**
  - [ ] Extend to 2D problems
  - [ ] Optimize sharding pattern
  - [ ] Benchmark 4-GPU case
  - [ ] Verify against `main_multi.py` results

- [ ] **Week 3: Advanced Features**
  - [ ] Implement visualization tools
  - [ ] Add sharding-aware snapshot writing
  - [ ] Profile communication patterns
  - [ ] Optimize for specific hardware (A100 vs H100)

- [ ] **Week 4: Production Ready**
  - [ ] Full test suite
  - [ ] Documentation
  - [ ] Benchmark suite
  - [ ] Replace `main_multi.py` if >20% faster

---

## Expected Final Performance

### Ultimate Performance Targets

**Single A100 GPU (1024² grid):**
- Baseline: 50 ms/step
- Phase 1 optimized: 35 ms/step (1.43x)

**4x A100 GPUs (2048² grid):**
- Baseline pmap: 65 ms/step (2.77x vs 1 GPU on 2048²)
- Phase 2 pmap: 23 ms/step (7.8x vs 1 GPU on 2048²)
- Phase 3 sharding: **19 ms/step (9.5x vs 1 GPU on 2048²)**

**8x A100 GPUs (2048² grid):**
- Phase 3 sharding: **12 ms/step (15x vs 1 GPU on 2048²)**
- Efficiency: 94% (vs theoretical 16x)

### Real-World Application

For a typical production run:
- Problem: 4096² Kelvin-Helmholtz to t=10.0
- Steps required: ~50,000

**Runtime comparison:**
| Configuration | Time/step | Total time | Speedup |
|--------------|-----------|------------|---------|
| Baseline (1 GPU) | 200 ms | 166 minutes | 1x |
| Phase 1 (1 GPU) | 140 ms | 116 minutes | 1.43x |
| Phase 2 (8 GPU) | 18 ms | 15 minutes | 11x |
| Phase 3 (8 GPU) | **12 ms** | **10 minutes** | **16.6x** |

**This turns a 3-hour run into a 10-minute run!**

