# Profiling Guide

This guide explains how to profile your JAX-Hydro simulations to identify performance bottlenecks.

---

## Quick Start

### 1. High-Level Profiling (Recommended First)

Profile the main simulation to see where time is spent overall:

```bash
python utils/profile_simulation.py
```

**What it does:**
- Runs a shortened simulation (~100-500 steps)
- Times major sections: CFL computation, time stepping, validation
- Provides a report with bottleneck analysis

**Expected output:**
```
PERFORMANCE PROFILE
============================================================
Section                        Avg (ms)     % Total
------------------------------------------------------------
time_step_total                   5.23        78.2%
cfl_computation                   1.12        16.8%
validation                        0.33         5.0%
------------------------------------------------------------
TOTAL                             6.68       100.0%

BOTTLENECK ANALYSIS
------------------------------------------------------------
1. time_step_total: 78.2% of total time
```

---

### 2. Detailed Component Profiling

Profile individual components within flux_divergence:

```bash
python utils/profile_detailed.py
```

**What it does:**
- Benchmarks individual operations: padding, slopes, HLLC flux
- Measures each component in isolation
- Shows overhead/fusion from combining operations

**Expected output:**
```
COMPONENT BENCHMARKS
============================================================
Padding:           0.15 ms (5.2%)
Slopes:            0.52 ms (18.1%)
HLLC flux (x):     1.21 ms (42.1%)
HLLC flux (y):     1.19 ms (41.4%)
  ----------------
Component sum:     3.07 ms
Full divergence:   2.52 ms
Overhead/fusion:  -0.55 ms (-17.9%)

✓ Negative overhead suggests XLA is fusing operations well
```

---

## Profiling Tools

### `utils/profiling.py`

Core profiling utilities with GPU synchronization support.

**Key classes:**
- `ProfileTimer` - Context manager for timing code sections
- `profile_section()` - Convenience context manager

**Usage in your own code:**
```python
from utils.profiling import ProfileTimer, profile_section

# Enable profiling
ProfileTimer.enable()

# Time a section
with profile_section("my_expensive_operation"):
    result = expensive_function()

# Report results
ProfileTimer.report()
```

---

### `profile_simulation.py`

**Purpose:** High-level profiling of the main simulation loop

**Key features:**
- Automatically shortens simulation for quick profiling
- Times CFL computation, time stepping, validation separately
- GPU-synchronized timing (accurate for async JAX operations)
- Bottleneck identification

**Configuration:**
Edit the script to change:
- Grid size (default: uses your config)
- Number of steps (default: ~0.1 time units)
- Profiling granularity

---

### `profile_detailed.py`

**Purpose:** Component-level profiling of flux_divergence internals

**Key features:**
- Benchmarks padding, slopes, HLLC flux individually
- Measures XLA fusion effectiveness (negative overhead = good)
- Identifies which sub-component is slowest

**What to look for:**
- If HLLC flux > 70%: HLLC is the bottleneck (already optimized!)
- If slopes > 30%: Slope computation could be optimized
- If padding > 20%: Boundary conditions could be optimized
- Negative overhead: XLA is doing a great job fusing operations

---

## Interpreting Results

### Example 1: HLLC Dominates

```
HLLC flux (x):     1.21 ms (42.1%)
HLLC flux (y):     1.19 ms (41.4%)
```

**Interpretation:** HLLC is 83.5% of flux_divergence time
**Action:** HLLC already optimized. Consider multi-GPU scaling instead.

---

### Example 2: Slopes Are Slow

```
Slopes:            1.52 ms (48.1%)
HLLC flux (x):     0.81 ms (25.6%)
```

**Interpretation:** Slope computation is the bottleneck
**Action:**
- Check if XLA is fusing slope operations
- Consider manual kernel fusion
- Verify limiter choice (minmod vs moncen vs vanleer)

---

### Example 3: Padding Overhead

```
Padding:           0.95 ms (32.1%)
```

**Interpretation:** Boundary conditions taking too long
**Action:**
- Check which boundary type (reflective is slowest)
- Consider optimizing reflective boundaries further
- Use periodic/zero-gradient if physically acceptable

---

## Advanced Profiling

### JAX's Built-in Profiler

For more detailed GPU kernel analysis:

```bash
# TensorBoard profiling (NVIDIA/AMD)
JAX_PROFILER_MODE=tensorboard python main.py

# View in TensorBoard
tensorboard --logdir=/tmp/tensorboard
```

**What it shows:**
- Individual GPU kernel launches
- Memory transfers
- Kernel fusion decisions
- Detailed timeline view

---

### GPU-Specific Tools

#### For AMD (ROCm):

```bash
# Profile with rocprof
rocprof --hip-trace python main.py

# View results
rocprof --stats output.csv
```

#### For NVIDIA (CUDA):

```bash
# Profile with nsys
nsys profile -o profile python main.py

# Profile with ncu (detailed kernel analysis)
ncu --set full python main.py
```

---

## What to Optimize Based on Results

### If time_step is 70-80% of runtime ✅
**Status:** Normal and expected
**Action:** Profile deeper with `profile_detailed.py`

### If CFL computation > 20% ⚠️
**Issue:** CFL taking too long
**Fix:** Already fused with diagnostics. Check if can reduce frequency.

### If validation > 5% ⚠️
**Issue:** Too frequent validation
**Fix:** Reduce validation frequency (currently every 100 steps)

### If I/O (snapshots) > 10% ⚠️
**Issue:** Writing snapshots too often
**Fix:** Reduce snapshot frequency or use async I/O

---

## Performance Expectations

### Current Performance (Post-Optimization)

Based on the HLLC benchmarks:
- **HLLC flux:** 2.52 ms (4096×4096 grid)
- **Expected flux_divergence:** ~3-4 ms
- **Expected full timestep (RK4):** ~12-16 ms (4× divergence calls)

### Target Breakdown

For optimal performance on GPU:
```
flux_divergence: 60-70% (HLLC + slopes + padding)
CFL computation: 10-15%
Time integration: 10-15% (RK overhead)
Validation:       1-5%
I/O:              5-10%
```

---

## Next Steps After Profiling

### If Single-GPU Performance is Maxed Out
- Implement multi-GPU domain decomposition (see `SCALING_STRATEGY.md`)
- Expected gain: Linear scaling up to 1000s of GPUs

### If HLLC is Still the Bottleneck
- Consider lower-order methods (first-order Godunov)
- Or accept that you've hit hardware limits
- Or write custom CUDA/ROCm kernels (high effort)

### If Memory-Bound
- Check memory bandwidth utilization
- Reduce data movement (already done with reciprocal caching)
- Consider mixed precision (risky for science)

---

## Questions?

If profiling shows unexpected results:
1. Verify JAX is using the GPU (`jax.devices()`)
2. Check that arrays are on GPU (`U.device()`)
3. Ensure `block_until_ready()` is called for accurate timing
4. Run with `JAX_TRACEBACK_FILTERING=off` for full error traces

---

**Date:** 2025-11-21
**Status:** Profiling tools ready to use
