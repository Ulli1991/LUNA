# Utils Directory

This directory contains profiling, benchmarking, and visualization utilities for JAX-Hydro.

## Profiling & Benchmarking Tools

### High-Level Profiling
```bash
python utils/profile_simulation.py
```
Profile the full simulation to see where time is spent (CFL, time stepping, validation).

### Detailed Component Profiling
```bash
python utils/profile_detailed.py
```
Benchmark individual components: padding, slopes, HLLC flux, and measure XLA fusion effectiveness.

### HLLC Flux Benchmark
```bash
python utils/benchmark_hllc.py
```
Benchmark the optimized HLLC flux implementation on a 4096×4096 grid.

## Documentation

See [`PROFILING_GUIDE.md`](PROFILING_GUIDE.md) for:
- Detailed usage instructions
- How to interpret profiling results
- Optimization recommendations
- Advanced profiling techniques

## Other Utilities

- `profiling.py` - Core profiling infrastructure (ProfileTimer class)
- `memory_utils.py` - Memory tracking and reporting
- `plot_1d_fluid_quants.py` - 1D fluid quantity visualization
- `plot_density2d.py` - 2D density visualization
- `quick_plot.py` - Quick plotting utilities

## Quick Reference

**Profile first, optimize second:**
1. Run `profile_simulation.py` to find bottlenecks
2. If time stepping dominates (expected), run `profile_detailed.py`
3. Optimize the slowest component
4. Repeat

**All scripts should be run from project root:**
```bash
# From /path/to/jax-hydro/
python utils/profile_simulation.py
```
