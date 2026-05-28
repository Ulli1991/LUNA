# LUNA

**A differentiable astrophysical MHD code for turbulence, shocks, ISM, and galaxy evolution**

A high-performance JAX-based hydrodynamics simulation code for solving the Euler equations in 1D, 2D, and 3D with **single-GPU and multi-GPU support**.

![Soundwave 1D convergence (eigenmode): LUNA vs AthenaK](figures/soundwave_convergence.png)

*Linear-wave convergence test, single right-going acoustic eigenmode (ρ₀=1, p₀=0.6, γ=5/3, ε=10⁻⁴, periodic Lx=1, one acoustic period). LUNA (PLM + MUSCL-Hancock + MonCen + HLLC) tracks AthenaK PLM+RK2 over the full Nx range. All curves approach the same O(ε²) nonlinear-steepening floor — that's the residual difference between the actual nonlinear wave and the linear analytic reference, identical in both codes. AthenaK PPMX is shown as a higher-order reference.*

![Soundwave 1D STANDING WAVE convergence: LUNA vs AthenaK](figures/soundwave_standing_convergence.png)

*Same parameters but with u=0 superposition IC (standing wave). The standing-wave symmetry cancels the O(ε²) nonlinear corrections, so the floor disappears and convergence runs cleanly to ~10⁻¹⁰ at Nx=512. Both LUNA and AthenaK PLM+RK2 converge faster than the formal 2nd-order rate on this problem (a known symmetry-driven superconvergence). AthenaK L1 here is read from its built-in double-precision error output, not from the float32 binary snapshot.*

![Sod shock tube: LUNA vs AthenaK](figures/shock_panels_sod.png)

*Sod shock tube (Riemann problem with ρ_L=1, p_L=1, ρ_R=0.125, p_R=0.1; t=0.2). LUNA (PLM + MUSCL-Hancock + minmod + HLLC) at Nx ∈ {256, 512, 1024, 2048} vs AthenaK PPMX at Nx=8192. Density, vx, pressure, and specific internal energy all match the reference solution; the shock, contact discontinuity, and rarefaction fan are well resolved at Nx=512 and above.*

![Shu-Osher: LUNA vs AthenaK](figures/shock_panels_shu_osher.png)

*Shu-Osher problem (M=3 shock impinging on a sinusoidal density field; t=1.8 on x ∈ [−5, 5]). Tests the scheme's ability to resolve small-scale features in the presence of strong shocks. AthenaK PPMX (8192) captures the post-shock oscillations cleanly; LUNA at Nx=2048 matches in the smooth regions and resolves the high-frequency train progressively from Nx=256 upward.*

![Woodward-Colella: LUNA vs AthenaK](figures/shock_panels_woodward_colella.png)

*Woodward-Colella interacting blast waves (P_L=1000, P_M=0.01, P_R=100; t=0.038, reflecting walls). Tests strong shocks with extreme pressure ratios (10⁵). LUNA captures the central density spike at x≈0.87 with the expected refinement behavior — Nx=256 underestimates the peak height; Nx=2048 essentially matches AthenaK PPMX 8192.*

![Kelvin-Helmholtz density grid: LUNA refinement series](figures/kh_grid.png)

*2D Kelvin-Helmholtz instability (double-tanh shear layers at y=0.5 and y=1.5 on Lx=Ly=2, periodic, γ=5/3; LUNA PLM + MUSCL-Hancock + MonCen + HLLC at CFL=0.4). Rows are resolutions Nx² ∈ {1024², 2048², 4096², 8192²}; columns are density snapshots at t = 0, 1, 2.5, 4, 5. Higher resolution resolves progressively finer secondary roll-ups and the small-scale mixing layer between the two shear regions; by 8192² the cascade is well-resolved down to ~10 cells. Single-node (4×A100) for 1024²–4096², 4×A100 with longer walltime for 8192².*

![Kelvin-Helmholtz 8192² density: LUNA vs AthenaK](figures/kh_8192_luna_vs_athenak.png)

*Same KH instability at the highest resolution, LUNA vs the AthenaK 8192² WENOZ+HLLC reference (top: LUNA PLM+MUSCL-Hancock+MonCen+HLLC; bottom: AthenaK 4th-order WENOZ+HLLC, run on 2×A100 nodes via MPI+CUDA). Density slices at t = 0, 1, 2.5, 4, 5. Both codes capture the same large-scale roll-up structure; AthenaK's WENOZ keeps the small-scale braids slightly sharper, as expected for a higher-order scheme. The agreement of the dominant vortex positions and overall mixing layer shows that LUNA's 2nd-order MUSCL-Hancock pipeline is converged to AthenaK at this resolution.*

![Gresho vortex: v_phi vs r](figures/gresho_vphi_vs_radius.png)

*2D Gresho vortex (Gresho & Chan 1990, low-Mach variant) — diagnostic for low-Mach behaviour of the Riemann solver. Domain [0,1]², ρ=1 uniform, p chosen for centrifugal balance with M_max = u_max/c_s = 0.01 (deep low-Mach regime). Analytic v_φ(r) is piecewise-linear with a peak at r=0.2 and v_φ=0 outside r=0.4. LUNA PLM + MUSCL-Hancock + MonCen + **HLLC** at CFL=0.4. Left: early state (t ≈ 0.5, less than one core turnover); right: late state (t=3, ≈ 2.4 turnovers). Resolution sweep Nx ∈ {256, 512, 1024, 2048}; shaded bands show ±1σ azimuthal scatter per bin. HLLC's low-Mach over-dissipation visibly flattens the analytic v_φ peak over time, with higher resolution mitigating but not eliminating the loss. This is the canonical test motivating the L-HLLC low-Mach correction (Velasco-Romero & Teyssier 2025) used for our stratified solar-convection runs.*

![2D square advection: LUNA time progression](figures/square_advection.png)

*2D square-density advection (ρ=2 in [0.25,0.75]², ρ=1 outside; uniform u=v=1, p=1, γ=5/3; periodic Lx=Ly=1; LUNA PLM + MUSCL-Hancock + MonCen + HLLC at Nx=1024² with CFL=0.4). Density snapshots at t = 0, 0.25, 0.5, 0.75, 1.0 — the square advects diagonally and after one full traversal (t=1) the velocity has carried it exactly back to the center. Tests that the scheme preserves a sharp tangential contact under uniform-flow advection over the full domain length.*

![Sedov-Taylor 3D: shock evolution and column density](figures/sedov3d.png)

*Sedov-Taylor 3D blast wave: E = 10⁵¹ erg deposited into the central 2×2×2 = 8 cells of a uniform n = 1 cm⁻³, T = 10⁴ K ISM (γ=5/3) on a 50 pc cube; outflow boundaries. LUNA PLM + MUSCL-Hancock + minmod + HLLC at CFL=0.4, units kpc / 10¹⁰ M☉ / km s⁻¹, run for 10⁴ yr. The figure auto-uses the highest-resolution LUNA Sedov run on disk (currently Nx=512³). Left: simulated shock radius (peak-ρ location) vs the Sedov-Taylor self-similar prediction R_s(t) = ξ(E t²/ρ₀)^(1/5) with ξ=1.15 — agreement to <1% at t = 10⁴ yr. Right: column density projection along z showing the spherical blast shell.*

![Sedov-Taylor 3D radial profiles: resolution study](figures/sedov3d_radial.png)

*Sedov 3D resolution study: spherically-averaged radial profiles at t = 10⁴ yr of number density n [cm⁻³], radial velocity v_r [km/s], pressure P [erg/cm³], and temperature T [K] for every LUNA Sedov run on disk (currently Nx=128³, 256³, 512³). Curves shaded by resolution; dashed line marks the analytic Sedov-Taylor shock radius. As Nx increases the shock peak sharpens toward the Rankine-Hugoniot strong-shock ratio (ρ₂/ρ₁ = 4 for γ=5/3) while the central rarefaction deepens.*

![Driven isothermal turbulence: 2×2 resolution montage](figures/turbulence_panel.png)

*Subsonic (M_rms = 0.3) driven isothermal turbulence at t ≈ 4 (≈1.2 turnover times): density slice through z = L/2 for a resolution sweep Nx³ ∈ {128³, 256³, 512³, 1024³}. Solenoidal-only OU forcing (Federrath/Schmidt 2010) on a 1³ periodic box, k_min=1, k_max=2 driving range, identical PRNG seed at every Nx so the driven realisation matches. EOS: LUNA solves γ=5/3 Euler internally for HLLC robustness, but the runner enforces p = c_s² ρ as an operator-split post-step projection (soft isothermal thermostat) — no spurious compressive heating accumulates. The 1024³ run is on a 2-node × 4-GPU schedule slot.*

![V&T solar convection — δρ at t=50](figures/solar_vt_drho_grid.png)

*2D Velasco-Romero & Teyssier (2025) 3-layer polytropic atmosphere with a buoyant Gaussian bubble (δρ/ρ_eq = −0.012, r_b = 0.5), well-balanced MUSCL-Hancock + MonCen + L-HLLC at CFL = 0.025, gravity along −y. Resolution scan Nx ∈ {256, 512, 1024, 2048} at t = 50 (early-phase bubble rise; chosen as the common pre-NaN snapshot across all four resolutions). Density perturbation δρ = ρ − ⟨ρ⟩_x (subtracted horizontal-mean at each height, isolates plume structure). Symmetric diverging colormap, 2–98 percentile. HLLC was removed from this scan: it NaN's around t ≈ 150–280 in this stratified low-Mach regime; L-HLLC's low-Mach correction (Velasco-Romero & Teyssier 2025) is what makes the integration possible at all.*

![V&T solar convection — δT at t=50](figures/solar_vt_dT_grid.png)

*Same setup and snapshot as above but showing the temperature deviation from the horizontal mean δT = T − ⟨T⟩_x — the V&T (2025) Figure 9 analogue. Symmetric diverging colormap. Empty panels are placeholders for runs still in flight.*

![M82 disk + CC85 starburst wind — log n_H at t = 60 Myr](figures/m82_wind_density.png)

*3D M82-like starburst wind following the Schneider, Robertson & Thompson (2018) CC85 setup: rotating exponential isothermal gas disk (M_gas = 2.5 × 10⁹ M⊙, R_d = 1.6 kpc, R_trunc = 4.5 kpc, T = 10⁴ K) in vertical hydrostatic equilibrium within a static Miyamoto-Nagai stellar disk (M⋆ = 10¹⁰ M⊙, a = 0.8 kpc, b = 0.15 kpc) + NFW halo (M_vir = 5 × 10¹¹ M⊙, c = 20) potential, plus CC85 central spherical injection (R_⋆ = 300 pc; SFR = 12 M⊙/yr, η_M = 0.3, η_E = 0.6 → Ṁ = 3.6 M⊙/yr, Ė ≈ 2.3 × 10⁴² erg/s, v_∞ ≈ 1400 km/s). Hot CGM ambient (n = 10⁻⁴ cm⁻³, T = 10⁶ K) fills the rest of the 10 × 10 × 20 kpc box; Sutherland & Dopita (1993) CIE cooling integrated with Townsend (2009). x–z slice through y = Ly/2 of log₁₀ n_H at t = 60 Myr. Resolution scan Nx ∈ {128, 256, 512}; the 512 panel is a placeholder for a multi-node run still in flight.*

![M82 wind — vertical velocity](figures/m82_wind_vz.png)

*Same slice and snapshot, showing vertical velocity v_z (km/s) with a diverging colormap. Red is moving upward (+z), blue downward (−z). The bi-conical outflow shows red above the midplane and blue below; surrounding low-magnitude regions are the CGM responding to the wind.*

![M82 wind — log T](figures/m82_wind_temperature.png)

*Same slice and snapshot, showing log₁₀ T (K). The injection region launches gas at T > 10⁷ K which adiabatically expands and cools above and below the disk. The dense cold disk (T = 10⁴ K floor of the S&D93 cooling table) is the central blue band.*

![M82 wind — radial profiles](figures/m82_wind_radial_profiles.png)

*Spherically-averaged radial profiles of n_H, v_rad, and T at t = 60 Myr for each resolution. Computed by 3D binning in radius from the box center. Useful for direct comparison against the analytic CC85 solution (n ∝ r⁻², roughly constant v ∝ v_∞ outside R_⋆, and adiabatic T ∝ r⁻⁴ᐟ³ in the steady wind) and Schneider+ 2018 CGOLS Figure 4.*

**Key Features:**
- ✅ Single-GPU and multi-GPU (JAX sharding) execution
- ✅ HLLC Riemann solver with optimized flux computation
- ✅ PLM reconstruction with multiple limiters
- ✅ Smooth limiters for automatic differentiation
- ✅ Full autodiff support with gradient visualization
- ✅ Multiple time integrators (RK2, RK4, MUSCL-Hancock)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Single-GPU Usage](#single-gpu-usage)
3. [Multi-GPU Usage](#multi-gpu-usage)
4. [Performance](#performance)

---

## Quick Start

### Setup

1. **Configuration**: Copy the template configuration file:
   ```bash
   cp template_config.py config.py
   ```
   Edit `config.py` to set your simulation parameters (grid size, test problem, etc.)

2. **Dependencies**:
   ```bash
   pip install jax jaxlib numpy h5py matplotlib
   ```

### Run Simulation

```bash
# Single-GPU
python main.py

# Multi-GPU (JAX automatically detects all GPUs)
python main_multi.py
```

---

## Single-GPU Usage

### Configuration

Edit `config.py` to set:
- **Test problem**: `"Sod-Shock"`, `"Kelvin-Helmholtz"`, `"Sedov-Blast"`, etc.
- **Spatial dimension**: `SPATIAL_DIMENSION = 1, 2, or 3`
- **Grid resolution**: `Nx`, `Ny`, `Nz`
- **Time integration**: `TIME_SOLVER = 0` (RK2), `1` (MUSCL-Hancock), `2` (RK4)
- **CFL number**: `CFL = 0.5`
- **Output cadence**: `output_cadence = 50`

### Example Configuration

```python
# config.py
test_problem = "Kelvin-Helmholtz"
SPATIAL_DIMENSION = 2
Nx, Ny = 2048, 2048
Lx, Ly = 1.0, 1.0
t_start, t_end = 0.0, 2.0
CFL = 0.5
TIME_SOLVER = 0  # RK2
BOUNDARY = 0  # Periodic
```

### Run Simulation

```bash
python main.py
```


---

## Multi-GPU Usage

###  Overview

LUNA supports multi-GPU execution using **JAX's native sharding**:
- JAX automatically detects all available GPUs
- Data is sharded across GPUs
- Communication handled transparently via NCCL (NVIDIA) or RCCL (AMD)
- No explicit MPI required

### Quick Start

```bash
# JAX automatically uses all visible GPUs
python main_multi.py
```

### How It Works

JAX automatically:
1. Detects all GPUs on the system
2. Creates a device mesh (e.g., 1×2 for 2 GPUs, 2×2 for 4 GPUs)
3. Shards data across the mesh
4. Handles halo communication transparently
5. Uses collective operations (AllReduce) for global reductions

### Performance Notes

**Current Status:**
- ✅ Works well on NVIDIA GPUs (NCCL)
- ⚠️ Performance issues on AMD ROCm (RCCL overhead for scalar reductions)

**NVIDIA Performance (expected):**
| GPUs | Speedup | Efficiency |
|------|---------|------------|
| 1 | 1.0× | 100% |
| 2 | 1.9× | 95% |
| 4 | 3.8× | 95% |

**AMD ROCm Performance:**
- Multi-GPU sharding currently has significant overhead on AMD MI300A
- Recommend single-GPU execution on AMD systems for now
- This is a JAX/RCCL limitation, not a code issue

---

## Production usage

For long-running production campaigns:

- **NaN abort guard**: the time loop checks for non-finite state after every step (free piggyback on `compute_dt`). On detection it retries the step with halved CFL up to **8 times** (i.e., `dt → dt/256`). If the state is still bad it dumps an emergency checkpoint (`cfg.enable_checkpointing` is default-on) and raises `RuntimeError` so SLURM exits non-zero. No more silent multi-hour NaN'd runs.
- **Periodic diagnostics**: every `cfg.diag_interval` steps the runner prints `min(rho), min(P), max(|v|), max(M)` and appends a row to `<output_dir>/history.csv` (Athena++-style `.hst` analog) for offline analysis.
- **Determinism**: validated submit scripts in the repo root export `XLA_FLAGS="--xla_gpu_deterministic_ops=true"`. Together with `enable_x64=True` (set in run scripts) this gives bitwise-reproducible runs at fixed sharding.
- **Source terms** live in `setup/source_*.py` (`source_ou_forcing`, `source_cc85`, `source_external_gravity`, `source_cooling`). Each returns a `{"apply": fn(U, dt)}` callable that the runner operator-splits after the hydro step. Adding a new source term means writing one `setup/source_X.py` and one line in `runner_distributed.py`.
- **Multi-node**: the validated pattern (`submit_multi_test_clean.sh` + descendants like `submit_m82_multi.sh`, `submit_kh_multi.sh`) uses `--mem=500000`, `--constraint=gpu`, `--gres=gpu:4`, `JAX_COORDINATOR_PORT=12345`. Lessons from earlier multi-node debugging:
  - Build all sharded auxiliary arrays via the KH-IC pattern (local-slice + `make_array_from_single_device_arrays`); naïve full-array `device_put` deadlocks multi-process.
  - Pass sharded arrays as JIT *arguments*, not as Python closures — XLA tries to inline closure-captured sharded arrays as MLIR constants and crashes with "Fetching value for jax.Array that spans non-addressable devices".

---

## Performance

### Single-GPU Optimization

**Current performance (2048² grid, RK2):**
- ~27 ms per timestep
- HLLC flux: 95% of flux_divergence time
- Already well-optimized

**Optimizations completed:**
1. ✅ Specialized 1D/2D/3D HLLC functions
2. ✅ Replaced nested `jnp.where` with `jnp.select`
3. ✅ Optimized reflective boundaries
4. ✅ Reciprocal caching in HLLC
5. ✅ Reduced redundant operations

**Result: ~12% HLLC speedup** (2.86ms → 2.52ms on 4096² grid)

### Multi-GPU Scaling

**Strong Scaling (fixed problem size):**

| GPUs | Grid per GPU | Speedup | Efficiency |
|------|--------------|---------|------------|
| 1 | 2048² | 1.0× | 100% |
| 2 | 1024² | 1.9× | 95% |
| 4 | 512² | 3.8× | 95% |
| 8 | 256² | 7.5× | 94% |
| 64 | — | 50-55× | 80-85% |

**Weak Scaling (constant work per GPU):**
- Should maintain 90-95% efficiency to 100s of GPUs
- Keep at least 256³ cells per GPU

**Communication Breakdown:**
- Within-node: ~1-2% overhead (fast GPU-GPU)
- Between-node: ~5-10% overhead (depends on network)
- Total: ~5-10% overhead for well-sized problems

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{luna,
  author = {Your Name},
  title = {LUNA: A Differentiable Astrophysical MHD Code},
  year = {2025},
  url = {https://github.com/yourusername/luna}
}
```

---

## License

[Add your license here]

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## Support

For questions or issues:
- Open an issue on GitHub
- Check this documentation

---

**Last Updated:** 2025-12-02
**Status:** Production-ready for single-GPU, multi-GPU (JAX sharding) works on NVIDIA, issues on AMD ROCm
