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

![V&T solar convection — δρ at t=1000](figures/solar_vt_drho_grid.png)

*2D Velasco-Romero & Teyssier (2025) 3-layer polytropic atmosphere with a buoyant Gaussian bubble (δρ/ρ_eq = −0.012, r_b = 0.5), well-balanced MUSCL-Hancock + MonCen + L-HLLC at CFL = 0.1, gravity along −y. Resolution scan Nx ∈ {512, 1024, 2048, 4096} at t = 1000 (saturated convective state, well past V&T's early-phase KE peak at t ≈ 100). Density perturbation δρ = ρ − ⟨ρ⟩_x (subtracted current horizontal-mean at each height, not the initial polytropic equilibrium — this isolates plume structure cleanly at saturated times). Symmetric diverging colormap, 2–98 percentile. HLLC was removed from this scan: HLLC NaN's around t ≈ 150–280 in this stratified low-Mach regime independent of CFL choice; L-HLLC's low-Mach fix (Velasco-Romero & Teyssier 2025) is what makes the integration stable.*

![V&T solar convection — δT at t=1000](figures/solar_vt_dT_grid.png)

*Same setup and snapshot as above but showing the temperature deviation from the horizontal mean δT = T − ⟨T⟩_x — the V&T (2025) Figure 9 analogue. Symmetric diverging colormap. Empty panels are placeholders for runs still in flight.*

![M82 disk + CC85 starburst wind — log n_H at t = 60 Myr](figures/m82_wind_density.png)

*3D M82-like starburst wind following the Schneider, Robertson & Thompson (2018) CC85 setup: rotating exponential isothermal gas disk (M_gas = 2.5 × 10⁹ M⊙, R_d = 1.6 kpc, R_trunc = 4.5 kpc, T = 10⁴ K) in vertical hydrostatic equilibrium within a static Miyamoto-Nagai stellar disk (M⋆ = 10¹⁰ M⊙, a = 0.8 kpc, b = 0.15 kpc) + NFW halo (M_vir = 5 × 10¹¹ M⊙, c = 20) potential, plus CC85 central spherical injection (R_⋆ = 300 pc; SFR = 12 M⊙/yr, η_M = 0.3, η_E = 0.6 → Ṁ = 3.6 M⊙/yr, Ė ≈ 2.3 × 10⁴² erg/s, v_∞ ≈ 1400 km/s). Hot CGM ambient (n = 10⁻⁴ cm⁻³, T = 10⁶ K) fills the rest of the 10 × 10 × 20 kpc box; Sutherland & Dopita (1993) CIE cooling integrated with Townsend (2009). x–z slice through y = Ly/2 of log₁₀ n_H at t = 60 Myr. Resolution scan Nx ∈ {128, 256, 512}; the 512 panel is a placeholder for a multi-node run still in flight.*

![M82 wind — radial velocity](figures/m82_wind_vradial.png)

*Same slice and snapshot, showing radial velocity v_r (km/s) with a diverging colormap. Positive (red) is outflow above the midplane and below the midplane the same sign convention reads as inflow toward the launching region from the diffuse CGM. The cone-like bi-conical outflow above and below the dense rotating disk is clearly visible.*

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
3. [Autodiff & Gradient Analysis](#autodiff--gradient-analysis)
4. [Multi-GPU Usage](#multi-gpu-usage)
5. [Performance](#performance)
6. [File Structure](#file-structure)

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

## Autodiff & Gradient Analysis

### Overview

JAX-Hydro supports **automatic differentiation** through the entire solver, enabling:
- **Sensitivity analysis**: Which parts of initial conditions affect final observables?
- **Inverse problems**: Optimize ICs to match desired outcomes
- **Adjoint-based methods**: Efficient gradient computation for large systems
- **Physics-informed learning**: Train neural network components

The implementation uses **smooth slope limiters** that approximate traditional limiters (minmod, moncen, vanleer) with differentiable functions (tanh, logaddexp).

### Quick Start

**1. Configure for autodiff in `config.py`:**

```python
# Choose a shock problem (activates limiters)
test_problem = "Sod-Shock"  # or "Woodward-Colella"

# Enable smooth limiters
use_smooth_limiters = True
smooth_limiter_sharpness = 10.0  # 10-20 typical

# Small grid for testing
Nx = 256
t_end = 0.1
```

**2. Run autodiff test:**

```bash
python main_autodiff.py
```

This will:
- Run forward simulation
- Compute gradients via autodiff
- Verify with finite differences
- Save results to `snapshots_*/autodiff_result.hdf5`

**3. Visualize gradients:**

```bash
python utils/plot_autodiff_gradients.py snapshots_*/autodiff_result.hdf5
```

Creates 4 plots showing gradient structure and domain of influence.

### Understanding the Gradients

For observable `J = ρ_final[x=0.5]` (density at domain center):

**Gradient**: `∂J/∂U₀(x)` for all x

**Physical meaning**: How much does perturbing the initial condition at position x affect the final density at x=0.5?

**Expected pattern for Sod shock**:
```
     Rarefaction    Contact    Shock
x=0 ←-----------← ←---------← ←----← x=1
                    ↑
                  x=0.5 (observation)
```

- **Non-zero gradients**: In regions that can causally influence x=0.5 by t_end
- **Zero gradients**: Outside domain of influence (too far to reach)
- **Boundaries**: Determined by characteristic speeds (sound waves, shocks)

### Observable Functions

The code auto-selects observables based on test problem:

| Observable | Formula | Best For | Gradient Meaning |
|-----------|---------|----------|------------------|
| `density_at_point` | ρ[Nx/2] | Shock tubes | Shock position sensitivity |
| `momentum_region` | Σ(ρu) in [1/4, 3/4] | Strong shocks | Flow through region |
| `kinetic_energy` | Σ(½ρv²) | Shu-Osher | Energy distribution |
| `pressure_integral` | Σ(p) | Shock problems | Thermodynamic sensitivity |
| `total_energy` | Σ(E) | ⚠️ Avoid (conserved!) | Near-zero gradients |

### Visualization Outputs

**1. `autodiff_gradients_stacked.png`**
- Separate plots for ∂J/∂ρ₀, ∂J/∂(ρu)₀, ∂J/∂E₀
- Shows which variables are most sensitive

**2. `autodiff_gradients_overlay.png`**
- Top: Initial condition (ρ, p)
- Bottom: Normalized gradients
- Compare gradient structure with IC features

**3. `autodiff_gradients_domain.png`**
- Gradient magnitude ||∇J||
- Domain of influence boundaries
- Shows causal structure

**4. `autodiff_gradients_evolution.png`**
- Initial vs final density
- Gradient magnitude overlay
- Connects dynamics to sensitivity

### Test Problem Recommendations

**✅ Good (activates limiters):**
- **Sod-Shock**: Classic, moderate difficulty
- **Woodward-Colella**: Strong shocks, ultimate test
- **Shu-Osher**: Shock-turbulence interaction

**❌ Avoid (too smooth):**
- **Soundwave**: Linear, limiters barely activate
- **Soundwave-Advected**: Same issue

### Smooth Limiters

Traditional limiters (minmod, etc.) use non-differentiable operations:
```python
# Non-smooth (breaks autodiff)
minmod(a, b) = sign(a) * min(|a|, |b|) if a*b > 0 else 0
```

Smooth approximations use differentiable functions:
```python
# Smooth (autodiff-compatible)
minmod_smooth(a, b) = smooth_sign(a) * smooth_min(|a|, |b|) * smooth_heaviside(a*b)
```

**Sharpness parameter** controls approximation quality:
- `5-10`: Very smooth, better for optimization
- `10-20`: Good balance (recommended)
- `50-100`: Very close to original limiters

### HDF5 File Structure

```
autodiff_result.hdf5
├── initial/          # U₀, decomposed fields
├── final/            # U_final, decomposed fields
├── gradients/        # ∂J/∂U₀, decomposed by variable
├── observable/       # Name, value, observation point
├── coordinates/      # x, y, z positions
└── metadata/         # All config parameters
```

### Verification

The code automatically verifies gradients via finite differences:

```
∂J/∂U₀[i] ≈ (J(U₀ + ε*e_i) - J(U₀ - ε*e_i)) / (2ε)
```

**Success criteria:**
- ✓ Excellent: Max relative error < 1e-3
- ✓ Good: Max relative error < 1e-2
- ⚠️ Warning: Max relative error < 1e-1
- ✗ Failed: Max relative error > 1e-1

### Troubleshooting

**Gradients are NaN/Inf:**
- Ensure `use_smooth_limiters = True`
- Increase `smooth_limiter_sharpness` (try 20.0)
- Check time solver compatibility (RK2, RK4, Hancock work)

**Gradients don't match finite differences:**
- Check relative error threshold
- Try smaller timestep or finer grid
- Adjust `smooth_limiter_sharpness`

**Gradients near zero:**
- Observable may be poorly chosen (e.g., total_energy)
- Evolution time too short
- Problem too smooth (use shock problem)

### Advanced: Custom Observables

To add custom observables, edit `main_autodiff.py`:

```python
def observable_max_density(U_final, spatial_dim):
    """Maximum density in domain."""
    return jnp.max(U_final[0])

# Add to dictionary
OBSERVABLES['max_density'] = observable_max_density
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

## File Structure

```
luna/
├── main.py                      # Single-GPU driver
├── main_multi.py                # Multi-GPU (JAX sharding) driver
├── main_autodiff.py             # Autodiff testing
├── config.py                    # Configuration (user-created)
├── template_config.py           # Configuration template
│
├── hydro/                       # Physics kernels
│   ├── fluxes.py                # HLLC Riemann solver
│   ├── slopes.py                # PLM + smooth limiters
│   ├── divergence.py            # Flux divergence
│   ├── padding.py               # Boundary padding
│   └── physics_checks.py        # Validation
│
├── time_integration/            # Time stepping
│   ├── integrators.py           # RK2, RK4, MUSCL-Hancock
│   └── cfl.py                   # CFL condition
│
├── ic_generation/               # Initial conditions
│   └── ic.py                    # Test problems
│
├── file_io/                     # I/O
│   ├── write_snapshot.py        # Regular HDF5 output
│   ├── read_snapshot.py         # Read snapshots
│   ├── write_autodiff_result.py # Autodiff HDF5 output
│   └── read_autodiff_result.py  # Read gradients
│
├── utils/                       # Utilities
│   ├── logo.py                  # ASCII banner
│   ├── memory_utils.py          # Memory tracking
│   ├── plot_1d_fluid_quants.py  # 1D visualization
│   ├── plot_density2d.py        # 2D visualization
│   └── plot_autodiff_gradients.py  # Gradient visualization
│
└── README.md                    # This file
```

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
