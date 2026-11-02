# Test plan

State of the suite on and what to add. Claims cite the code; where this file and the
code disagree, the code wins.

## Where tests run

| | files | |
|---|---|---|
| CI, every push | 35 of 183 (fast tier, 117 s) | `.github/workflows/tests.yml` |
| nightly, viper, full tier | 183 | `submit/examples/nightly_tests.sbatch`; 59 files have no measured cost, so it has likely not run recently |
| never in CI | gravity 0/7, chemistry 0/13, autodiff 0/9, sinks 0/3 | GPU or nightly only |

Multi-device tests skip silently with too few devices (`test_isolated_gravity.py:62`,
`test_pm_blocks.py:16`); `test_poisson_multigrid_multigpu.py:60` runs unsharded on one device.

## Standards for an accuracy test

- Report L1, L2 and, for smooth problems, L∞ against an exact or reference solution, at a
  resolution ladder; bound each norm and its fitted order. Helpers: `scripts/validation/metrics.py`
  (`l1_error`, `l2_error`, `convergence_slope`), to be shared with `tests/`.
- Expected orders for a 2nd-order FV scheme, so bounds mean something:

  | solution | L1 | L2 | L∞ |
  |---|---|---|---|
  | smooth | 2 | 2 | 1-1.5 at extrema (minmod clipping) |
  | shock | 1 | 1/2 | none |
  | contact | 2/3 | 1/3 | none |

- Conservation in every run: `tests/_helpers.evolve` and `_run_short` check mass, energy (and momentum
  when periodic) in closed boxes; open boxes check the exact flux budget (`check_tube_budget`).
  Coverage (audit 2026-10-14): own-loop and
  staged runs check too (odd-even, CFL noise, AMR composites M/E/P to 1e-13); open boxes whose faces see
  the ambient at rest (Sedov) to round-off; diode faces against `boundary_mass_budget` (well-balanced,
  2e-16). With sources the budget includes them: E + rho phi in a static potential (A6, order 2.0 /
  3.0 MC; A13 already), Layzer-Irvine (top-hat), gas + radiation energy and momentum, photon budgets.
  Not checked, with the reason in the file: transmissive edges that waves reach (LW Gresho, carbuncle,
  Titarev-Toro), M1 beams leaving through zero-gradient faces, chemistry energy (no rate accumulator).
- Self-gravity energy E_kin + E_th + W (W = 1/2 sum rho phi dV, phi the 7-point FFT inverse, mean
  removed; `_helpers.self_gravity_energies`) is truncation, not round-off: the kick's work rho v . g with
  the central-difference g is not the hydro's discrete -phi div(rho v). Jeans (A7, AMP 1e-3, hllc-minmod,
  |dE| / |W0| over a period / e-fold): stable 0.63, 0.21, 0.051 at N = 16, 32, 64 (order 1.81), unstable
  0.35, 0.088, 0.021 (2.02); halving CFL down to 0.1 moves it < 25 % (spatial). Evrard (A8, |dE| / |E0| at
  t = 3): hllc-minmod 3.0, 1.55, 1.01, 0.58 at 32-256; hllc-mc 1.08, 0.57, 0.24 at 64-256 (order ~0.6-1.2,
  the accretion shock). Doubling G in the run fails the Jeans bound (0.76 at 32). Momentum: sum rho D L^-1
  rho = 0 for the exact inverse, so periodic momentum drifts by the solve's tolerance only: round-off
  while the state is symmetric (Evrard minmod 5e-15), 4e-8 after the MC bounce (1.6e-12 at warm_tol 1e-9),
  Jeans 1e-12 .. 1.2e-10 (8 fixed V-cycles), AMR + gravity 2.7e-9 from a symmetric IC (the smoother breaks
  the mirror symmetry; not verified).
- Self-similar and radial problems also report: wave position R_s(t) against exact, the similarity
  exponent fitted from R_s(t), the symmetry error (angular spread at fixed r over the radial mean: on a
  Cartesian grid this is the grid anisotropy), entropy error, positivity minima (rho, p, e_int), and the
  similarity collapse (profiles at several times rescaled onto one curve).
- A teeth check: a deliberately broken run must fail the bound.
- A known bug is a strict xfail with its reason, so a fix turns it red.
- Test FP32 as well as FP64: FP32 is the production default (`src/luna/template_config.py:45`).

## Strong today

- Riemann fluxes vs the exact Godunov flux, Toro 1-5, to 1e-14 (`test_hydro_accuracy_riemann.py:76`).
- Shock tubes and smooth problems with fitted orders, 1D-3D; Sedov, Noh, isentropic vortex,
  Taylor-Green vs references; Galilean invariance; Pallas == XLA.
- Sharded == one device to round-off, hydro and every gravity backend.
- Isolated gravity vs the analytic sphere and multipoles; adjoints vs FD and pseudo-inverse;
  exact CIC deposit; linear growth vs D(a); Zel'dovich vs analytic.

## Gaps

- No FP32 accuracy test. All accuracy runs use minmod + HLLC + FP64 (`tests/_helpers.py:125`).
- No accuracy test for recon32, the fused path, ORDER=0; van Leer only in LeBlanc (MC in LeBlanc,
  Titarev-Toro, Shu-Osher, Woodward-Colella; `BOUNDARY=2` in Woodward-Colella).
- Most accuracy tests bound L1 only.
- Bounds are ~1.3x measured: regression guards, not quality statements; low-Mach bounds ~0.6.
- Poisson: one resolution, rel < 1e-2, no order; Neumann has no analytic test.
- No PM (particle) momentum conservation, no PM gather-force test (gas: Evrard, Jeans energy and momentum)
  (promised in `test_pm_blocks.py:3`).
- Hygiene: `test_regressions.py:103` sets x64 globally and never restores it;
  `test_floor_conservation.py:129` uses ORDER=2, which `validate()` refuses, and the AMUSE
  stepper never validates, so it silently runs PLM. `docs/GRAVITY.md:123-126` table is stale.

- Sinks (open, Phase C / W6): `docs/specs/sinks/test_coupling_sink_accretion.py` and
  `test_coupling_sink_formation.py` are the specification for reviving sink formation and accretion
  (SEREN / Bate-Bonnell-Price criteria; exact accreted-gas moments). They drive the old
  `sources.sinks.setup_sinks` API removed in 4848abe and are out of the suite until the `Sinks` class
  (species/sinks.py) gets the setup mapping and the running accumulators (macc, sv, sv2, scs, sL).
  To do: port them to `Sinks`, move them back to tests/, add a Bondi accretion test.

## Open bugs (strict xfails)

- HLLC carbuncle at grid-aligned strong shocks (`test_hydro_hard_carbuncle.py:119`).
- Cold ambient drains through diode faces, pressure < 0 near 2.85 t_sc (`test_hydro_hard_well_balanced.py:126`).
- Registered Lane-Emden IC has rho = 0 at the surface (`test_hydro_hard_well_balanced.py:164`).
- Boss-Bodenheimer scripts need `cfg.units`; CC85 injection runs the step in FP32 (`tests/_run_script_cases.py`).
- Well-balance drift (`tests/test_hydro_hard_well_balanced.py`): the n = 1 polytrope's central density
  drifts 23% over 10 crossings under the RAMSES dt rule (CFL 0.8), 20% under the old per-axis 0.4;
  the bound failed unnoticed from 2026-10-05 (slow tier, no nightly) and was re-measured.
  To fix: a well-balanced gravity source (reconstruction about the hydrostatic profile).
- Isothermal mode, MUSCL-Hancock predictor uses the adiabatic c^2 = gamma P / rho
  (`src/luna/hydro/jacobian.py:11`, `:30`) before the projection (`src/luna/setup/assemble.py:192`):
  a numpy toy (not LUNA) drops to first order on a smooth isothermal wave (1.10 / 1.02 / 1.00 vs 1.9
  with an isothermal predictor); shocks converge at the isothermal-HLL rate. To measure in LUNA first
  (smooth isothermal wave ladder through the full stage pipeline), then decide.

## Fixed

- Evrard collapse, HLLC + MC at 32^3: p < 0 at the bounce (t = 1.10), then NaN dt. Cured by the C8 face fix:
  the run reaches t = 3. Energy is conserved only with `gravity_energy="flux"`: |dE/E0| <= 4.9e-5 (32) and
  7.7e-5 (64) to t = 3, the solve tolerance, limiter idle (`test_gravity_energy_flux.py`); the kick-mode budget
  row (`test_gravity_evrard_energy.py`) still loses 2.39 (minmod 3.02): kick mode does not conserve, by design.

- Audit C8 (HLLC NaN face state, `docs/history/AUDIT_2026-09-26.md:70`): non-physical predicted faces take
  the first-order state. Found by LA-14379 RCVCR and streams with MC (mirror error 8.7e-3, 2.9e-2); now
  mirror symmetric to 1e-13, streams MC L1 rho at N = 1024 8.4e-3 -> 2.9e-3 (order 0.46 -> 0.87), CPU FP64.

- Koyama-Inutsuka cooling was 1000x too strong below ~1.5e4 K: the misprinted
  coefficient 14.0 sqrt(T) in `src/luna/sources/cooling.py` is now 1.4e-2; with it the published
  two-phase states are equilibria (n Lambda/Gamma = 0.9986 at n = 2.93, T = 756 K, Koyama & Inutsuka
  2004; 1.0199 at n = 2, T = 1500 K, Jennings & Li 2021), `tests/test_cooling_tables.py`. Unit-free:
  Lambda and Gamma share gamma_ki; the runtime path (`cooling.py:238-253`) has no compensating factor.
  Affected `koyama_inutsuka` and `combined` only (default `sd93_solar` not). Note: KI's fit is per
  n_H^2 (neutral gas); LUNA's default `cooling_ne_per_nH = 1.189` makes the KI part 19 % stronger, so
  KI-faithful runs set `cooling_ne_per_nH = 1`.
- The isothermal projection rounded cs2 and gamma - 1 to float32 in every run:
  `src/luna/sources/eos_projection.py` casts to the state's dtype; fp64 now exact to 1.1e-15,
  `tests/test_cooling_tables.py`.

- Reflecting walls leaked (): BOUNDARY=2 ran the narrow path, its wall face a raw ghost
  against the reconstructed state; |dM|/M up to 4e-3. Now two mirrored ghosts on the wide path,
  `docs/HYDRO_KERNEL_PATHS.md`; `tests/test_hydro_wall_conservation.py` checks mass, energy and
  momentum to round-off and sharded == one device. Lesson: every test checks what it conserves.

- The M82 external gravity built its coordinates in float32 in every run (fixed): fp64 runs now use
  float64 (acceleration error 7.6e-6 relative before), `tests/test_external_potential.py`.

## Broken test problems

- `Rotate-Slotted-Disk`: the rotating slotted cylinder is a linear-advection
  benchmark, now run with `src/luna/hydro/advection.py` as in Palafoutas, Velasco Romero &
  Teyssier 2024 (sec. 4.4); `tests/test_advection_slotted_cylinder.py`. The Euler entry refuses:
  no stable Euler setup in a walled box (pressure jump, wall drift, shielded-vortex instability,
  confirmed with euler_3d); details in `src/luna/ic_generation/rotation_slotted_disk.py`.
  A stable Euler rotation needs ghosts held at the exact flow [infra, with DMR].
- `Double-Mach-Reflection` (renamed): the old IC was a planar shock tube, now
  `Planar-Riemann-2D` (`src/luna/ic_generation/planar_riemann_2d.py`, tested exact on a strip);
  the DMR entry refuses until per-face, time-dependent boundaries exist.

## Duplicated support code (audit, each verified in the code)

In order of risk. Done: the exact Riemann solver, one copy in `src/luna/hydro/exact_riemann.py`.

Tests that did not test the production path:
1. `tests/_helpers.py:_make_step_fns` now wraps `create_solver_distributed` (bitwise equal on the
   CPU, 1D-3D, all limiters; on a GPU the tests now run the Pallas kernel). 8 files still build
   the step positionally; halo, seam and Pallas tests do so on purpose.
2. `tests/test_sedov_gradients.py` builds U0(E0, rho0) with the registered `Sedov-Blast-3D` (now
   with `Params`, `r_blast` 1.24 cells) and differentiates `build_simulate`; `sedov_ad.py` removed.
   Finding: the default 6-cell top-hat is half the final shock radius at N = 32 and gives
   dlnR/dlnE = 0.243; a point blast through the same path gives 0.2009 (exact 0.2).

One truth per exact solution:
3. Zel'dovich: `zeldovich.exact()` serves the IC, `tests/test_cosmo.py` (no more hardcoded
   k = 2π) and `scripts/cosmo/run_pancake.py`.
4. Sedov: the Kamm closed form is `src/luna/hydro/exact_sedov.py` (`xi0`, `solution`); the ODE
   copy is gone (it agreed to 2e-5 at γ = 5/3, 4e-4 at 1.4); `sedov3d_plot.py` takes ξ0 = 1.1517
   from it instead of 1.15; alpha vs the tabulated 0.8511 / 0.4936 is now a test.
   Done 2026-10-29 (W1, `tests/test_reference_solvers.py`, CPU-8 job 31663813, 27 passed, 5.5 s): alpha was
   short by 1e-4 (j = 3) to 1.4e-3 (j = 1), the lam grid stopped at s_min^|a2|; now regular quadrature, KT 2007
   Tables 1-5 to printed digits. Same file: Riemann vs Toro Table 4.2 (vacuum now refused), Guderley vs RKB
   Table 1, Noh vs Noh 1987.
   Also (jobs 31664644, 31664761, CPU-8): `tests/test_reference_analytic.py` (Jeans operator eig/expm, Euler
   Jacobian eigenvectors of the linear waves, Zel'dovich round-trip/mass/growth ODE; exact() now refuses non-EdS
   and its inversion is bracket-safeguarded: plain Newton missed x by 0.1 at a/a_c >= 0.99) and
   `tests/test_reference_cj.py` (CJ vs minimum Hugoniot speed, ZND endpoint -> CJ as sqrt(delta)).
   Microphysics 99b9b47 references now asserted (jobs 31665351, 31665738): `tests/test_reference_microphysics.py`
   (Helmholtz 50 states, CJ 10 states within the cj_det.H Newton tol 1e-8) and `tests/test_reference_burn.py`
   (He burn_cell); full Timmes table where present, else luna-reference-solutions/tables/helmholtz/helm_table_subset.npz (tools/make_helm_subset.py,
   bitwise equal on 120000 box states, 50 reference states, 10 CJ solves).
5. Jeans: `jeans3d.sigma_squared(cfg)` serves the IC, the run script and the plot; the mode
   amplitude and σ fit live once in the run script. Still open: no test fits σ on a run.
6. MUSIC: one `growing_mode_ic(path, gas, bulk_velocity)` in `tests/_music_growing_mode.py`.
7. Double Sod tube: `exact_riemann.double_tube()` holds the geometry; each test keeps its own
   cell averaging (oblique 3D vs boosted 1D) on purpose.

Cleanup:
8. Same runs twice: KH growth (`accuracy_kh`, `hard_galilean` b=0), Gresho (`accuracy_smooth`,
   `hard_low_mach`).
9. Two sound-wave ladders; `soundwave.py` fixes amp = 1e-4, so neither uses the registered IC as is.
10. Order fits written 9 times (`metrics.convergence_slope` unused in tests); `_x64` fixture 14
    times, x64 switched on at import in 142 test files.

## Group A progress (catalogue `reports/Hydro test suite literature catalogue.md`)

- A1 done: `tests/test_hydro_accuracy_linear_waves.py`. Sound (both ways), entropy, shear,
  1D (32-512) and oblique 2D (32-256, wave vector at atan 2), HLLC/HLL minmod and HLLC MC; L1, L2,
  Linf of (U - U0)/eps and their orders, left/right equality (to O(eps): the left wave swaps
  compression and rarefaction), conservation, a short run must fail. Minmod L1 orders ~1.85 (1D),
  ~1.6 (2D oblique). Athena++'s gate (`hydro_linwave.py`) is err64/err32 <= 0.325 (HLLE 0.33, 0.34), order
  >= ~1.6, not the 0.25 its comment says; each scheme is bounded by its own measurement. Entropy and shear ride on v0 = 0.5: at
  v0 = c both solvers reduce to upwinding and the contact resolution goes untested. It covers the
  sound ladders in `test_hydro_accuracy_smooth.py` and `test_regressions.py` (duplicate item 9).
  The error is the part of U - U0 odd in EPS, ((U+ - U0+) - (U- - U0-)) / 2: the even part is a constant
  ~4e-6 EPS (the wave's own nonlinearity), 17 % of the 1D MC error at 512 before. Bounds re-measured on
  this code (the old 'measured' comments predate later commits: L1 at 32 is 0.0225, not 0.0302).
  Against AthenaK on the same wave (rho 1, P 0.6, gamma 5/3, EPS 1e-6, one period), its metric (RMS over
  variables of L1 / EPS), 1D, N = 32..1024: LUNA minmod 3.1e-2 .. 4.5e-5, MC 1.1e-2 .. 1.0e-5; AthenaK
  PLM + RK2 7.4e-2 .. 8.2e-5 (1.8-2.4x LUNA minmod); PPMX + RK4 fourth order (3.9e-4, 2.5e-5 at 32, 64)
  until its EPS^2 floor (5.5e-6) by 128.
- A9 done: `tests/test_hydro_hard_odd_even.py`, LW sec. 4.9 (WC blast, 800 x 10, reflective x,
  periodic y; on 1024 x 16, BOUNDARY=2, equivalent for y-uniform data). v_y = 0, rows equal, and
  bitwise equal to the 1D run at equal dt, HLLC and HLL (LW: WAFT 5e-4, VH1 0.1).
  Against AthenaK from the same initial data (pgen luna_ic), N x 16 strips, N = 256..2048, GPU FP64,
  `luna-reference-solutions/athenak/woodward_colella/a9_*_16384.npz`, `lit_a09/athenak.md`: all three checks are exactly 0 for every LUNA scheme
  and for AthenaK PLM + RK2, PPMX + RK4 and PPM4 + RK4 (each against its own 1D run; periodic y at 1024 too).
  L1(rho) of the strip (ours, the test bounds none) against the LUNA MC 2^16 reference
  (`luna-reference-solutions/luna/woodward_colella/woodward_colella_mc.npz`; AthenaK PPMX 16384 is 0.0019 from it at 1024 cells, AthenaK PLM 16384
  0.0055) and, in brackets, against AthenaK PLM 16384:

  | N | hllc-minmod | hll-minmod | hllc-mc | hll-mc | hllc-vl | AthenaK PLM+RK2 | AthenaK PPMX+RK4 | AthenaK PPM4+RK4 |
  |---|---|---|---|---|---|---|---|---|
  | 256 | 0.185 (0.182) | 0.200 (0.197) | 0.142 (0.139) | 0.158 (0.155) | 0.152 (0.149) | 0.162 (0.159) | 0.110 (0.107) | 0.112 (0.109) |
  | 512 | 0.122 (0.119) | 0.132 (0.129) | 0.0818 (0.0786) | 0.0895 (0.0866) | 0.0911 (0.0878) | 0.0984 (0.0951) | 0.0598 (0.0572) | 0.0676 (0.0647) |
  | 1024 | 0.0776 (0.0725) | 0.0836 (0.0785) | 0.0467 (0.0416) | 0.0504 (0.0453) | 0.0529 (0.0478) | 0.0579 (0.0528) | 0.0324 (0.0277) | 0.0373 (0.0326) |
  | 2048 | 0.0472 (0.0423) | 0.0507 (0.0458) | 0.0272 (0.0223) | 0.0290 (0.0243) | 0.0301 (0.0250) | 0.0330 (0.0279) | 0.0201 (0.0157) | 0.0223 (0.0177) |
  | order 256-2048 | 0.66 (0.70) | 0.66 (0.70) | 0.80 (0.88) | 0.81 (0.90) | 0.78 (0.86) | 0.76 (0.84) | 0.82 (0.94) | 0.78 (0.89) |

  At 1024 AthenaK PPMX is 0.69x LUNA MC (PPM4 0.80x); LUNA MC is 0.81x AthenaK PLM, LUNA van Leer (AthenaK's
  slope) 0.91x; LUNA's default minmod is 1.34x AthenaK PLM. The PLM reference slightly favours LUNA MC (0.79x
  against it). PPMX and PPM4 need `dfloor = pfloor = 1e-8` (no floor fires; mass and energy to 6e-12): at AthenaK's
  default FLT_MIN floors PPMX gives NaN at the blast collision (t 0.0274-0.0276 at 1024) and AthenaK still reports
  a normal end (critic, `critic_p1/critic.md`, jobs 31105754, 31106973).
- A10 done: `tests/test_hydro_hard_implosion.py`, the Hui / LW implosion (LW 2003 sec. 4.7): quadrant (0, 0.3)^2,
  reflecting walls, inside x + y < 0.15 rho 0.125, p 0.14, outside 1, 1, gamma 1.4, CFL 0.8. IC `LW-Implosion`
  (`src/luna/ic_generation/implosion.py`): exact cell averages (the line halves the cells i + j = N/2 - 1: rho
  0.5625, p 0.57), so block averages of a fine IC are the coarse IC; `tools/luna_ic.py` writes it unchanged.
  No paper gives a number (every result is a contour plot); all bounds are ours. Measured (A100 FP64 unless
  stated; jobs 31109197-31110359, `lit_a10/dossier.md`):
  - x <-> y mirror, max |rho - rho^T|, |m_x - m_y^T|, |E - E^T|, is exactly 0 on the GPU at t = 0.045, 0.5, 2.5
    for HLLC/HLL x minmod/MC/van Leer at 64..1024^2; in FP32 (400^2 to 2.5); with recon32 and with the Pallas
    kernel (3-D z-uniform 128 x 128 x 4 slab, minmod, bitwise equal to XLA; and in 2-D at 400^2, which Pallas
    runs for every limiter since c2032fa); on (2,1) and (1,2)
    meshes (bitwise the one-device run, 400^2 to 2.5, 2 GPUs; CPU (2,1)..(4,1),(1,4) at 64^2); and in the
    production loop (`evolve_staged` bitwise `evolve`). Bounded at 0. A 1e-14 seed in one off-diagonal cell is
    1.5e-12 at 0.045 and 8e-4 at 2.5 (400^2), so nothing but 0 is a meaningful bound.
  - Finding: XLA CPU is not bitwise symmetric: 1e-15 .. 1.6e-14 (FP64, 64-128^2), 6e-7 .. 2e-6 (FP32). Cause: FMA
    contraction, a*b + c*d -> fma(a, b, c*d) rounds the x and y terms of a cell differently from its mirror; with
    `--xla_cpu_max_isa=AVX` (no FMA) it is exactly 0 (jobs 31109433, 31109434). The exact tests are a strict xfail
    on CPU; the CPU tier bounds the mirror error at 1e-14 (FP64) / 5e-6 (FP32) instead, where a 1e-12 seed
    (5.3e-13 at 0.045) is still visible but a 1e-14 seed (1.2e-14) is not. Not a solver change.
  - Mass and energy |dM|/M, |dE|/E <= 3.3e-16 everywhere, bounded 1e-15.
  - L1 / L2(rho) at LW's t = 0.045 (pre-focus) against a LUNA HLLC-MC 6400^2 reference
    (`luna-reference-solutions/luna/lw_implosion/lw_implosion_mc.npz`, `tools/implosion_reference.py`, 800^2 block averages; 6400 vs 3200 L1
    8.0e-5): HLLC-MC 2.64e-3, 1.46e-3, 8.20e-4, 4.39e-4 at 100, 200, 400, 800^2, orders L1 0.75-0.86, L2
    0.34-0.41 over the six schemes (shock + contact: L1 2/3..1, L2 1/3..1/2); MC < van Leer < minmod, HLLC < HLL
    (minmod HLL / HLLC 1.4x at 800). Bounded 1.3x, orders -0.1/+0.3. A run stopped at 0.95 t has L1 4.5e-3 at
    400^2 (bound 1.1e-3). The reference itself is off by an estimated ~1.4e-4 in L1 (checker: 1600^2 scores 2.05e-4
    against it, effective order ~0.64 at the fine end; `luna_tests/lit_a10/check.md`), a third of the 800^2 value,
    so the 800 point and the fitted orders lean high; against this fixed reference the bounds are regression guards.
    Against the exact planar Riemann solution, in the 58.6 % of cells the corners and mirror-face shocks have not
    reached (critic, `critic_a10/critic.md`, job 31168630), HLLC-MC's error is 1.70e-5, 8.56e-6, 4.20e-6 at 800,
    1600, 3200^2: true order 1.0 there, against 1.19 / 1.57 for the same pairs measured against the 6400^2
    reference. The reference is itself 2.0e-6 off the exact solution there, and every scheme's error is lowered by
    that same 2.0e-6, so the ranking of schemes is unchanged; the 100-800 orders above are upper estimates.
  - t = 2.5 (LW's late time): L1(rho) against the 2048^2 reference on min(N, 256)^2 is 0.011-0.018 for every scheme
    and N in 128..1024, not decreasing; 2048 vs 1024 differ by 0.012. The late flow (jet, Richtmyer-Meshkov
    fingers, vortex pair) does not converge pointwise: recorded, not bounded.
  - Jet: s_jet, the largest distance along the diagonal (s < 0.25) with rho < 0.78 (between the jet gas, <= 0.76,
    and the background plateau, >= 0.80). At LW's 400^2 and t = 2.5: HLLC-MC 0.218, HLLC-vL 0.133, HLL-MC 0.121;
    HLLC-minmod 0.077, HLL-minmod 0.084, HLL-vL 0.088. The value does not depend on the time step (CFL 0.7-0.8,
    other output stops, symmetric noise <= 1e-8 change nothing; critic, `critic_a10/critic.md`, jobs 31168340-1)
    but moves with t as structures pass along the diagonal: HLLC-MC 0.198, 0.124, 0.218, 0.229, 0.149 at t = 2.40,
    2.45, .., 2.60, and at 2.6 the minmod and HLL-vL runs also read 0.149. A single snapshot is therefore a
    regression value (printed), not a jet criterion. Bounded: the median over t = 2.40..2.60 (step 0.05), >= 0.18
    (HLLC-MC), 0.12 (HLLC-vL), 0.11 (HLL-MC) for the schemes with a jet (as LW's CLAW/WENO and Athena), <= 0.10 for
    HLLC-minmod, HLL-minmod, HLL-vL (medians 0.198, 0.133, 0.121; 0.077, 0.084, 0.088, job 31169063). s_jet at
    fixed t is not monotone in N (HLLC-MC 0.07, 0.12, 0.22, 0.17, 0.18, 0.11 at 128..1024): each N sits at a
    different phase of the moving structure, so only LW's grid is gated.
  - Teeth: a 1e-14 seed fails the exact bound (GPU); the centre rule x + y < 0.15 with y_j = j dy + dy/2 against
    x_i = (i + 1/2) dx flips 2 half cells at 64^2 and fails by 0.06; the short run above. The directionally split
    driver was skipped (orchestrator).
  - CPU (1 device): 12 passed, 10 xfailed (strict), 35 s, 64^2 (2 devices: + 2 sharded, 46 s); GPU (1 A100,
    jobs 31110823-4): 60 passed in ~10 min, + 2 sharded on 2 A100 (31110825). GPU tier: early ladder, late L1,
    jet, 400^2 symmetry to 2.5, FP32, Pallas slab and 2-D Pallas at 400^2, 1e-14 seed.
  - Against AthenaK (2026-10-14, `lit_a10/athenak.md`, `luna-reference-solutions/athenak/lw_implosion/a10_*.npz`, critic `critic_a10/critic.md`):
    the same exact cell-average IC through pgen `luna_ic` (not AthenaK's `lw_implode`, which puts the interface dx/4
    out), reflecting walls, HLLC, CFL 0.4, `dfloor = pfloor = 1e-8`, FP64, one A100, MeshBlocks 50^2 below 1024;
    LUNA jobs 31169576-80, AthenaK 31110999-31111002, 31165951, 31169072-76, 31169394-99. L1(rho) at t = 0.045
    against the LUNA 6400^2 reference (in brackets: against AthenaK PPMX 6400^2 built with `-fmad=false`):

    | N | hllc-minmod | hllc-mc | hllc-vl | AthenaK PLM+RK2 | AthenaK PPMX+RK4 | AthenaK PPM4+RK4 |
    |---|---|---|---|---|---|---|
    | 100 | 3.91e-3 | 2.64e-3 | 2.87e-3 | 3.10e-3 | 2.27e-3 | 2.16e-3 |
    | 200 | 2.25e-3 | 1.46e-3 | 1.57e-3 | 1.73e-3 | 1.27e-3 | 1.21e-3 |
    | 400 | 1.32e-3 | 8.20e-4 | 8.93e-4 | 9.91e-4 | 6.80e-4 | 6.63e-4 |
    | 800 | 7.51e-4 (7.53e-4) | 4.39e-4 (4.44e-4) | 4.85e-4 (4.90e-4) | 5.40e-4 (5.45e-4) | 3.54e-4 (3.57e-4) | 3.49e-4 (3.53e-4) |
    | order 100-800 | 0.79 | 0.86 | 0.85 | 0.84 | 0.89 | 0.88 |

    L2 orders 0.37-0.44 for all six. The orders are upper estimates (reference bias, above; the exact planar region
    gives 1.0 for both codes). At 800^2 LUNA MC is about 0.8x AthenaK PLM and about 1.2x PPMX and PPM4 (0.81, 1.24,
    1.26 against the LUNA reference; 0.72 and 1.20 against the exact planar solution, critic_a10; ~5 % from the
    correlated reference bias); van Leer, AthenaK's slope, 0.90x PLM; minmod 1.39x PLM. At equal dt (LUNA CFL 0.614,
    978 steps vs 980, job 31169579) MC is 4.49e-4: the ranking does not depend on dt. LUNA MC reaches AthenaK PPMX's 800^2 error at
    N ~ 970-1020 (estimate from the local orders 0.9-1.1; not measurable against the 800^2 reference). The two references: AthenaK PPMX 6400^2 (no FMA, exactly
    symmetric) is 5.9e-5 (L1, 800^2) from LUNA's, below either code's self-convergence (LUNA 6400 vs 3200 8.0e-5,
    AthenaK 8.7e-5); against it every L1 changes by <= 2 %.
  - AthenaK's mirror symmetry: with nvcc's default FMA contraction no AthenaK run is exactly symmetric. One step is
    exact, 10 steps 2e-16 .. 9e-16; at t = 0.045 PLM 1.6e-15 (100^2) .. 2.0e-5 (800^2), PPM4 8e-15 .. 5e-13, PPMX
    2.7e-5 (100^2) .. 1.6e-2 (800^2) and 0.33 at 6400^2 (its limiter's switches amplify the round-off seed); at
    t = 2.5, 400^2: PLM 1.4e-7, PPM4 0.31, PPMX 0.41 (max abs(rho - rho^T); m, E similar). The same source built
    with `-fmad=false` (sha256 2e963374...) is exactly 0 for PLM, PPMX and PPM4 at every N and time run (100..800^2
    and PPMX 3200, 6400^2 at 0.045; 128..1024^2 at 2.5; the jet times), as Quokka found. FMA moves the early L1 in
    the 4th digit and the late flow. AthenaK's results also depend on the MeshBlock layout at round-off (each block
    computes its dx from its own edges, `athenak/src/mesh/meshblock.cpp:113-118`; PPMX 800^2 MeshBlock 50 vs 100
    differ by 1.2e-2, still symmetric without FMA, critic_a10): it has no equivalent of LUNA's N shards == one
    device. L1 moves by <= 0.11 % with the layout. AthenaK PPMX/PPM4 drift in mass and energy by 1.4e-15 per cycle
    (its 15-digit RK4 coefficients, critic_a10); PLM and LUNA conserve to round-off.
  - Jet, from the symmetric (`-fmad=false`) AthenaK runs: the test's median of s_jet over t = 2.40..2.60 at 400^2 is
    AthenaK PLM 0.130, PPMX 0.097, PPM4 0.158 (LUNA: MC 0.198, vL 0.133, minmod 0.077). The default (FMA) build gives
    PLM 0.130, PPMX 0.091, PPM4 0.228 from asymmetric runs (asymmetry up to 0.50); PPM4's 0.228 is an FMA artefact,
    and FMA PPMX s_jet moves with the MeshBlock layout (0.091 / 0.112 / 0.106 at 50 / 80 / 100, critic_a10). s_jet
    at t = 2.5, N = 128, 256, 400, 512, 1024, no FMA: PLM 0.071, 0.090, 0.130, 0.204, 0.111 (with LUNA vL 0.071,
    0.090, 0.133, 0.206, 0.110: same slope, same diagonal); PPMX 0.091, 0.218, 0.097, 0.111, 0.108; PPM4 0.078,
    0.153, 0.150, 0.079, 0.122. Late L1 (recorded, min(N, 256)^2 against the LUNA 2048^2 reference) 0.012-0.018 for
    both codes at every N, not decreasing.
- A3 done: `tests/test_hydro_accuracy_lw_smooth.py`, LW sec. 4.1 (checked in the paper). At 200^2
  HLLC + MC gives 4.45e-3 % (LW's best second-order: CLAW 3.7e-3, VH1 3.9e-3); minmod gives 0.136 %
  (HLL 0.212 %), above all eight LW schemes (worst WAFT 6.7e-2).
  Against AthenaK from the same initial data (pgen luna_ic), the test's `errors()`, HLLC, AthenaK CFL 0.4,
  GPU (FP64; LUNA 32..256 equal the CPU measurements above), `luna-reference-solutions/athenak/lw_smooth_advection/a3_*_1024.npz`,
  `lit_a03/athenak.md`:

  | N | L1 hllc-minmod | hll-minmod | hllc-mc | hll-mc | hllc-vl | AthenaK PLM+RK2 | AthenaK PPMX+RK4 |
  |---|---|---|---|---|---|---|---|
  | 32 | 0.0226 | 0.0362 | 0.00252 | 0.00394 | 0.00939 | 0.00984 | 3.91e-5 |
  | 64 | 0.0101 | 0.0135 | 5.20e-4 | 5.94e-4 | 0.00254 | 0.00292 | 2.47e-6 |
  | 128 | 0.00300 | 0.00487 | 1.19e-4 | 1.24e-4 | 5.94e-4 | 7.27e-4 | 1.55e-7 |
  | 256 | 8.43e-4 | 0.00136 | 2.58e-5 | 2.81e-5 | 1.35e-4 | 1.72e-4 | 9.67e-9 |
  | 512 | 2.35e-4 | 3.75e-4 | 5.56e-6 | 6.33e-6 | 3.02e-5 | 4.04e-5 | 6.05e-10 |
  | 1024 | 6.30e-5 | 1.01e-4 | 1.23e-6 | 1.44e-6 | 6.68e-6 | 9.39e-6 | 3.93e-11 |
  | 1024 at AthenaK's step count (LUNA CFL 0.714) | | | 1.65e-6 | | 6.92e-6 | 9.39e-6 | |
  | L1 order 32-256 / 256-1024 | 1.60 / 1.87 | 1.57 / 1.87 | 2.20 / 2.19 | 2.36 / 2.14 | 2.05 / 2.17 | 1.95 / 2.10 | 3.99 / 3.97 |
  | Linf order 32-256 / 256-1024 | 1.24 / 1.31 | 1.22 / 1.31 | 1.73 / 1.66 | 1.89 / 1.75 | 1.43 / 1.46 | 1.43 / 1.46 | 4.00 / 3.82 |
  | LW rel. L1 % at 200^2 | 0.136 | 0.212 | 4.45e-3 | 4.78e-3 | 0.0230 | 2.89e-2 | 2.6e-6 |

  hllc-vl is LUNA with van Leer, AthenaK's only PLM slope. At 1024^2 and LUNA's CFL 0.8, LUNA MC is 7.6x below
  AthenaK PLM; at equal dt (critic job 31107131) 5.7x, at LUNA CFL 0.4 (31105760) 2.9x: MUSCL-Hancock MC's error grows as dt
  shrinks (space-time cancellation), AthenaK's does not change with dt. With the same slope LUNA is 0.71x (0.74x at
  equal dt). LUNA's default minmod is 6.7x above AthenaK PLM. PPMX + RK4 is fourth order on this linear advection
  (3e4x below LUNA MC).
- A4 done: `tests/test_hydro_hard_lw_gresho.py` (LW sec. 4.10, p(0) = 5): static and moving vortex,
  periodic, one revolution, L1/L2/Linf of v, rho, p and orders; Galilean gate; MC no worse than LW's
  worst scheme. `test_hydro_hard_low_mach.py` now runs one revolution (it ran 0.8) with L1/L2/Linf:
  HLL loses the whole vortex at M <= 0.01, HLLC + minmod saturates at ~50 % KE loss, MC holds to ~3e-3.
  Zero-gradient edges reflect the acoustic start-up waves (HLL density error rises 32 -> 64).
  Against AthenaK from the same initial data (pgen luna_ic), the test's `periodic()` and `lw_metrics()` (on LW's
  40^2 and 160 x 40 directly), HLLC, AthenaK CFL 0.4, GPU FP64, `luna-reference-solutions/athenak/gresho_vortex/a4_*_1024.npz`,
  `lit_a04/athenak.md` (all fields and norms there):

  | N | v L1 hllc-minmod | hll-minmod | hllc-mc | hll-mc | hllc-vl | AthenaK PLM+RK2 | AthenaK PPMX+RK4 | AthenaK PPM4+RK4 |
  |---|---|---|---|---|---|---|---|---|
  | static 32 | 0.0688 | 0.144 | 0.0216 | 0.0541 | 0.0298 | 0.0295 | 0.0132 | 0.0137 |
  | static 256 | 0.00484 | 0.0131 | 0.00114 | 0.00264 | 0.00153 | 0.00153 | 4.48e-4 | 4.61e-4 |
  | static 1024 | 7.13e-4 | 0.00216 | 1.33e-4 | 3.18e-4 | 1.70e-4 | 1.70e-4 | 7.46e-5 | 5.23e-5 |
  | order 32-256 / 256-1024 | 1.27 / 1.38 | 1.15 / 1.30 | 1.43 / 1.55 | 1.42 / 1.53 | 1.44 / 1.59 | 1.43 / 1.58 | 1.62 / 1.29 | 1.63 / 1.57 |
  | moving 32 | 0.0819 | 0.143 | 0.0291 | 0.0552 | 0.0400 | 0.0435 | 0.0309 | 0.0340 |
  | moving 256 | 0.00558 | 0.0130 | 0.00147 | 0.00269 | 0.00197 | 0.00207 | 9.01e-4 | 0.00106 |
  | moving 1024 | 8.33e-4 | 0.00215 | 1.81e-4 | 3.27e-4 | 2.33e-4 | 2.51e-4 | 1.35e-4 | 1.50e-4 |
  | order 32-256 / 256-1024 | 1.27 / 1.37 | 1.15 / 1.30 | 1.45 / 1.51 | 1.42 / 1.52 | 1.46 / 1.54 | 1.47 / 1.52 | 1.70 / 1.37 | 1.68 / 1.41 |
  | static rho L1 1024 (order 256-1024) | 1.43e-5 (1.67) | 3.46e-5 (1.55) | 1.26e-6 (1.98) | 2.12e-6 (2.11) | 2.10e-6 (2.07) | 2.11e-6 (2.06) | 2.52e-6 (1.19) | 8.93e-7 (1.81) |
  | static p L1 1024 (order 256-1024) | 9.42e-5 (1.79) | 2.48e-4 (1.72) | 6.87e-6 (2.21) | 1.55e-5 (2.14) | 1.04e-5 (2.28) | 1.07e-5 (2.26) | 1.90e-5 (0.87) | 5.44e-6 (1.73) |
  | moving rho / p L1 1024 | 1.37e-5 / 8.92e-5 | 3.43e-5 / 2.45e-4 | 1.49e-6 / 8.25e-6 | 2.24e-6 / 1.58e-5 | 2.44e-6 / 1.23e-5 | 2.70e-6 / 1.35e-5 | 1.91e-6 / 1.25e-5 | 1.84e-6 / 1.27e-5 |
  | LW 40^2 static: vort %, rho %, TKE % | 52.4, 0.33, 38.3 | 87.8, 0.82, 71.3 | 25.2, 0.090, 8.8 | 50.4, 0.25, 23.8 | 28.0, 0.13, 14.3 | 28.0, 0.13, 14.3 | 15.3, 0.050, 0.95 | 24.1, 0.088, 1.96 |
  | LW 160 x 40 moving | 70.8, 0.17, 45.2 | 94.8, 0.21, 64.6 | 34.4, 0.054, 11.2 | 52.5, 0.071, 23.7 | 40.3, 0.082, 17.8 | 45.8, 0.10, 18.7 | 63.2, 0.25, 0.42 | 77.3, 0.28, 1.95 |

  With AthenaK's slope (van Leer) the codes tie on the static vortex (v L1 1.70e-4 both) and LUNA is 0.93x moving;
  LUNA MC is below AthenaK PLM in every field, norm and N (0.78x / 0.72x), LUNA's default minmod above it everywhere
  (4.2x / 3.3x). PPMX flattens from 512 (p L1 order 0.87; CFL 0.2 changes < 5 %): its extremum-preserving limiter,
  in the bulk of the vortex, not at the kinks (critic's localisation). PPM4, the same reconstruction without that
  limiter, beats LUNA MC in v, rho and p on the static vortex at 1024 (v 0.39x); on the moving vortex MC has the
  smaller rho and p errors than both. On LW's moving grid every AthenaK high-order scheme has larger vorticity and
  density errors than PLM and the smallest TKE errors (dt- and integrator-independent; open).
- A5 done: `tests/test_hydro_hard_mcnally_kh.py`, McNally, Lyra & Passy 2012 (arXiv:1111.1764v2) smooth KH, IC
  `KH-McNally-2D` (`src/luna/ic_generation/kh_mcnally.py`, eqs. 1-5 point-sampled at cell centres, exactly symmetric
  under (x, y, V_y) -> (x + 1/4, 1 - y, -V_y) for 4 | N), HLLC, CFL 0.8, periodic, to t = 1.5 with outputs at
  t = 0.02 k. M(t) of eqs. 6-9 against the Pencil 4096^2 curve of fig. 4 (`luna-reference-solutions/pencil/mcnally_kh/mcnally2012_fig4_Mref.txt`,
  read from the arXiv source PDF content stream by `tools/mcnally_digitize.py`: 10-decimal coordinates in float64,
  axes from the tick marks; agrees with an independent raw parse to 3.5e-10, with the MuPDF float32 read to 4.3e-6;
  the paper's GCI is <= 1.9e-6 relative). max(rho V_y^2 / 2) against fig. 8 (`mcnally2012_fig8_KEref.txt`; the seven
  panels' copies agree to 7.5e-12). FP64, GPU A100 (jobs 31109735, 31109736, 31110062, 31110238, pytest 31110674):

  | N | MC max_t eM | MC int_t eM / T | minmod max / int | van Leer max / int | MC max_t eK | Athena (paper) max_t eM |
  |---|---|---|---|---|---|---|
  | 64 | 0.240 | 0.120 | 0.495 / 0.258 | 0.300 / 0.152 | 0.715 | |
  | 128 | 0.0389 | 0.0173 | 0.167 / 0.0809 | 0.0613 / 0.0289 | 0.321 | 0.038 |
  | 256 | 4.27e-3 | 1.44e-3 | 0.0481 / 0.0220 | 8.93e-3 / 3.77e-3 | 0.0944 | 7.3e-3 |
  | 512 | 4.76e-4 | 1.65e-4 | 0.0128 / 5.67e-3 | 1.17e-3 / 3.32e-4 | 0.0197 | 1.6e-3 |
  | 1024 | 1.15e-4 | 5.32e-5 | 3.28e-3 / 1.43e-3 | 1.48e-4 / 4.02e-5 | 2.36e-3 | |
  | 2048 | 3.78e-5 | 1.75e-5 | 8.29e-4 / 3.58e-4 | 2.92e-5 / 1.36e-5 | 6.43e-4 | |
  | order of max | 2.52 (128-2048) | | 1.89 (128-1024) | 2.90 (128-1024) | | |

  The order of eM is not asymptotic (MC 3.2 at 128-512, 1.8 at 512-2048). Both the max over t and the integral over
  t are bounded; eM at t = 1.5 alone is printed, not bounded (critic A5, `luna_tests/critic_a05/critic.md`): MC and
  van Leer overshoot M until t ~ 1.0-1.3 and then change sign (MC crosses zero at t 1.10, 1.34, 1.44 at 512, 1024,
  2048; job 31169458), so MC's eM(1.5) = 4.8e-6 at 2048 and any order of eM(1.5) are crossing artefacts. Minmod is
  negative at every t. The absolute M errors are specific to the point-sampled IC, which is what the paper used: a
  point sample is the cell average minus (dx^2/24) f'', anti-diffused, which pre-compensates the schemes' numerical
  diffusion. A cell-averaged IC makes every code's M error 1.7-2.7x larger and removes the sign change (critic jobs
  31169034, 31168971). dt matters at this level: at CFL 0.4 (AthenaK's) MC's max moves -10 % at 512 and +19 % at
  1024 (1.37e-4, eM(1.5) -24 %), van Leer's -8 %; at equal step count (CFL 0.72) the max moves <= 5 % (MC 1024 +4.3 %), but eM(1.5)
  at 2048 still moves 13 % (comparator data, critic item 5). Fields at t = 1.5 against LUNA MC 4096^2
  FP64 (`luna-reference-solutions/luna/mcnally_kh/mcnally2012_luna_mc_ref.npz`, 458 KB: rho, rho V_y block-averaged to 256^2 in float32, M and K of
  the 4096 and 2048 runs, provenance; `tools/mcnally_kh_reference.py`, one A100, 35535 steps in 352 s, job 31110447):
  L1 of V_y 4096 vs 2048 on 256^2 is 6.7e-6, ~5 % of MC's error at 1024 if the order is ~2. L1(rho), L1(V_y) at
  64 / 1024 and the L1 order over 64-1024: MC 0.0214 / 1.36e-4, 1.79; 0.0173 / 3.42e-5, 2.24. Minmod 0.0413 /
  4.21e-4, 1.66; 0.0353 / 2.84e-4, 1.74. Van Leer 0.0261 / 1.34e-4, 1.89; 0.0214 / 3.76e-5, 2.30. MC's
  reference-free (Cauchy) orders are the same, 2.0-2.4 (V_y) and 1.6-2.0 (rho) (checker, `lit_a05/check.md`): the flow
  is smooth and resolved at t = 1.5, so the ~1 MUSIC reports (Goffrey+ 2017, van Leer, 256-2048) does not carry over.
  L2 and Linf bounded too (orders not).
  Invariants: M(0) = 0.01 to 2.2e-16; IC = eq. 1/3/5 to 9e-16; mass, energy, momentum to round-off at every output;
  symmetry kept to <= 2.9e-15 (FP64, not bitwise: a cell and its mirror differ in the last bit; no growth by 1.5),
  1.3e-6 FP32; sharded 2x1, 1x2, 2x2 at 64^2 and 2x2 at 128^2 bitwise equal to one device at every output (CPU, 8
  devices; the checker also 4x2). FP32 (MC): max_t |M32/M64 - 1| <= 4.2e-6 at 64-2048, so FP32 does not move M at any accuracy the test
  resolves. Teeth, all at 128 (CPU) and 512 (GPU): L = 0.0275 gives eM 0.21 / 0.18, gamma 1.4 0.118 / 0.113, the seed
  sign flipped above y = 1/2 gives M = 0 (eM 1.0; M(0) = 2e-19). Never bounded against 4.384 (erratum E3: this
  profile's compressible periodic linear rate is 2.83, the incompressible isolated one 3.02; Wang+ 2010 unread).
  Harness findings (`tests/_helpers.py:evolve`): FP64 lands on t = 0.02 k exactly (sum of dt = 0.02 k bitwise);
  FP32 could not reach a target that is not a float32 number (RunAborted at t = 0.02, job 31109737, fixed: the
  target is float32(t)), and its float32 clock drifted from the state's time by up to 3e-6 relative (Kahan-summed
  now, FP32 only, FP64 unchanged bitwise): that drift was 90 % of |M32/M64 - 1| at 2048 (3.7e-5 -> 4.2e-6, job
  31110238). The 4096 reference itself is 1.1e-5 from fig. 4 (bounded 1.5e-5). CPU: the file runs in 41-48 s on 8 CPU devices
  (tools/run_tests.py's default; 64^2, 128^2, sharded included; job 31167241, the checker 51-61 s), GPU 3.8 min on one
  A100 (31167242).
  Against AthenaK from the same initial data (pgen luna_ic), the test's own m_errors, ke_errors and field_errors,
  HLLC, AthenaK CFL 0.4, `dfloor = pfloor = 1e-8`, 64^2 MeshBlocks, GPU FP64, `luna-reference-solutions/athenak/mcnally_kh/a5_*.npz`,
  `lit_a05/athenak.md`. AthenaK clamps dt to tlim only (`mesh.cpp` NewTimeStep), so each run is 75 chained runs of
  0.02 from the previous final dump and M is read at t = 0.02 k exactly. LUNA: `athenak_refs.py luna a5`, job 31168485
  (CFL 0.8; 0.90x AthenaK's steps) and 31168486 (CFL 0.72, equal step count 9891 vs 9894 at 1024: M moves by <= 5 %,
  fields by <= 1 %; at CFL 0.4, job 31167303, MC's max eM at 1024 is 1.37e-4, 19 % above CFL 0.8). All absolute M
  errors are for the point-sampled IC (the paper's choice); from cell-averaged ICs every code's M error is 1.7-2.7x
  larger and the sign changes below disappear (A5 critic, `critic_a05/critic.md` sec. 4). max_t |M/M_ref - 1| against
  fig. 4:

  | N | LUNA minmod | LUNA MC | LUNA van Leer | AthenaK PLM+RK2 | AthenaK PPMX+RK4 | AthenaK PPM4+RK4 |
  |---|---|---|---|---|---|---|
  | 128 | 0.167 | 0.0389 | 0.0613 | 0.0589 | 0.0299 | 0.0324 |
  | 256 | 0.0481 | 4.27e-3 | 8.93e-3 | 8.44e-3 | 5.93e-3 | 5.90e-3 |
  | 512 | 0.0128 | 4.76e-4 | 1.17e-3 | 1.08e-3 | 1.40e-3 | 1.36e-3 |
  | 1024 | 3.28e-3 | 1.15e-4 | 1.48e-4 | 1.23e-4 | 3.45e-4 | 3.38e-4 |
  | 2048 | 8.29e-4 | 3.78e-5 | 2.92e-5 | 4.11e-5 | 8.56e-5 | 8.45e-5 |
  | 4096 | | | | | 2.09e-5 | |
  | order 128-2048 | 1.92 | 2.52 | 2.80 | 2.71 | 2.10 | 2.13 |
  | max_t K error 2048 | 0.0187 | 6.43e-4 | 2.10e-3 | 1.93e-3 | 7.90e-4 | 1.35e-3 |
  | integral over t / 1.5 (the test's bound), 512 | 5.67e-3 | 1.65e-4 | 3.32e-4 | 2.62e-4 | 6.85e-4 | 6.67e-4 |
  | integral, 1024 | 1.43e-3 | 5.32e-5 | 4.02e-5 | 5.27e-5 | 1.68e-4 | 1.65e-4 |
  | integral, 2048 (4096) | 3.58e-4 | 1.75e-5 | 1.36e-5 | 1.91e-5 | 4.15e-5 (9.89e-6) | 4.08e-5 |

  Fields at t = 1.5 on min(N, 256)^2, L1(V_y) / L1(rho), against the AthenaK PPMX 4096^2 reference (and, in brackets,
  against the LUNA MC 4096^2 one):

  | N | LUNA minmod | LUNA MC | LUNA van Leer | AthenaK PLM+RK2 | AthenaK PPMX+RK4 | AthenaK PPM4+RK4 |
  |---|---|---|---|---|---|---|
  | 128 | 0.0120 / 0.0153 | 3.10e-3 / 5.12e-3 | 4.59e-3 / 6.31e-3 | 4.46e-3 / 6.17e-3 | 2.25e-3 / 4.13e-3 | 2.55e-3 / 4.29e-3 |
  | 512 | 1.05e-3 / 1.44e-3 | 1.41e-4 / 5.35e-4 | 1.78e-4 / 5.04e-4 | 1.71e-4 / 5.16e-4 | 1.05e-4 / 3.23e-4 | 1.14e-4 / 3.33e-4 |
  | 1024 | 2.83e-4 / 4.20e-4 | 3.51e-5 / 1.44e-4 (3.42e-5 / 1.36e-4) | 3.81e-5 / 1.42e-4 | 3.70e-5 / 1.47e-4 | 2.33e-5 / 5.88e-5 (2.46e-5 / 6.49e-5) | 2.45e-5 / 6.08e-5 |
  | 2048 | 7.29e-5 / 1.15e-4 | 8.20e-6 / 3.70e-5 (6.70e-6 / 2.77e-5) | 8.41e-6 / 3.76e-5 | 8.55e-6 / 3.96e-5 | 4.52e-6 / 1.05e-5 (6.09e-6 / 1.75e-5) | 4.66e-6 / 1.07e-5 |
  | L1 order 64-1024 (128-1024 AthenaK) | 1.74 / 1.67 | 2.23 / 1.77 | 2.30 / 1.87 | 2.30 / 1.79 | 2.19 / 2.02 | 2.23 / 2.03 |

  - The reference gate: AthenaK PPMX 4096^2 is 2.09e-5 from fig. 4 (max over t, at t = 1.2); its M converges at
    order 2.0 (3.45e-4, 8.56e-5, 2.09e-5), and the order-2 Richardson value from 2048 and 4096 is 1.2e-6 from fig. 4
    (LUNA MC 4096/2048: 2.6e-6), at the level of the Pencil curve's own GCI. IC, pgen and diagnostic agree with
    McNally to ~1e-6. The two 4096^2 references differ by L1(V_y) 2.26e-6, L1(rho) 1.02e-5 on 256^2 (2.7e-6 and
    1.4e-5 on 1024^2), less than each differs from its own 2048 run (AthenaK 4.5e-6 / 1.05e-5, LUNA 6.7e-6 / 2.8e-5).
  - M: LUNA MC / AthenaK PPMX is 0.24-0.53x at 512-2048 in every metric the critic tried (max over t, integral and
    rms of abs(e) over t; 0.33x, 0.44x in the max at 1024, 2048; 0.53x at 4096 against LUNA's 4096 reference run).
    Against the PLM family (LUNA van Leer, AthenaK PLM) MC ties at 1024: smallest in the max, but in the integral
    and rms metrics van Leer is lowest (4.02e-5 vs MC 5.32e-5, AthenaK PLM 1.01x MC), and at CFL 0.4 MC (1.37e-4) is
    above AthenaK PLM (1.23e-4). The errors of LUNA MC, van Leer and AthenaK PLM change sign at t ~ 1.0-1.3, so their
    values at t = 1.5 sit near zero crossings and say nothing about accuracy; PPMX and PPM4 undershoot at every t.
    That error structure, not a higher order, is why AthenaK PLM is below PPMX in M from 512 on: at the finest pair
    1024 -> 2048 the orders are MC 1.61, AthenaK PLM 1.58, PPMX 2.01, and MC / PPMX grows with N. LUNA's default
    minmod is 27x / 20x above AthenaK PLM at 1024 / 2048.
  - Fields: PPMX wins in every reading. LUNA MC / PPMX at 1024 is 1.3-1.5x in V_y and 1.5-2.4x in rho in L1 (256^2 or
    own grid, either reference), 3-4x in L2 and Linf; from cell-averaged ICs 1.11x and 1.93x (critic sec. 4, 7).
    LUNA van Leer and AthenaK PLM agree within 4 % (L1) at 128-1024 (rho 5-6 % at 2048, reference-limited); LUNA
    MC is below them in V_y (0.95x AthenaK PLM at 1024) and equal in rho.
  - The field order: against the AthenaK reference LUNA's L1 orders are unchanged (MC 1.77 / 2.23 vs 1.79 / 2.24
    against its own), and every AthenaK scheme converges at the same ~2 (1.8-2.3) against either reference. On the
    run's own grid (no averaging to 256^2) and in MUSIC's relative measure the V_y order 256-1024 is 1.85-2.20 for all
    six schemes (MC 2.0); MUSIC's relative L1(V_y) at 1024 is 4.55e-3, LUNA MC's 5.6e-4 and AthenaK PPMX's 3.8e-4.
    So the ~2 is not LUNA converging onto its own reference, nor the 256^2 averaging: two independent codes and five
    reconstructions agree. MUSIC's order ~1 (and its 8x larger error) is MUSIC's, not the problem's.
- A6 done: `tests/test_hydro_hard_isothermal_atmosphere.py` (Berberich et al. sec. 3.1, checked: rho = p =
  exp(-sin 2 pi x), gamma 1.4, periodic, t = 2/sqrt(gamma)): unbalanced drift, L1/L2/Linf of rho, rho u, E,
  order ~2 (HLL's density drift ~50x HLLC's). New: `external_potential` = sin | uniform | point_mass
  (`src/luna/sources/external_gravity.py`, `tests/test_external_potential.py`) and `tests/_helpers.
  evolve_staged`, the production loop with stages and projections. Unblocks A13 (RT) and A17.
  Against AthenaK from the same cells (2026-10-14, `luna_tests/lit_a06/athenak.md`): AthenaK's `const_accel` is uniform
  only, so pgen `luna_ic` got a static potential (`<problem> potential = sin`, a user source term: rho g and rho v g
  with g = -dphi/dx at the cell centres, every RK stage; AthenaK branch `luna_ic_a6` 687f5a64, binary
  `build_gpu_lunaic_a6`, `tools/athenak/`). The test's `errors()`, HLLC, FP64, floors 1e-8, AthenaK CFL 0.4 and 0.8
  (= LUNA's dt, equal step counts), `luna-reference-solutions/athenak/isothermal_atmosphere/a6_*_2048.npz`, LUNA `luna_tests/lit_a06/luna_a6.json`
  (job 31172111, clean 05815aa). L1 of rho(T) - rho(0) and the energy budget abs(d(E + rho phi)) / (E + rho phi):

  | N | minmod | hll-minmod | MC | van Leer | AthenaK PLM+RK2 | PLM at LUNA's dt | PPMX+RK4 | PPMX at LUNA's dt | PPM4+RK4 |
  |---|---|---|---|---|---|---|---|---|---|
  | rho 128 | 7.60e-5 | 4.37e-3 | 4.65e-5 | 4.46e-5 | 4.86e-5 | 4.82e-5 | 2.94e-5 | 2.94e-5 | 3.60e-5 |
  | rho 512 | 4.78e-6 | 3.72e-4 | 3.44e-6 | 3.41e-6 | 7.96e-7 | 8.03e-7 | 1.84e-6 | 1.84e-6 | 1.94e-6 |
  | rho 1024 | 1.20e-6 | 1.02e-4 | 8.83e-7 | 8.79e-7 | 1.14e-7 | 1.18e-7 | 4.61e-7 | 4.61e-7 | 4.72e-7 |
  | rho 2048 | 3.00e-7 | 2.72e-5 | 2.23e-7 | 2.23e-7 | 2.04e-8 | 2.07e-8 | 1.15e-7 | 1.15e-7 | 1.17e-7 |
  | order 128-1024 | 2.00 | 1.81 | 1.91 | 1.89 | 2.91 | 2.89 | 2.00 | 2.00 | 2.08 |
  | budget 128 | 2.21e-4 | 7.19e-4 | 7.48e-6 | 1.55e-5 | 7.19e-5 | 7.19e-5 | 1.86e-9 | 1.91e-9 | 2.92e-6 |
  | budget 1024 | 3.35e-6 | 1.01e-5 | 1.38e-8 | 3.04e-8 | 1.40e-7 | 1.40e-7 | 6.9e-12 | 3.4e-12 | 7.07e-10 |
  | budget 2048 | 8.35e-7 | 2.51e-6 | 1.72e-9 | 3.79e-9 | 1.75e-8 | 1.75e-8 | 1.4e-11 | 7.1e-12 | 3.05e-11 |
  | budget order 128-1024 | 2.01 | 2.05 | 3.03 | 3.00 | 3.00 | 3.00 | (floor) | (floor) | 4.00 |
  | steps 2048 | 5121 | 5121 | 5121 | 5121 | 10241 | 5121 | 10241 | 5121 | 10241 |

  In the drift AthenaK wins: at 2048 LUNA MC is 11x AthenaK PLM and 1.9x PPMX (at the test's 1024: 7.7x, 1.9x);
  van Leer (AthenaK's slope) is the same as MC, so it is not the limiter. AthenaK PLM converges at 2.9 (2.6 over
  512-2048), every LUNA limiter and AthenaK PPMX/PPM4 at 2.0; E L1 follows rho (PLM order 2.9, MC 10x PLM at 2048),
  mom L1 less so (PLM order 2.35, MC 3.7x); in Linf every HLLC scheme is at order ~2 and AthenaK PLM's mom Linf is
  4.9x MC's. AthenaK's drift does not depend on dt (CFL 0.4 vs 0.8 within 4 %); LUNA's does:
  MC at 1024 is 8.8e-7 at CFL 0.8, 6.8e-7 at 0.4, 3.8e-7 at 0.2 (van Leer 8.8e-7, 6.6e-7, 3.7e-7;
  `lit_a06/luna_a6_cfl.json`, job 31172324), still 3.4x AthenaK PLM at 0.2. Candidates, not pinned: the source-free
  Hancock predictor (`src/luna/hydro/muscl_hancock.py:229`) and the Strang split around it. In the energy budget LUNA
  MC and van Leer beat AthenaK PLM (0.10x, 0.22x at 2048, all three at order 3.0); AthenaK PPMX holds E + rho phi to
  1e-11..1e-14 from 256 on (no converging series), PPM4 converges at 4.0. The teeth run (phi -> -phi at 256) gives
  rho L1 1.28-1.29 in every scheme of both codes. Cost: 2048 cells in 1D are launch-latency bound (AthenaK PLM 0.48 ms
  per step, LUNA ~0.8-1.5 ms per step with the Python loop), not a throughput comparison.
- A7 done: `tests/test_gravity_jeans_wave.py` (as Athena++ jeans): oblique wave, k = 2 pi (1, 2, 2),
  nj = 0.5 one period, nj = 1.5 growing mode one e-fold; L1/L2/Linf and mode amplitude, orders ~2
  (MC ~3 stable). Gravity precision now follows the test mode (`_MODE_CFG`, `check_precision`).
  Against AthenaK from the same initial data (pgen luna_ic on upstream c5a0d7f, multigrid self-gravity solved to
  round-off, binary `build_gpu_lunaic_mg`), the test's `errors()`, HLLC, GPU FP64, `luna-reference-solutions/athenak/jeans_wave/a7_*_128.npz`
  (CFL 0.3) and `a7eqdt_*` (CFL 0.8/3 = LUNA's dt: LUNA's sum rule, AthenaK's min rule, cubic cells, v ~ 0),
  `luna_tests/a07_a08_athenak.md`; L1 of (rho - exact) / AMP:

  | N | hllc-minmod | hllc-mc | hllc-vl | AthenaK PLM+RK2 | PLM at LUNA's dt | AthenaK PPMX+RK4 | PPMX at LUNA's dt |
  |---|---|---|---|---|---|---|---|
  | stable 32 | 0.0821 | 0.0223 | 0.0345 | 0.0611 | 0.0610 | 2.29e-5 | 2.03e-5 |
  | stable 64 | 0.0197 | 0.00257 | 0.00459 | 0.00842 | 0.00842 | 2.57e-6 | 2.54e-6 |
  | stable 128 | 0.00457 | 3.05e-4 | 5.79e-4 | 0.00102 | 0.00103 | 5.87e-7 | 5.92e-7 |
  | order 32-128 | 2.08 | 3.10 | 2.95 | 2.95 | 2.94 | 2.64 | 2.55 |
  | unstable 32 | 0.0683 | 0.0430 | 0.0482 | 0.0419 | 0.0398 | 0.0109 | 0.0109 |
  | unstable 64 | 0.0160 | 0.00964 | 0.0104 | 0.00711 | 0.00656 | 0.00263 | 0.00263 |
  | unstable 128 | 0.00387 | 0.00225 | 0.00238 | 0.00131 | 0.00117 | 6.53e-4 | 6.53e-4 |
  | order 32-128 | 2.07 | 2.13 | 2.17 | 2.50 | 2.54 | 2.03 | 2.03 |
  | amp 128 stable / unstable | 0.00751 / 0.00237 | 4.86e-4 / 0.00128 | 9.61e-4 / 0.00135 | 0.00165 / 6.46e-4 | 0.00166 / 5.66e-4 | 2.86e-7 / 3.78e-4 | 2.84e-7 / 3.78e-4 |

  Stable wave (one period): LUNA MC is 0.30x AthenaK PLM, van Leer (AthenaK's slope) 0.57x, minmod 4.5x; PPMX is
  ~500x below MC (fourth-order space, RK4). Unstable mode (one e-fold): AthenaK PLM beats every LUNA limiter (MC
  1.7x PLM, 1.9x at equal dt). LUNA's growing-mode error is temporal: MC at 128 gives 2.25e-3 at CFL 0.8, 1.19e-3
  at 0.4, 7.5e-4 at 0.2 (job 31167801), toward PPMX's 6.5e-4, which converges at exactly second order and is
  likely the shared 7-point Poisson / centred-gradient error; the stable case is blind to a frequency error at one
  period to first order. MC's stable error grows as dt shrinks (3.05e-4, 4.08e-4, 4.84e-4), as on A3. AthenaK PLM +
  RK2 at CFL 0.4 is unstable in 3D (stable wave at 128: L1 1.5e5, `luna_tests/a07_a08_ak/cfl04/`); 0.3 is used.
- A8 done: `tests/test_gravity_evrard_energy.py` (Springel 2010 sec. 9.1): periodic L = 4R, W = 1/2
  sum rho phi from the 7-point periodic solve. |dE/E0| reaches 3.0 (32^3) and 1.5 (64^3) by t = 3,
  first order (cusp, then the accretion shock), as AREPO's cell-centred ~50 % (isolated |E0| is 2.2x
  larger); conservative gravity coupling is the fix (< 1e-3 in AREPO). 128^3, 256^3 on viper.
  Flux mode (`gravity_energy='flux'`, what 'auto' picks; Mullen, Hanawa & Gammie 2020; docs/GRAVITY.md):
  |dE/E0| at t = 3 is 1.1e-6, 1.1e-5, 1.7e-5, 1.2e-5 at 32-256 with no floor (A100). That is the discrete
  invariant: with the spectral Green's function the energy error is the Poisson truncation, 3.3e-2 to 3.5e-3
  at 64-256 (critic). The accuracy claim is the profile against AthenaK PLM 256^3: L1 ln rho at t = 3 is
  0.104, 0.049, 0.0067, against kick's 0.73, 0.40, 0.21.
  `tests/test_gravity_energy_flux.py`:
  - bounds 1.3x measured;
  - teeth: the kick, Phi^n+1 alone (0.018 against 4.4e-11), and the half split without its donor faces (NaN at
    step 1);
  - the turbulent box, A7 in flux mode, gas + DM;
  - sharded == one device, Pallas == XLA mass fluxes, VJP.
  AthenaK's coupling is not this one: a single-time Phi (`srcterms.cpp:207-213`). Its 3.7e-6 is truncation.
  Against AthenaK from the same initial data (upstream c5a0d7f multigrid to round-off, `build_gpu_lunaic_mg`),
  periodic as LUNA, the test's `energies()` on AthenaK's float64 states, floors 1e-8, CFL 0.3, HLLC,
  `luna-reference-solutions/athenak/evrard/a8_*_256.npz`, `luna_tests/a07_a08_athenak.md`. 32, 64: one AthenaK run per output time
  (tlim = 0.25 k, exact); 128, 256: its dumps, up to 0.018 late (at 64 the late dumps change |dE/E0| by < 7 % at
  t = 0.25, < 0.3 % from t = 1.25). LUNA 128, 256 measured here (job 31167760, GPU); hllc-mc 32 aborts (p < 0).
  |dE/E0| at t = 1 / t = 3:

  | N | hllc-minmod | hllc-mc | hllc-vl | AthenaK PLM+RK2 | AthenaK PPMX+RK4 |
  |---|---|---|---|---|---|
  | 32 | 1.31 / 3.02 | aborts | 1.34 / 2.60 | 1.23e-3 / 2.36e-3 | 0.154 / 0.162 |
  | 64 | 0.726 / 1.55 | 0.617 / 1.08 | 0.634 / 1.19 | 1.93e-4 / 2.43e-4 | 4.77e-5 / 1.22e-5 |
  | 128 | 0.390 / 1.01 | 0.299 / 0.568 | 0.312 / 0.666 | 1.84e-5 / 2.79e-5 | 1.21e-5 / 9.83e-5 |
  | 256 | 0.203 / 0.581 | 0.119 / 0.241 | 0.137 / 0.304 | 1.99e-6 / 3.71e-6 | 2.07e-6 / 4.48e-5 |
  | order 64-256 at t = 3 | 0.71 | 1.08 | 0.99 | 3.02 | -0.94 |

  AthenaK's energy error is 1e-4..1e-5 of LUNA's at every N: its gravity energy source uses the Godunov mass fluxes
  times the face potential differences (Mullen, Hanawa & Gammie 2020; `src/srcterms/srcterms.cpp:215-305` upstream),
  LUNA kicks with the cell-centred gradient. The error is made at the collapse and bounce (t 0.75-1.25) and is
  flat after t = 1.5 for AthenaK, while LUNA's keeps growing (accretion shock). The trajectories agree (W(t = 1):
  AthenaK PLM 64 -1.84, LUNA MC 256 -1.92, MC 64 -1.60). At 32 both AthenaK schemes reset cells to the pressure
  floor at the bounce (PLM 8-112 cells, its error doubles there; PPMX 52 cells, |dE/E0| jumps to 0.154 between
  t = 0.75 and 1; floor 1e-12 gives 0.1547 vs 0.1543, so the reset, not its value), the same place LUNA MC 32
  aborts with p < 0. PPMX is not monotone from 64 (1.2e-5, 9.8e-5, 4.5e-5 at t = 3, the jump at t 1-1.25, no
  floored cell in the dumps; not analysed). LUNA's order is ~1 to 256 (MC 1.08), AthenaK PLM's ~3.
- KH linear (`tests/test_hydro_accuracy_kh.py`): seeds the exact compressible eigenmode of the double tanh
  layer (`luna.hydro.exact_kh_linear.linear_mode`, sigma = 1.257385, 1024-point Fourier = 256-point to 8 digits)
  as cell averages; L1/L2/Linf of v_y against the exact mode at T = 4 and the fitted growth rate, orders
  ~2.0 (CPU 32, 64; GPU 64..256). The old seed (sin kx times a Gaussian) excited other modes: growth-rate
  error flattened at order 0.61 by 256. AMP / 10 agrees to 1.4e-4 (10 AMP does not, 1.6e-3). AthenaK
  PPMX + RK4 from LUNA's exact initial data (pgen luna_ic) converges to the same sigma at order ~2.0 with
  L1 3.1-3.3x smaller at every N (0.0132, 0.00305, 0.00074 at 64, 128, 256): second order in 2D, as
  Felker & Stone 2018 explain (no transverse flux correction).
- A13 done: `tests/test_hydro_hard_rayleigh_taylor.py`, LW sec. 4.6 box and walls (checked in the paper: no
  gamma, pressure level or smoothing there); gamma 1.4 and p = 2.5 at the interface as Springel 2010 sec. 8.8,
  tanh width 0.025 ours. Seeds the exact linear eigenmode (shooting, sigma = 0.661770; a finite-difference
  eigen-solve agrees to 1e-6) at +AMP and -AMP and uses (U+ - U-)/2: no drift, no even orders in AMP.
  AMP = 1e-7 (1e-8 agrees to 3 digits); the old 1e-6 against an unseeded control went nonlinear at 512
  (hllc-mc L1 step order 0.68, any limiter). Ladders 32..128 (CPU) and 128..512 (GPU): growth-rate error
  order ~1.9 (minmod), ~3.0 / 2.3 (MC, CPU / GPU); L1 at 512 hllc-mc 4.8e-6; E + rho g y order 2; the
  background drift first order (mirrored wall ghosts, not hydrostatic). HLL smears the resting interface,
  so its rate converges slowly on the CPU ladder (orders ~1.2-1.5), ~1.8 by 512. No paper read runs RT as a
  linear convergence test (docs/history/AUDIT_TESTS_2026-10-07.md).
- AthenaK references (the benchmark): Shu-Osher and Titarev-Toro at
  16384 (+ 8192), PPMX + RK4 + HLLC, stored in `$LUNA_DATA/hydro/athenak/*_ppmx_rk4.npz` with provenance.
  LUNA-MC 8192 vs AthenaK 16384: L1(rho) 0.0063 (SO), 0.030 (TT); AthenaK 8192 vs 16384: 0.0088, 0.015.
  Up to 2048 every scheme's L1 is the same against AthenaK and against LUNA's own fine run (2-3 digits).
  RT (A13) from LUNA's exact initial data (pgen luna_ic), +-AMP, L1 / growth-rate error at 32..512:
  AthenaK 0.068/0.24, 0.037/0.093, 0.0085/0.013, 0.0013/0.0016, 3.4e-4/3.0e-4; LUNA minmod 0.0195/0.052 ..
  1.3e-4/2.6e-4, MC 0.0044/0.013 .. 4.8e-6/7.5e-6. SO, TT, WC gated 2026-10-30 (next bullet). Woodward-Colella runs
  on the GPU build with `dfloor = pfloor = 1e-8` (AthenaK PPMX/PPM4 + RK4 at 16384, `luna-reference-solutions/athenak/woodward_colella/a9_*`); at
  AthenaK's default FLT_MIN floors PPMX goes NaN at the blast collision, and FOFC does not help (1D: illegal memory
  access on this GPU build; 2D: still NaN; critic_p1, 2026-10-12).
- AthenaK gates done (phase B, 2026-10-30): Shu-Osher, Woodward-Colella, Titarev-Toro, A10 (t = 0.045), A11 (config 3
  outside the core) and A5 (rho, V_y at t = 1.5) are gated against AthenaK PPMX + RK4 references (SO 16384, WC 16384,
  TT 65536 cells, A10 6400^2 no-FMA, A11 8192^2, A5 4096^2; `tests/_athenak.py` asserts each run's scheme from its
  stored athinput and prints its provenance). Bounds: ~1.3x LUNA's measured L1/L2 against AthenaK; each reference's
  own error EPS_REF (half-N run or LUNA's fine run, the larger) is bounded and printed. Matched method: LUNA van Leer
  (MUSCL-Hancock) against AthenaK PLM (van Leer limiter + RK2) at the same N, bound L1(AthenaK PLM) + 2 EPS_REF, no
  LUNA number in it; it holds for SO, WC, TT, A10, A11 (A11 1024, 2048: LUNA 3 %, 2 % above AthenaK PLM, inside
  EPS_REF) and fails for A5 V_y by 3-4 % at 128..1024 (rho at 128 by 2 %), so A5 asserts the weaker "within 5 %"
  (PLM_RATIO 1.05). New SO/TT AthenaK runs: `luna_tests/phase_b/ak` (SLURM 31699054-6), reduced by
  `tools/athenak_refs.py reduce so|tt` (31699122); measured CPU-8 FP64 (WC 31699086, SO 31699149, TT 31699150) and
  A100 FP64 (A10 31699123, A11 31699124, A5 31699125). LUNA's own fine-grid references stay as regression only;
  A11 config 6 and the cores, and the A10 late time, have no converged reference in either code.
- A14 done: `tests/test_hydro_hard_richtmyer_meshkov.py`, single-mode Richtmyer-Meshkov in a closed tube (walls
  all round, lab frame, box half a wavelength wide), heavy -> light (reflected rarefaction) and light -> heavy
  (reflected shock), theta-group states (Thornber et al. 2017: gamma 5/3, Mach 1.8439, rho 3 and 1), k psi0 =
  0.01, seeded +psi0 and -psi0 (odd part). Exact target `tools/rm_linear.py`: Cobos-Campos & Wouchuk 2017
  (rarefaction) and 2016 (shock), asymptotic velocity matching every printed digit of their worked examples
  (CW17 App. B, CW16 App. D) and the full psi(t) by Bessel series and by numerical inverse Laplace transform
  (agreeing to ~1e-9). Napieralski et al. 2024 (JFM) covers the shock case only. L1/L2/Linf of psi/psi0 and
  the fitted growth-rate error over 2 <= k c_af t <= 20, minmod and MC, CPU 32, 64 and GPU 128..512; orders
  ~0.6-0.8 (minmod light -> heavy L1 0.87, 0.62 .. 0.18 at 512). The error is mostly the post-shock growth
  rate (minmod -7.3 %, MC -3.7 % at 512, light -> heavy), not the start-up offset: the deposited vortex sheet
  smeared by PLM + limiter; CFL 0.4 changes nothing, HLL is 2-4 points worse than HLLC. AthenaK PPMX + RK4 from
  the same initial data (pgen luna_ic) converges to the same exact curve at ~first order, L1 0.040 / 0.049 at
  512 (rate -3.2 % / -1.9 %), stored in `luna-reference-solutions/athenak/richtmyer_meshkov/rm_ppmx_rk4_512.npz` (regenerated on Raven 2026-10-11); Richardson-extrapolated
  AthenaK end values land within 0.2-0.4 % of the theory. LUNA's weakest problem so far relative to AthenaK.
  Not read: Wouchuk 2026 (Phys. Plasmas 33, 022108, closed form for the rarefaction case; not fetchable here).
- A15 done: `tests/test_hydro_hard_taylor_green_reynolds.py`, LUNA's effective Reynolds number on the Taylor-Green
  vortex (M = 0.1, the existing IC) to t = 20: eps(t) = -dK/dt matched over 4 <= t <= 14 to viscous pseudo-spectral
  Taylor-Green runs at Re = 25 .. 1600 (`$LUNA_DATA/hydro/tg_viscous_spectral.npz`, euler_3d branch 'viscosity'
  3940684; Re 1600 at 512^3 reproduces the HiOCFD C3.5 reference, peak 0.012855682 at t 8.95 vs 0.012857528 at
  8.97). APSARA's (Wongwathanarat et al. 2016) Re = -Omega/(dK/dt) is half the usual u0/(k nu) and only a
  figure; Grete et al. 2023 / Malvadi Shivakumar & Federrath 2025 fit driven turbulence. Re_eff: minmod 26, 61,
  107, 472 at 32, 64, 128, 256^3; MC 55, 189, 813, >= 1600; AthenaK PPMX + RK4 549 at 64^3, >= 1600 above. The
  limiter dominates (128^3: minmod 107, van Leer 514, MC 813), low Mach second (minmod 92 / 107 / 335 at M =
  0.05 / 0.1 / 0.29), CFL 0.4 identical, HLL lower (minmod 81, MC ~200). At M = 0.29 LUNA MC (1166 at 128^3)
  matches APSARA / Prometheus (~1200 in our convention). Gated: Re_eff (lower bound) and its rate on the GPU
  ladder; the distance D to the Re 1600 curve for MC only (minmod's D is within 1.3x of a first-order run's).
  Minmod at 32^3 cannot be told from first order (<= 25): the CPU teeth use MC's bound, the GPU teeth minmod's
  at 64^3. Open: Re 3200 reference (768^3) to place AthenaK 128-256 and LUNA MC 256.
- A16 part A done: `tests/test_cosmo_adiabatic_expansion.py`. Fixes a silent bug: comoving runs with gamma !=
  5/3 had no (3 gamma - 5) internal-energy term (Martel & Shapiro 1998 eq. 41; Teyssier 2002 sec. 2.6 states it
  is needed and omits it), so T(a) was wrong (p~ off by 21^0.8 at z = 0 for gamma 7/5, reproduced by the
  teeth). Now `sources/comoving_energy.py` multiplies the internal energy by the exact (a_1 / a_0)^(5 - 3 gamma)
  around the hydro (off at 5/3: those runs unchanged); comoving AMR refuses gamma != 5/3. Uniform gas, EdS, z =
  20 -> 0: p~ ~ a^(5 - 3 gamma) (MS98 eq. 44) to round-off (1.8e-14 / 2.8e-14 at 5/3 / 7/5), v~ and rho~ exact,
  the a(tau) table against the exact EdS tau(a) to 4.4e-6. Enzo's AdiabaticExpansion has no exact check (answer
  tests only; its docs call it a poor test, the T floor dominates); no published numbers elsewhere.
- Comoving flux gravity energy done (2026-10-23, A100 + CPU-8, FP64): `tests/test_gravity_energy_flux_comoving.py`,
  docs/GRAVITY.md "Comoving flux". Top-hat 64^3 per-step Layzer-Irvine ledger 3.1e-5 (tol 1e-4) and 2.6e-11
  (tol 1e-10), kick 2.4e-3; teeth: old Phibar 1.7e-5, kick. Gas + DM 1.6e-11. Cold gas heats: top-hat
  flux_moved/|W| 3.4 %, IGM e_th 12x floor (kick 1.8x); pancake entropy L1 at a = 0.75 flux 1246/408/38 vs
  kick 275/75/19 at 16/32/64. Regression only for the heating; 'auto' stays kick in comoving runs.
- A16 part B done: `tests/test_cosmo_tophat_collapse.py`. Cold uniform sphere (comoving R0 0.15, z = 50,
  turnaround at a = 0.5) in EdS against the closed-FRW interior (`luna.cosmology.exact_tophat`: parametric
  R = A (1 - cos th), t = B (th - sin th), checked against a direct R'' = -GM/R^2 integration to 2e-10, and
  9 pi^2/16 at turnaround). The box outside is lowered to mean 1, so by Gauss the interior feels exactly its own
  overdensity under the mean-subtracted Poisson solve (R0 0.1 and 0.15 agree: periodic images are negligible).
  Gated: the mean rho~ of the inner half of the sphere, the interior v~ slope, and the Layzer-Irvine budget
  d(K~ + U~ + W~) = W~ dln a (super-comoving, gamma 5/3, W~ on the 7-point stencil, Simpson over 160 snapshots).
  Minmod, dloga_max 0.01: core error at turnaround 7.9e-2 / 2.9e-3 / 4.4e-4 at 64 / 128 / 256^3 (order 2.7),
  at 1 + delta = 20 1.0e-3 at 256^3. Teeth: gravity 1 % too strong fails all three kinds of 64^3 bound.
  Findings: pure cold gas goes negative-pressure within a few steps (E - KE truncation), so p~ = P0 = 1e-10 with
  `cosmo_p_floor` = P0 (1e-12 .. 1e-9 agree to 1e-5); the 8 centre cells keep a ~1 % spike where the flow meets at
  a point (gate the core mean, not the centre); at 256^3 dloga_max is the binding step (core error at a = 0.65:
  -6.2e-3 / -2.9e-3 / -5.1e-4 / +4.6e-4 at dloga 0.04 / 0.02 / 0.01 / 0.005); FP32 V-cycles and the 1e-4 warm
  tolerance are not the limit. MC at 128^3 matches minmod at 256^3.
- A16 cross-check done (2026-10-22, Raven): mini-ramses 55daaa5 (gfortran 13, OpenMPI 4.1; minmod, HLLC,
  courant 0.8, uniform grid, float32 grafic ICs and outputs) and public Arepo 1fc135a (Intel 19.1.2, FP64 with
  double I/O, PMGRID 128, CourantFac 0.3, no (de)refinement) from the same ICs (`tools/a16_xcheck.py`;
  references `luna-reference-solutions/{ramses,arepo}/cosmo_a16_xcheck/`, printed by `tests/test_cosmo_a16_xcheck.py`, never
  gated; runs, logs and `a16_xcheck.png` in `luna_tests/a16_xcheck/`). LUNA FP64 minmod on one A100.
  Errors against the exact answer at the snapshot's own a, interpolated between bracketing snapshots.
  Part A (8^3, z = 20 -> 0), max |p~/exact - 1| at a = 0.1 / 0.25 / 0.5 / 1:
  gamma 5/3: LUNA 1.8e-14 at all a; RAMSES 4.8e-8 (the float32 IC/output); Arepo 3.1e-12 .. 1.1e-11.
  gamma 7/5: LUNA 7.9e-15 .. 2.1e-14; RAMSES -0.447 / -0.734 / -0.847 / -0.912; Arepo -0.448 / -0.735 / -0.848
  / -0.9125, both equal to the prediction without the (3 gamma - 5) term, 1 - (a/a_i)^-(5 - 3 gamma)
  (0.4476 / 0.7346 / 0.8476 / 0.9125): both keep p~ constant, i.e. evolve T as for gamma 5/3. Code reason:
  mini-ramses `hydro/` has no a-dependent term (the super-comoving form is exact only for 5/3; the expansion
  dt cap is `pm/newdt_fine.f90:73`); Arepo stores Energy = a^2 m u + KE and takes u = (E/m - KE)/a^2
  (`hydro/update_primitive_variables.c:240-243`) with no expansion source, so a^2 u is constant. A known
  limitation of both for gamma != 5/3 (as LUNA had before `sources/comoving_energy.py`), not a LUNA merit:
  at gamma 5/3 all three agree to their precision. v~ and rho~ exact in all three (RAMSES 4e-8 float32).
  Part B (cold top-hat, equal cell count), signed core rho~/exact - 1 at a = 0.1 / 0.25 / 0.5 (turnaround) /
  0.65 (1 + delta = 19.9):
  64^3: LUNA +8.4e-4 / +4.4e-3 / +7.9e-2 / +0.38; RAMSES +2.3e-4 / +7.4e-4 / +2.4e-2 / +0.17; Arepo -4.2e-5 /
  -1.0e-4 / -3.2e-4 / -2.6e-3.  128^3: LUNA +1.9e-4 / +6.1e-4 / +2.9e-3 / +3.8e-2; RAMSES -4.7e-4 / -2.1e-3 /
  -5.9e-3 / -7.5e-3; Arepo -2.6e-5 / -9.8e-5 / -4.4e-4 / -2.0e-3. v~ slope: LUNA 3.9e-3 / 7.1e-3 / 6.9e-2 /
  0.25 (64^3), 9.6e-4 / 1.4e-3 / 3.3e-3 / 2.5e-2 (128^3); RAMSES 1.1e-2 .. 1.8e-2 at a <= 0.25 at both N;
  Arepo 2.7e-4 .. 4.9e-3. Reading: Arepo's Lagrangian cells follow the collapse (the 480 / 3648 core cells stay
  in the core while the grid codes' core shrinks to 24 / 181 cells at a = 0.65), so equal cell count is not
  equal core resolution; its 64 -> 128 gain at a = 0.65 is small (2.6e-3 -> 2.0e-3: dt or gravity-limited,
  not resolution). Among the grid codes, LUNA is ahead at a <= 0.5 at 128^3 but behind RAMSES at 1 + delta = 20
  (3.8e-2 vs 7.5e-3), and behind RAMSES at 64^3 at every a (3-4x), with RAMSES's step ~10x larger in dln a
  (33-41 steps to a = 0.65, the 0.1/H cap) and RAMSES's error negative at 128^3 (underdense) where LUNA's is
  positive. Not chased: why LUNA's minmod core overshoots more than RAMSES's (same limiter, same Riemann
  solver); RAMSES entropy + dual_energy changes its 128^3 error at 0.65 to -1.4e-2.
- A11 done: `tests/test_hydro_hard_kt_riemann.py`, Kurganov & Tadmor 2002 configurations 3 (four backward shocks)
  and 6 (four slip lines), data as printed (KT02 pp. 594, 596; LW03 Table 4.3), gamma 1.4, (0, 1)^2, T = 0.3,
  `BOUNDARY=1` (the papers state none; diode would block the supersonic inflow), IC `KT-Riemann-2D`
  (`src/luna/ic_generation/kt_riemann_2d.py`, refuses a split off a cell face, gamma != 1.4). No exact 2D solution:
  reference LUNA MC + HLLC FP64 at 8192^2 (4 A100, 2.6 / 1.8 min, gpudev jobs 31109275, 31109276; 4096^2 check runs
  job 31109274), `luna-reference-solutions/luna/kt_riemann/kt_riemann_c{3,6}_mc.npz` (rho on 256^2, provenance inside), `tools/kt_riemann.py`
  (run, pack, the exact 1D pieces from `luna.hydro.exact_riemann`; interface waves match the literature agent's
  independent solver to all printed digits, wall budget against brute-force quadrature to 1e-7). Finding: the density
  does not converge pointwise in a core box (config 3's jet roll-up, (0.05, 0.65)^2; config 6's vortex,
  (0.2, 0.7)^2), even block-averaged onto 64^2 or 128^2: the reference's own 4096 vs 8192 L1 (M = 128) is 2.35e-3 /
  4.53e-3, and MC at 1024, 2048, 4096 gives 3.65e-3, 2.43e-3, 2.35e-3 (config 3). Outside the core it converges.
  L1(rho), M = 128, against the reference (GPU job 31110810; CPU 256, 512 equal to the printed digits; config 3's
  split for the (0.05, 0.65) box job 31169083, verified by the test, GPU job 31169375):

  | N | c3 minmod total / outside core | c3 MC | c6 minmod | c6 MC |
  |---|---|---|---|---|
  | 256 | 7.87e-3 / 1.43e-3 | 5.72e-3 / 1.42e-3 | 2.99e-2 / 1.63e-2 | 1.86e-2 / 9.74e-3 |
  | 512 | 4.79e-3 / 6.99e-4 | 3.13e-3 / 6.89e-4 | 1.84e-2 / 9.90e-3 | 1.08e-2 / 5.18e-3 |
  | 1024 | 2.54e-3 / 3.66e-4 | 3.65e-3 / 3.40e-4 | 1.15e-2 / 5.75e-3 | 6.84e-3 / 2.55e-3 |
  | 2048 | 2.53e-3 / 2.03e-4 | 2.43e-3 / 1.92e-4 | 7.68e-3 / 3.17e-3 | 5.23e-3 / 1.23e-3 |
  | outside-core order 256-2048 (pairwise) | 0.94 (1.03, 0.93, 0.86) | 0.97 (1.04, 1.02, 0.83) | 0.79 | 1.00 |

  The config-3 core box was (0.15, 0.55)^2 first; there the reference's own error outside it (3.3e-4, in the shock
  and post-shock bands of interfaces 21/41) was 0.83 of the 2048 rung's and the fitted order moved 0.68-0.97 with
  the box (checker, critic_a11 O2). With (0.05, 0.65)^2 the reference's 4096 vs 8192 difference outside the core is
  1.09e-4, the 2048 rungs sit 1.8x above it, and the order is ~1 (expected for shocks). The core itself stalls (MC
  ~2.2e-3 from 2048 to 4096): roll-up that does not converge pointwise. Config 6's slip lines are Kelvin-Helmholtz
  unstable: the reference is the laminar self-similar solution, code-independent (AthenaK PLM 4096 unseeded is
  2.75e-4 from LUNA MC 4096 outside the core), held by the unperturbed 8192 run, but a 1e-15 seed rolls it up (4.1e-3
  outside the core, 7x its self-difference; critic_a11 O3). The test guards it: the npz stores a 4096 run seeded 1e-12
  (job 31169083; 6.16e-4 from the reference outside the core, unseeded 6.11e-4; bound 8e-4, a rolled-up reference
  would sit ~4e-3 away). The rungs <= 2048 are seed-insensitive (a 1e-6 seed moves MC 2048 from 1.23e-3 to 1.28e-3).
  Config 6 pressure strips (exactly 1): L1 |p - 1| 7.5e-4 / 9.0e-4 (MC 256, strips 21 / 34) .. 1.8e-4 / 2.2e-4
  (MC 2048), minmod ~1.8x; a spike of ~1e-2 rides each moving contact (8192: 5.6e-3 / 7.1e-3); teeth p4 = 1.01.
  CPU symmetry is FMA round-off (exact 0 with `--xla_cpu_max_isa=AVX`).

  Strips (the reference holds the 1D solution to 0.14-0.23 from the wall in config 3, 0.097 and 0.14 from the y
  walls in config 6; config 6's x walls are reached by 2D signals, as lit.md predicted): strip L1 vs exact 1D at
  2048 2.2e-4 (c3 21, minmod) .. 2.6e-3 (c6 34, MC), orders 0.84-1.2 (shocks) and 0.52-0.65 (slip lines), fronts
  within 0.31 cells (c3) and 0.70 cells (c6 minmod; MC 0.21). Config 3 diagonal asymmetry: exactly 0 on the GPU in
  FP64 and FP32 at every N; on the CPU round-off, 2.9e-13 (FP64) and 9.9e-6 (FP32) at 512. Wall budget vs exact:
  3.5e-5, 1.7e-5 (c3, 256, 512), 8.8e-4, 5.8e-4 (c6 at t = 0.25); own boundary-flux balance <= 4.1e-16. FP32 L1
  within 2.5e-5 of FP64 (512). 2 x 2 shards bitwise equal to one device (CPU). Teeth, each failing its bound at 512:
  first order (L1 8.45e-3 / 4.44e-2 vs bounds 6.2e-3 / 2.4e-2), rho3 = 0.183 (L1 6.7e-2, strips 34/32 6.4e-2),
  config 6 q2 <-> q4 (1.02) and v -> -v (0.92), split one cell off (asymmetry 0.36, fronts 0.80-0.82 cells), gamma
  5/3 (fronts 21-24 cells), `BOUNDARY=4` (wall budget 0.44 / 0.28 at 256). CPU file 202 s (tiers.json), GPU 103 s.
  Against AthenaK (2026-10-14, `luna_tests/lit_a11/athenak.md`): same IC (`luna_ic`), outflow walls, 64^2
  MeshBlocks, CFL 0.4, HLLC, dfloor = pfloor = 1e-8; PLM + RK2 256-2048, PPMX + RK4 256-8192 (4096, 8192 on 4 A100);
  `luna-reference-solutions/athenak/kt_riemann/a11_c{3,6}_{plm_rk2_2048,ppmx_rk4_8192}.npz`, `tools/athenak_refs.py prep|reduce|luna a11`; LUNA
  hllc-minmod/mc/vl (vl = AthenaK's PLM slope), CFL 0.8. Re-reduced after the core-box change (clean f8e0a6b; AthenaK
  reductions 31169848-55, LUNA 31169856). Test metrics, rho; config-3 core (0.05, 0.65)^2:

  | N | metric | LUNA minmod | LUNA MC | LUNA vL | AthenaK PLM | AthenaK PPMX |
  |---|---|---|---|---|---|---|
  | 1024 | c3 L1 outside core | 3.66e-4 | 3.40e-4 | 3.38e-4 | 3.26e-4 | 2.66e-4 |
  | 2048 | c3 L1 outside core | 2.03e-4 | 1.92e-4 | 1.90e-4 | 1.82e-4 | 1.60e-4 |
  | 256-2048 | c3 L1 outside core order | 0.94 | 0.97 | 0.94 | 1.05 | 1.02 |
  | 2048 | c3 L1 total (core-dominated) | 2.53e-3 | 2.43e-3 | 2.48e-3 | 2.33e-3 | 2.55e-3 |
  | 2048 | c3 strip 21 / 34 vs exact 1D | 2.22e-4 / 1.05e-4 | 2.30e-4 / 1.28e-4 | 2.26e-4 / 1.46e-4 | 4.98e-4 / 2.26e-4 | 4.90e-4 / 3.02e-4 |
  | 256-2048 | c3 strip 21 order | 1.09 | 0.99 | 1.00 | 0.71 | 0.60 |
  | 2048 | c3 diagonal asymmetry | 0 | 0 | 0 | 4.8e-12 | 0.43 |
  | 1024 | c6 L1 outside core | 5.75e-3 | 2.55e-3 | 3.20e-3 | 3.50e-3 | 1.70e-3 |
  | 2048 | c6 L1 outside core | 3.17e-3 | 1.23e-3 | 1.57e-3 | 1.72e-3 | 5.13e-3 (rolled up) |
  | 2048 | c6 strip 21 / 34 vs exact 1D | 2.70e-3 / 5.23e-3 | 1.34e-3 / 2.64e-3 | 1.57e-3 / 3.08e-3 | 1.77e-3 / 3.48e-3 | 8.67e-4 / 1.76e-3 |
  | 2048 | c6 pressure strips 21 / 34, L1 abs(p - 1) | 3.21e-4 / 3.69e-4 | 1.82e-4 / 2.21e-4 | 2.01e-4 / 2.41e-4 | 2.23e-4 / 2.65e-4 | 1.44e-4 / 1.84e-4 |
  | 512 | wall budget c3 / c6 | 1.7e-5 / 5.8e-4 | 1.5e-5 / 3.7e-4 | 1.5e-5 / 4.6e-4 | 2.4e-5 / 4.5e-4 | 2.0e-5 / 2.2e-4 |

  - Config 3 strips (LUNA HLLC vs AthenaK HLLC): LUNA has half AthenaK's error (MC / PLM 0.46, MC / PPMX 0.47 on 21).
    The error is AthenaK's HLLC on a slowly moving shock (critic `luna_tests/critic_a11/critic.md` O1): the strip's 1D
    Riemann problem alone gives it (AthenaK PLM 4.99e-4 in 1D vs 4.98e-4 in 2D), independent of CFL, integrator and
    jump placement; with `rsolver = hlle` AthenaK PLM converges at order ~1 and matches LUNA MC (2.20e-4 vs 2.32e-4 at
    2048, 1D). AthenaK HLLE matches LUNA; this is a Riemann-solver difference, not a scheme advantage.
  - Config 3 outside the core ((0.05, 0.65)^2 box): all five schemes converge at order ~1 (0.94-1.05, 256-2048);
    at 2048 MC / PLM 1.05, MC / PPMX 1.20 (as the critic measured), vl / PLM 1.04, minmod / PLM 1.12.
  - Config 3 symmetry: AthenaK PLM keeps it to round-off (<= 4.8e-12); AthenaK PPMX starts at round-off (4e-15 at
    t = 0.03) and jumps to 2e-5 by t = 0.06-0.12, then 0.43 at 2048, 0.54 at 8192. The break is in the PPMX
    reconstruction: bitwise the same on 64^2 and 256^2 MeshBlocks, PPMX + RK2 breaks it, PLM + RK4 does not (critic O4).
    LUNA: exactly 0 on the GPU, i.e. its arithmetic is axis-symmetric; a 1e-15 seed grows to 1.4e-9 at 2048.
  - Config 6: AthenaK PPMX is the best scheme to 1024 (outside core) and 2048 (strips). From 2048 its slip lines roll up
    (Kelvin-Helmholtz) and leave the laminar solution: outside-core L1 5.1e-3 (2048), 1.6e-2 (4096), 2.7e-2 (8192);
    strip fronts move 5.8 (4096) and 16 cells (8192). The laminar solution is code-independent: AthenaK PLM 4096 is
    laminar and 2.75e-4 from LUNA MC 4096 outside the core. Reference bias is <= ~15 %: against an AthenaK PLM 4096
    reference MC / PLM is 0.66 (0.73 against LUNA's) and MC / PPMX 1.75 (1.50) at 1024. LUNA stays laminar only
    because no seed is present: with a 1e-6 seed at 4096 MC departs more than AthenaK PLM (8.8e-3 vs 3.9e-3), and a
    1e-15 seed moves LUNA MC 8192 by 4.1e-3 (critic O3). With AthenaK's slope (vl) LUNA is 0.91x AthenaK PLM outside
    the core at 1024.
  - Equal dt (LUNA CFL 0.779 / 0.7385, 3976 / 3020 steps at 2048, jobs 31169857, 31169858): L1 changes by <= 1 %.
  - Cost at 2048: LUNA 11-12 s including JIT, AthenaK PLM 33 s on 128^2 MeshBlocks (bitwise identical to 64^2, 52 s):
    ~3x.
  - Reference cross-check (the open doubt): AthenaK PPMX 8192 vs LUNA MC 8192 outside the core on 128^2: 1.5e-4 (c3,
    new box; 4.3e-4 in the old one, mostly in the 21/41 shock bands where AthenaK carries its HLLC strip error),
    2.7e-2 (c6, rolled up); AthenaK 4096 vs 8192 1.6e-4 (c3); LUNA 4096 vs 8192 1.1e-4, 6.1e-4. Against the AthenaK
    PPMX 8192 reference the c3 ranking holds (2048: MC 2.35e-4, PLM 2.25e-4, PPMX 1.85e-4). For config 6 AthenaK PPMX
    8192 is no reference; the stored laminar LUNA reference stands, guarded by the seeded 4096 run.
  - Pressure strips (config 6, exact p = 1): every scheme converges; at 2048 MC / PLM 0.82-0.83, MC / PPMX 1.20-1.26.
- A12 parked (2026-10-08: the existing `test_hydro_accuracy_sedov.py` covers Sedov well). Kept:
  `tools/sedov_kt.py`, Kamm & Timmes 2007 (LA-UR-07-2849) for j = 1-3 and power-law ambient rho0 r^-omega
  (standard, singular, vacuum), checked against their Tables 1-9 and, independently of their formulas, by
  mass and energy of the profiles to 1e-8. Errors in the report: eq. 56 needs lambda^(j-1), not j+1; eqs. 58-60
  give the wrong singular alpha (eq. 54 integrated gives 8 C0 / ((gamma^2-1) (j+2-omega)^2 j) = their Table 6);
  Table 9's cylindrical r_v is lambda_v (r_v = 0.1156); Table 4's alpha is 1-3e-5 low (their E for the sphere
  equals the right alpha). KT ran only RAGE, 1D spherical, standard case (L1 rho 0.159 -> 8.8e-3 for 120 -> 3840
  cells, order ~1); nobody has published singular or vacuum runs. LUNA findings (Cartesian quadrant/octant, reflecting walls):
  a point blast with omega near j loads the origin cell with a finite share of the mass (converges as
  h^(j-omega)), so start from the exact solution at t0; then 2D standard and singular converge at order
  0.5-0.8 with the shock within 0.25-0.75 cells; the vacuum case aborts on negative pressure at N = 240 (HLL at
  120 too): no positivity fix on the pure-hydro path.
  Closed with this: a grid code cannot hold a true vacuum, and on LUNA's Cartesian grid the
  declining-density (singular) case adds little beyond the uniform Sedov test; revisit it if LUNA gets a
  cylindrical or spherical grid. 3D point blast, standard sphere, 120^3 octant vs KT's RAGE 1D spherical at
  120 cells (absolute L1): rho 0.126 vs 0.159, p 2.86e-3 vs 3.57e-3, v 1.05e-2 vs 8.70e-3; shock +0.25 cells.
- A17 done (2026-10-22): `tests/test_hydro_hard_keplerian_disc.py`, IC `Keplerian-Disc-2D`
  (`src/luna/ic_generation/keplerian_disc_2d.py`): a pressure-supported disc (cs0 = 0.05, power-law edges
  R_in = 0.5, R_out = 1.6, floor 1e-8) in the Plummer-softened point mass (eps = 0.2, `external_potential=point_mass`),
  an exact steady state (all gas rotates, Rayleigh stable, checked); box [-4, 4]^2, zero-gradient walls, HLLC, FP64.
  Drift from the exact cell averages after 10 orbits at R = 1, annulus 0.8-1.3 (A100, jobs 31387574/5/6, pytest
  31401962): minmod 256/512/1024/2048 rho L1 7.8e-2, 3.4e-2, 1.1e-2, 1.6e-3 (order 1.85), v_phi 2.2e-2, 4.6e-3,
  1.1e-3, 2.7e-4 (2.11), disc-band L_z -0.119, -0.050, -0.017, -0.0050 (1.52), box L_z -1.2e-2 .. -8.7e-4 (1.23);
  van Leer 512/1024/2048 rho 6.2e-3, 2.1e-4, 3.6e-5 (3.7, pre-asymptotic), v_phi 2.2e-4, 3.4e-5, 5.3e-6 (2.70),
  disc L_z -6.2e-3 .. -2.1e-4 (2.45), box L_z (2.61); box mass changes < 2e-5 (walls). Break-up (annulus rho L1 >
  10 %, 40 orbits, job 31402729): minmod 128 at 4.0 orbits, 256 at 13.5, 512 at 38.5; van Leer 512 none (rho 2e-2
  at 40). Failures: MC aborts (NaN dt) at every N, van Leer at 256 and in the cold disc (cs0 = 0.01) at 1024: the
  pressure goes negative inside the softening radius; minmod runs. Not C8: with the C8 face fix (93b1efd) the
  same runs still abort (job 31402991, 31402992); split step by step (job 31404344/5) the hydro update itself
  makes p < 0 (faces physical; the kick keeps p exactly) in disc-tail gas at R 0.12-0.15, rho 1e-7 .. 1e-6,
  Mach 15 (warm MC 256, step ~21) to 400 (cold MC 1024): E - KE cancellation, no positivity limiter or dual
  energy on this path. Strict xfail `test_kepler_mc_survives_one_orbit`. Setup notes:
  a static hot halo makes edge shear layers (unstable); at L = 5 or with a hot floor (P0 = 1e-7) the wall noise of
  the disc tail drives N-independent waves (rho L1 ~2e-3 .. 5e-3). CPU: 64, 128 one orbit + the xfail, 17.7 s (job 31404508; 4.9 s without the xfail on an earlier node).
  Rebased on main (C8 fix): GPU file 4 passed + 1 xfailed, same numbers (job 31404836). No AthenaK run:
  needs a luna_ic pgen extension (point-mass source + this IC). Scripts: `luna_tests/a17_kepler/scripts/`.
- Next in group A: A18, A2 (`docs/WORKPLAN_A.md`,
  Phase 2); A12 closed, A16 B done, A19 parked.
- GPU ladders: `tools/measure_gpu_ladders.py --out gpu_ladders.json` on viper measures the 2D tests at
  512-1024 (low Mach 128-512, Toro 1024-4096); bounds and the GPU test variants come from its output.
  2D problems without exact solutions (A5, A10, A11) get stored high-resolution LUNA references.

## Group A against AthenaK

Where LUNA stands on every group-A problem with an AthenaK comparison, at the finest N both codes ran. Same
initial data (pgen `luna_ic`), same metric, so the difference is the scheme. "Error" is per row: L1 against the
exact solution (A1, A3, A4, A6 (the drift from the exact steady state), A7, A13, A14, the A11 strips), the energy-budget error (A8, A6 energy), L1 against a
converged reference (A9; A11 outside the core, against LUNA MC 8192^2), the growth-rate error (A13 rate,
A14 rate), and for A15 the effective Reynolds number, where higher is better and the ratio is Re_eff(AthenaK) /
Re_eff(LUNA). A ratio below 1 means LUNA's error is smaller. Built by `python tools/athenak_summary.py` (gpudev jobs
31108338; A11 rows 31170008; A5 rows 31170615; A6 rows 31172489, `luna_tests/lit_a06/summary/`) from the stored files only; figure `code_paper_figs/athenak_group_a.png`. LUNA A1 and A15 were measured for this table (jobs 31108103, 31108104,
`tools/athenak_refs.py luna a1|tg`, clean 5d1c3cc); they reproduce the numbers quoted in the A1 and A15 entries. A7, A8 rows
added 2026-10-14; table and figure regenerated by job 31170217 (`luna_tests/a07_a08_ak/summary/`); the
figure leaves out the A8 row (an energy-coupling ratio of 6.5e4 would set its log axis).

| row | metric | N | LUNA minmod (default) | LUNA MC | LUNA van Leer | AthenaK PLM+RK2 | AthenaK PPMX+RK4 | AthenaK PPM4+RK4 | MC / PLM | MC / PPMX |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | RMS over variables of L1(U - U0) / EPS, sound wave after one period (AthenaK's `errs`), vs exact | 1024 (1D) | 4.5e-05 | 1.05e-05 | 1.5e-05 | 8.18e-05 | 6.44e-06 | not measured | 0.128 | 1.63 |
| A3 | L1(rho) vs exact cell averages, LW smooth advection, T = 4 | 1024^2 | 6.3e-05 | 1.23e-06 | 6.68e-06 | 9.39e-06 | 3.93e-11 | not measured | 0.131 | 3.14e+04 |
| A3 equal dt | as A3, LUNA at CFL 0.714 = AthenaK's step count | 1024^2 | not measured | 1.65e-06 | 6.92e-06 | 9.39e-06 | 3.93e-11 | not measured | 0.176 | 4.19e+04 |
| A4 static | L1 of abs(v - v0) vs exact, Gresho p(0) = 5, one revolution | 1024^2 | 0.000713 | 0.000133 | 0.00017 | 0.00017 | 7.46e-05 | 5.23e-05 | 0.783 | 1.78 |
| A4 moving | as A4 static, bulk flow one box per revolution | 1024^2 | 0.000833 | 0.000181 | 0.000233 | 0.000251 | 0.000135 | 0.00015 | 0.72 | 1.34 |
| A5 M | max over t <= 1.5 of abs(M / M_ref - 1), McNally KH mode amplitude vs the Pencil 4096^2 curve (fig. 4) | 2048^2 | 0.000829 | 3.78e-05 | 2.92e-05 | 4.11e-05 | 8.56e-05 | 8.45e-05 | 0.921 | 0.442 |
| A5 fields | L1(V_y) at t = 1.5 on 256^2 vs AthenaK PPMX + RK4 4096^2 (`luna-reference-solutions/athenak/mcnally_kh/a5_ppmx_rk4_4096.npz`) | 1024^2 | 0.000283 | 3.51e-05 | 3.81e-05 | 3.7e-05 | 2.33e-05 | 2.45e-05 | 0.949 | 1.51 |
| A6 | L1(rho(T) - rho(0)), drift from the exact hydrostatic state, isothermal atmosphere rho = p = exp(-phi), phi = sin 2 pi x, T = 2 / sqrt(gamma) | 2048 (1D) | 3e-07 | 2.23e-07 | 2.23e-07 | 2.04e-08 | 1.15e-07 | 1.17e-07 | 10.9 | 1.94 |
| A6 energy | abs(d(E + rho phi)) / (E + rho phi) at T, the static-potential energy budget | 2048 (1D) | 8.35e-07 | 1.72e-09 | 3.79e-09 | 1.75e-08 | 1.43e-11 | 3.05e-11 | 0.0982 | 120 |
| A9 | L1(rho) of the N x 16 strip vs reference (LUNA MC 2^16, `luna-reference-solutions/luna/woodward_colella/woodward_colella_mc.npz`), WC blast, T = 0.038 | 2048 | 0.0472 | 0.0272 | 0.0301 | 0.033 | 0.0201 | 0.0223 | 0.826 | 1.35 |
| A10 | L1(rho) vs reference (LUNA HLLC-MC 6400^2, `luna-reference-solutions/luna/lw_implosion/lw_implosion_mc.npz`), LW implosion, t = 0.045 | 800^2 | 0.000751 | 0.000439 | 0.000485 | 0.00054 | 0.000353 | 0.000349 | 0.813 | 1.24 |
| A11 c3 strip | KT config 3 (four shocks), T = 0.3: L1(rho) of the strip along the y = 1 wall (interface 21, backward shock) vs exact 1D cell averages | 2048^2 | 0.000222 | 0.00023 | 0.000226 | 0.000498 | 0.00049 | not measured | 0.462 | 0.469 |
| A11 c3 | KT config 3: L1(rho) outside the core box on 128^2 blocks vs reference (LUNA MC 8192^2, `luna-reference-solutions/luna/kt_riemann/kt_riemann_c3_mc.npz`), core (0.05, 0.65)^2 | 2048^2 | 0.000203 | 0.000192 | 0.00019 | 0.000182 | 0.00016 | not measured | 1.05 | 1.2 |
| A11 c6 strip | KT config 6 (four slip lines), T = 0.3: L1(rho) of the strip along the y = 0 wall (interface 34, slip line) vs exact 1D cell averages | 2048^2 | 0.00523 | 0.00264 | 0.00308 | 0.00348 | 0.00176 | not measured | 0.758 | 1.5 |
| A11 c6 | KT config 6: L1(rho) outside the core box on 128^2 blocks vs reference (LUNA MC 8192^2, `luna-reference-solutions/luna/kt_riemann/kt_riemann_c6_mc.npz`) | 1024^2 | 0.00575 | 0.00255 | 0.0032 | 0.0035 | 0.0017 | not measured | 0.729 | 1.5 |
| A13 | L1 of the odd part of v_y vs the exact linear eigenmode, RT, T fixed | 512 x 2048 | 0.000127 | 4.84e-06 | not measured | not measured | 0.000348 | not measured | - | 0.0139 |
| A13 rate | abs(sigma_meas / sigma - 1), RT growth-rate error | 512 x 2048 | 0.000257 | 7.47e-06 | not measured | not measured | 0.00029 | not measured | - | 0.0258 |
| A14 heavy-light | RM L1 of psi / psi0 vs exact (Laplace route), 2 <= tau <= 20 | 512 per wavelength | 0.0939 | 0.0611 | not measured | not measured | 0.0395 | not measured | - | 1.55 |
| A14 heavy-light rate | RM abs(rate error), fitted post-shock growth rate vs exact | 512 per wavelength | 0.064 | 0.0393 | not measured | not measured | 0.0316 | not measured | - | 1.25 |
| A14 light-heavy | RM L1 of psi / psi0 vs exact (Laplace route), 2 <= tau <= 20 | 512 per wavelength | 0.177 | 0.093 | not measured | not measured | 0.05 | not measured | - | 1.86 |
| A14 light-heavy rate | RM abs(rate error), fitted post-shock growth rate vs exact | 512 per wavelength | 0.0726 | 0.0367 | not measured | not measured | 0.0193 | not measured | - | 1.9 |
| A7 stable | L1 of (rho - exact) / AMP, Jeans wave n_J = 0.5, one period | 128 x 64^2 | 0.00457 | 0.000305 | 0.000579 | 0.00102 | 5.87e-07 | not measured | 0.299 | 519 |
| A7 unstable | L1 of (rho - exact) / AMP, Jeans wave n_J = 1.5, one e-fold | 128 x 64^2 | 0.00387 | 0.00225 | 0.00238 | 0.00131 | 0.000653 | not measured | 1.72 | 3.45 |
| A8 | abs(E(t) - E0) / abs(E0) at t = 3, periodic Evrard collapse (L = 4R), W from the 7-point periodic solve | 256^3 | 0.581 | 0.241 | 0.304 | 3.71e-06 | 4.48e-05 | not measured | 6.5e+04 | 5.39e+03 |
| A15 | Re_eff (higher is better): eps(t) matched to viscous spectral TG, M = 0.1; error ratio = Re_eff(AthenaK) / Re_eff(LUNA) | 64^3 | 61 | 189 | 120 | not measured | 550 | not measured | - | 2.91 |
| A15 | Re_eff (higher is better): eps(t) matched to viscous spectral TG, M = 0.1; error ratio = Re_eff(AthenaK) / Re_eff(LUNA) | 128^3 | 107 | 814 | 514 | not measured | >= 1600 | not measured | - | >= 1.97 |

Notes and sources (AthenaK: CFL 0.4, HLLC, FP64, one A100; LUNA: MUSCL-Hancock, HLLC, CFL 0.8, FP64, one A100):

- A1: Single sign of EPS (AthenaK's metric), so the EPS^2 nonlinear floor (~5.5e-6, PPMX's value from N = 128 on) is in every number; the PPMX ratio compares with that floor, not with a scheme error. LUNA CFL 0.8, AthenaK 0.4. Sources: minmod: `luna_tests/summary/luna_a1.json` `hllc-minmod_N1024.rms` (job 31108103); mc: `luna_tests/summary/luna_a1.json` `hllc-mc_N1024.rms` (job 31108103); vl: `luna_tests/summary/luna_a1.json` `hllc-vl_N1024.rms` (job 31108103); plm: `luna-reference-solutions/athenak/sound_wave/a1_plm_rk2_1024.npz` `N1024_errs_rms` (AthenaK jobs 31078027); ppmx: `luna-reference-solutions/athenak/sound_wave/a1_ppmx_rk4_1024.npz` `N1024_errs_rms` (AthenaK jobs 31078027).
- A3: LUNA CFL 0.8 (sum rule), AthenaK 0.4 (min rule). LUNA MC's error grows as dt shrinks (MUSCL-Hancock space-time cancellation), AthenaK's does not: see the equal-dt row. Sources: minmod: `luna_tests/lit_a03/luna_gpu.json` `hllc-minmod_N1024.L1` (job 31104179); mc: `luna_tests/lit_a03/luna_gpu.json` `hllc-mc_N1024.L1` (job 31104179); vl: `luna_tests/lit_a03/luna_gpu_vl.json` `hllc-vl_N1024.L1` (job 31107272); plm: `luna-reference-solutions/athenak/lw_smooth_advection/a3_plm_rk2_1024.npz` `N1024_L1` (AthenaK jobs 31104177); ppmx: `luna-reference-solutions/athenak/lw_smooth_advection/a3_ppmx_rk4_1024.npz` `N1024_L1` (AthenaK jobs 31104178).
- A3 equal dt: Critic run, job 31107131 (`critic_p1/jobs/luna.luna_extra.31107131.out`: 11896 steps vs AthenaK 11893); minmod not run at equal dt. Sources: mc: `luna_tests/critic_p1/out/luna_extra.json` `a3_hllc-mc-m_N1024.L1`; vl: `luna_tests/critic_p1/out/luna_extra.json` `a3_hllc-vl-m_N1024.L1`; plm: `luna-reference-solutions/athenak/lw_smooth_advection/a3_plm_rk2_1024.npz` `N1024_L1` (AthenaK jobs 31104177); ppmx: `luna-reference-solutions/athenak/lw_smooth_advection/a3_ppmx_rk4_1024.npz` `N1024_L1` (AthenaK jobs 31104178).
- A4 static: dt has no effect (LUNA MC at CFL 0.4: 1.329e-4). PPMX flattens from 512 (its extremum-preserving limiter); PPM4 also beats MC in rho and p here. Sources: minmod: `luna_tests/lit_a04/luna_gpu_minmod.json` `hllc-minmod_static_N1024.v_L1` (job 31104231); mc: `luna_tests/lit_a04/luna_gpu_mc.json` `hllc-mc_static_N1024.v_L1` (job 31104232); vl: `luna_tests/lit_a04/luna_gpu_vl.json` `hllc-vl_static_N1024.v_L1` (job 31107273); plm: `luna-reference-solutions/athenak/gresho_vortex/a4_plm_rk2_1024.npz` `static_N1024_v_L1` (AthenaK jobs 31104225); ppmx: `luna-reference-solutions/athenak/gresho_vortex/a4_ppmx_rk4_1024.npz` `static_N1024_v_L1` (AthenaK jobs 31104226); ppm4: `luna-reference-solutions/athenak/gresho_vortex/a4_ppm4_rk4_1024.npz` `static_N1024_v_L1` (AthenaK jobs 31107271).
- A4 moving: In rho and p LUNA MC is below PPMX and PPM4 (1.49e-6 vs 1.91e-6 / 1.84e-6 rho L1). Sources: minmod: `luna_tests/lit_a04/luna_gpu_minmod.json` `hllc-minmod_moving_N1024.v_L1` (job 31104231); mc: `luna_tests/lit_a04/luna_gpu_mc.json` `hllc-mc_moving_N1024.v_L1` (job 31104232); vl: `luna_tests/lit_a04/luna_gpu_vl.json` `hllc-vl_moving_N1024.v_L1` (job 31107273); plm: `luna-reference-solutions/athenak/gresho_vortex/a4_plm_rk2_1024.npz` `moving_N1024_v_L1` (AthenaK jobs 31104225); ppmx: `luna-reference-solutions/athenak/gresho_vortex/a4_ppmx_rk4_1024.npz` `moving_N1024_v_L1` (AthenaK jobs 31104226); ppm4: `luna-reference-solutions/athenak/gresho_vortex/a4_ppm4_rk4_1024.npz` `moving_N1024_v_L1` (AthenaK jobs 31107271).
- A5 M: AthenaK runs chained per output interval (75 runs of 0.02 from the previous final dump) so M is read at t = 0.02 k exactly. LUNA CFL 0.8 takes 0.90x AthenaK's steps; at equal dt (LUNA CFL 0.72) M moves by <= 5 % (`lit_a05/luna_a5_cfl0.72.json`), at CFL 0.4 MC's 1024 value by +19 %. MC / PPMX is 0.24-0.53x at 512-2048 in every metric (A5 critic); against the PLM family the ranking depends on metric and CFL. Point-sampled IC (the paper's): cell averages raise every code's M error 1.7-2.7x. AthenaK PPMX 4096^2 is 2.1e-5 from fig. 4, its order-2 Richardson value from 2048 and 4096 1.2e-6 (LUNA MC 4096^2: 1.1e-5). Sources: minmod: `luna_tests/lit_a05/luna_a5.json` `minmod_N2048.eM_max` (job 31170409); mc: `luna_tests/lit_a05/luna_a5.json` `mc_N2048.eM_max` (job 31170409); vl: `luna_tests/lit_a05/luna_a5.json` `vanleer_N2048.eM_max` (job 31170409); plm: `luna-reference-solutions/athenak/mcnally_kh/a5_plm_rk2_2048.npz` `N2048_eM_max` (AthenaK jobs 31166704, 31167253, 31167254); ppmx: `luna-reference-solutions/athenak/mcnally_kh/a5_ppmx_rk4_4096.npz` `N2048_eM_max` (AthenaK jobs 31166777, 31166981, 31166982, 31166983, 31166984); ppm4: `luna-reference-solutions/athenak/mcnally_kh/a5_ppm4_rk4_2048.npz` `N2048_eM_max` (AthenaK jobs 31166781, 31166985, 31166986).
- A5 fields: The test's field_errors with the AthenaK reference in place of the LUNA MC 4096^2 one; the two 4096^2 references differ by L1(V_y) 2.3e-6 on 256^2, AthenaK 4096 vs its 2048 by 4.5e-6. L1 orders 64-1024 are ~2 for every scheme against either reference. MC / PPMX is 1.3-1.5x in V_y, 1.5-2.4x in rho (L1, any reference or grid), 3-4x in L2 and Linf. Sources: minmod: `luna_tests/lit_a05/luna_a5.json` `minmod_N1024.vy_L1_ak` (job 31170409); mc: `luna_tests/lit_a05/luna_a5.json` `mc_N1024.vy_L1_ak` (job 31170409); vl: `luna_tests/lit_a05/luna_a5.json` `vanleer_N1024.vy_L1_ak` (job 31170409); plm: `luna-reference-solutions/athenak/mcnally_kh/a5_plm_rk2_2048.npz` `N1024_vy_L1_ak` (AthenaK jobs 31166704, 31167253, 31167254); ppmx: `luna-reference-solutions/athenak/mcnally_kh/a5_ppmx_rk4_4096.npz` `N1024_vy_L1_ak` (AthenaK jobs 31166777, 31166981, 31166982, 31166983, 31166984); ppm4: `luna-reference-solutions/athenak/mcnally_kh/a5_ppm4_rk4_2048.npz` `N1024_vy_L1_ak` (AthenaK jobs 31166781, 31166985, 31166986).
- A6: Neither code is well balanced; both add rho g at the cell centres (LUNA Strang-split around MUSCL-Hancock, AthenaK in every RK stage, pgen luna_ic_a6). AthenaK PLM converges at order 2.9 (128-1024; 2.6 at 512-2048), every LUNA limiter and AthenaK PPMX/PPM4 at 2.0. AthenaK's drift does not depend on dt: at LUNA's dt (CFL 0.8, LUNA's step counts) PLM gives 2.07e-08, PPMX 1.15e-07. LUNA's does: MC at 1024 8.8e-7 at CFL 0.8, 6.8e-7 at 0.4, 3.8e-7 at 0.2 (`lit_a06/luna_a6_cfl.json`, job 31172324), still 3.4x AthenaK PLM at 0.2. Cost is launch latency at 2048 cells in 1D (AthenaK PLM 0.48 ms per step). Sources: minmod: `luna_tests/lit_a06/luna_a6.json` `hllc-minmod_N2048.rho_L1` (job 31172111); mc: `luna_tests/lit_a06/luna_a6.json` `hllc-mc_N2048.rho_L1` (job 31172111); vl: `luna_tests/lit_a06/luna_a6.json` `hllc-vl_N2048.rho_L1` (job 31172111); plm: `luna-reference-solutions/athenak/isothermal_atmosphere/a6_plm_rk2_2048.npz` `N2048_rho_L1` (AthenaK jobs 31171978); ppmx: `luna-reference-solutions/athenak/isothermal_atmosphere/a6_ppmx_rk4_2048.npz` `N2048_rho_L1` (AthenaK jobs 31172018); ppm4: `luna-reference-solutions/athenak/isothermal_atmosphere/a6_ppm4_rk4_2048.npz` `N2048_rho_L1` (AthenaK jobs 31172022).
- A6 energy: A truncation drift in both codes (g at cell centres, not the discrete gradient of phi). AthenaK PPMX sits at 1e-11..1e-14 from 256 on, not a converging series; LUNA MC, van Leer and AthenaK PLM converge at 3.0, PPM4 at 4.0, minmod at 2.0. Sources: minmod: `luna_tests/lit_a06/luna_a6.json` `hllc-minmod_N2048.budget` (job 31172111); mc: `luna_tests/lit_a06/luna_a6.json` `hllc-mc_N2048.budget` (job 31172111); vl: `luna_tests/lit_a06/luna_a6.json` `hllc-vl_N2048.budget` (job 31172111); plm: `luna-reference-solutions/athenak/isothermal_atmosphere/a6_plm_rk2_2048.npz` `N2048_budget` (AthenaK jobs 31171978); ppmx: `luna-reference-solutions/athenak/isothermal_atmosphere/a6_ppmx_rk4_2048.npz` `N2048_budget` (AthenaK jobs 31172018); ppm4: `luna-reference-solutions/athenak/isothermal_atmosphere/a6_ppm4_rk4_2048.npz` `N2048_budget` (AthenaK jobs 31172022).
- A9: Reference choice: the LUNA MC 2^16 run (PPMX 16384 is 0.0019 from it at 1024 cells); against AthenaK PLM 16384 instead the ratios are given below. AthenaK PPMX/PPM4 need dfloor = pfloor = 1e-8 (default FLT_MIN floors: NaN at the collision). Symmetry checks exactly 0 for all. Against AthenaK PLM 16384: MC 0.0223, PLM 0.0279, PPMX 0.0157; MC / PLM 0.799, MC / PPMX 1.42. Sources: minmod: `luna_tests/lit_a09/luna_gpu_v2.json` `hllc-minmod_N2048.L1_rho_strip` (job 31107274); mc: `luna_tests/lit_a09/luna_gpu_v2.json` `hllc-mc_N2048.L1_rho_strip` (job 31107274); vl: `luna_tests/lit_a09/luna_gpu_v2.json` `hllc-vl_N2048.L1_rho_strip` (job 31107274); plm: `luna-reference-solutions/athenak/woodward_colella/a9_plm_rk2_16384.npz` `N2048_r_L1_rho_strip` (AthenaK jobs 31104233); ppmx: `luna-reference-solutions/athenak/woodward_colella/a9_ppmx_rk4_16384.npz` `N2048_r_L1_rho_strip` (AthenaK jobs 31107269); ppm4: `luna-reference-solutions/athenak/woodward_colella/a9_ppm4_rk4_16384.npz` `N2048_r_L1_rho_strip` (AthenaK jobs 31107270).
- A10: About 0.8x PLM and 1.2x PPMX with a ~5 % reference-bias uncertainty (against the exact planar solution, critic_a10: MC / PLM 0.72, MC / PPMX 1.20). LUNA MC reaches PPMX's 800^2 error at N ~ 970-1020 (estimate). Equal dt (LUNA CFL 0.614, 978 vs 980 steps, job 31169579): MC 4.49e-4. AthenaK's no-FMA PPMX 6400^2 reference is 5.9e-5 (L1) from LUNA's (self-convergence 8.0e-5 and 8.7e-5); against it the ratios are given below. AthenaK dfloor = pfloor = 1e-8; its default (FMA) build breaks the diagonal mirror symmetry from round-off, the -fmad=false build is exact, as LUNA. Against AthenaK PPMX 6400^2 (-fmad=false): MC 0.000444, PLM 0.000545, PPMX 0.000357; MC / PLM 0.815, MC / PPMX 1.24. Sources: minmod: `luna_tests/lit_a10/athenak/luna_hllc-minmod.json` `hllc-minmod_N800_e.L1` (job 31169576); mc: `luna_tests/lit_a10/athenak/luna_hllc-mc.json` `hllc-mc_N800_e.L1` (job 31169577); vl: `luna_tests/lit_a10/athenak/luna_hllc-vl.json` `hllc-vl_N800_e.L1` (job 31169578); plm: `luna-reference-solutions/athenak/lw_implosion/a10_plm_rk2_1024.npz` `N800_e_L1` (AthenaK jobs 31110999, 31165952, 31169072, 31169394, 31169395); ppmx: `luna-reference-solutions/athenak/lw_implosion/a10_ppmx_rk4_6400.npz` `N800_e_L1` (AthenaK jobs 31111000, 31111002, 31165951, 31165952, 31167428, 31169076, 31169396, 31169397); ppm4: `luna-reference-solutions/athenak/lw_implosion/a10_ppm4_rk4_1024.npz` `N800_e_L1` (AthenaK jobs 31111001, 31165952, 31169075, 31169398, 31169399).
- A11 c3 strip: Exact 1D, no reference involved. LUNA HLLC vs AthenaK HLLC: AthenaK's error is its HLLC on a slowly moving shock (the 1D Riemann problem alone gives it; order ~0.2). AthenaK PLM with HLLE matches LUNA MC (1D, 2048: 2.20e-4 vs 2.32e-4; `luna_tests/critic_a11/critic.md` O1). Sources: minmod: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c3_hllc-minmod_N2048.strip21_L1` (job 31169856); mc: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c3_hllc-mc_N2048.strip21_L1` (job 31169856); vl: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c3_hllc-vl_N2048.strip21_L1` (job 31169856); plm: `luna-reference-solutions/athenak/kt_riemann/a11_c3_plm_rk2_2048.npz` `c3_N2048_strip21_L1` (AthenaK jobs 31111397); ppmx: `luna-reference-solutions/athenak/kt_riemann/a11_c3_ppmx_rk4_8192.npz` `c3_N2048_strip21_L1` (AthenaK jobs 31111398, 31165946, 31165947, 31165948).
- A11 c3: AthenaK PPMX/PLM with dfloor = pfloor = 1e-8. All five schemes converge at order ~1 (256-2048); the two 8192^2 references differ by 1.5e-4 here (LUNA's own 4096 vs 8192: 1.1e-4). Against AthenaK PPMX 8192^2: MC 0.000235, PLM 0.000225, PPMX 0.000185; MC / PLM 1.04, MC / PPMX 1.27. Sources: minmod: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c3_hllc-minmod_N2048.L1_out` (job 31169856); mc: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c3_hllc-mc_N2048.L1_out` (job 31169856); vl: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c3_hllc-vl_N2048.L1_out` (job 31169856); plm: `luna-reference-solutions/athenak/kt_riemann/a11_c3_plm_rk2_2048.npz` `c3_N2048_L1_out` (AthenaK jobs 31111397); ppmx: `luna-reference-solutions/athenak/kt_riemann/a11_c3_ppmx_rk4_8192.npz` `c3_N2048_L1_out` (AthenaK jobs 31111398, 31165946, 31165947, 31165948).
- A11 c6 strip: Exact 1D. AthenaK PPMX is best to 2048; from 4096 its slip lines roll up (Kelvin-Helmholtz): front offsets 5.8 cells (strip 21, 4096) and 16 cells (strip 34, 8192). Sources: minmod: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c6_hllc-minmod_N2048.strip34_L1` (job 31169856); mc: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c6_hllc-mc_N2048.strip34_L1` (job 31169856); vl: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c6_hllc-vl_N2048.strip34_L1` (job 31169856); plm: `luna-reference-solutions/athenak/kt_riemann/a11_c6_plm_rk2_2048.npz` `c6_N2048_strip34_L1` (AthenaK jobs 31111397); ppmx: `luna-reference-solutions/athenak/kt_riemann/a11_c6_ppmx_rk4_8192.npz` `c6_N2048_strip34_L1` (AthenaK jobs 31111398, 31165946, 31165947, 31165948).
- A11 c6: The reference is the laminar solution, code-independent (AthenaK PLM 4096 is 2.75e-4 from LUNA MC 4096); against an AthenaK PLM 4096 reference the ratios move <= ~15 % (MC / PLM 0.66, MC / PPMX 1.75 at 1024). From 2048 AthenaK PPMX's slip lines roll up (L1 5.1e-3 at 2048, 2.7e-2 at 8192), so AthenaK PPMX 8192^2 is no reference for config 6. Sources: minmod: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c6_hllc-minmod_N1024.L1_out` (job 31169856); mc: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c6_hllc-mc_N1024.L1_out` (job 31169856); vl: `luna_tests/lit_a11/ak/out/luna_gpu.json` `c6_hllc-vl_N1024.L1_out` (job 31169856); plm: `luna-reference-solutions/athenak/kt_riemann/a11_c6_plm_rk2_2048.npz` `c6_N1024_L1_out` (AthenaK jobs 31111397); ppmx: `luna-reference-solutions/athenak/kt_riemann/a11_c6_ppmx_rk4_8192.npz` `c6_N1024_L1_out` (AthenaK jobs 31111398, 31165946, 31165947, 31165948).
- A13: AthenaK PLM and LUNA van Leer not run. LUNA values are the test's recorded GPU measurements. Sources: minmod: `tests/test_hydro_hard_rayleigh_taylor.py:117` (L1, measured, no job id); mc: `tests/test_hydro_hard_rayleigh_taylor.py:133` (L1, measured, no job id); ppmx: `luna-reference-solutions/athenak/rayleigh_taylor/rt_ppmx_rk4_512.npz` `N512_L1` (AthenaK jobs 31079751, 31079752, 31079753, 31079754).
- A13 rate: As A13. Sources: minmod: `tests/test_hydro_hard_rayleigh_taylor.py:118` (sigma, measured, no job id); mc: `tests/test_hydro_hard_rayleigh_taylor.py:134` (sigma, measured, no job id); ppmx: `luna-reference-solutions/athenak/rayleigh_taylor/rt_ppmx_rk4_512.npz` `N512_sigma` (AthenaK jobs 31079751, 31079752, 31079753, 31079754).
- A14 heavy-light: AthenaK PLM and LUNA van Leer not run; both codes converge at ~first order (vortex sheet). Sources: minmod: `tests/test_hydro_hard_richtmyer_meshkov.py:78` (L1, measured, no job id); mc: `tests/test_hydro_hard_richtmyer_meshkov.py:69` (L1, measured, no job id); ppmx: `luna-reference-solutions/athenak/richtmyer_meshkov/rm_ppmx_rk4_512.npz` `heavy-light_N512_L1` (AthenaK jobs 31078887, 31078888, 31078889, 31078890).
- A14 heavy-light rate: AthenaK PLM and LUNA van Leer not run; both codes converge at ~first order (vortex sheet). Sources: minmod: `tests/test_hydro_hard_richtmyer_meshkov.py:78` (rate, measured, no job id); mc: `tests/test_hydro_hard_richtmyer_meshkov.py:69` (rate, measured, no job id); ppmx: `luna-reference-solutions/athenak/richtmyer_meshkov/rm_ppmx_rk4_512.npz` `heavy-light_N512_rate` (AthenaK jobs 31078887, 31078888, 31078889, 31078890).
- A14 light-heavy: AthenaK PLM and LUNA van Leer not run; both codes converge at ~first order (vortex sheet). Sources: minmod: `tests/test_hydro_hard_richtmyer_meshkov.py:96` (L1, measured, no job id); mc: `tests/test_hydro_hard_richtmyer_meshkov.py:87` (L1, measured, no job id); ppmx: `luna-reference-solutions/athenak/richtmyer_meshkov/rm_ppmx_rk4_512.npz` `light-heavy_N512_L1` (AthenaK jobs 31078887, 31078888, 31078889, 31078890).
- A14 light-heavy rate: AthenaK PLM and LUNA van Leer not run; both codes converge at ~first order (vortex sheet). Sources: minmod: `tests/test_hydro_hard_richtmyer_meshkov.py:96` (rate, measured, no job id); mc: `tests/test_hydro_hard_richtmyer_meshkov.py:87` (rate, measured, no job id); ppmx: `luna-reference-solutions/athenak/richtmyer_meshkov/rm_ppmx_rk4_512.npz` `light-heavy_N512_rate` (AthenaK jobs 31078887, 31078888, 31078889, 31078890).
- A7 stable: AthenaK multigrid self-gravity (upstream c5a0d7f, binary build_gpu_lunaic_mg), CFL 0.3 (PLM + RK2 at 0.4 grows a 3D instability by 128); at LUNA's dt (CFL 0.8/3, `a7eqdt_*`) AthenaK PLM gives 0.00103, PPMX 5.92e-07. Sources: minmod: `luna_tests/a07_a08_ak/luna_a7.json` `hllc-minmod_stable_N128.L1` (job 31167105); mc: `luna_tests/a07_a08_ak/luna_a7.json` `hllc-mc_stable_N128.L1` (job 31167105); vl: `luna_tests/a07_a08_ak/luna_a7.json` `hllc-vl_stable_N128.L1` (job 31167105); plm: `luna-reference-solutions/athenak/jeans_wave/a7_plm_rk2_128.npz` `stable_N128_L1` (AthenaK jobs 31167512); ppmx: `luna-reference-solutions/athenak/jeans_wave/a7_ppmx_rk4_128.npz` `stable_N128_L1` (AthenaK jobs 31167513).
- A7 unstable: AthenaK multigrid self-gravity (upstream c5a0d7f, binary build_gpu_lunaic_mg), CFL 0.3 (PLM + RK2 at 0.4 grows a 3D instability by 128); at LUNA's dt (CFL 0.8/3, `a7eqdt_*`) AthenaK PLM gives 0.00117, PPMX 0.000653. Sources: minmod: `luna_tests/a07_a08_ak/luna_a7.json` `hllc-minmod_unstable_N128.L1` (job 31167105); mc: `luna_tests/a07_a08_ak/luna_a7.json` `hllc-mc_unstable_N128.L1` (job 31167105); vl: `luna_tests/a07_a08_ak/luna_a7.json` `hllc-vl_unstable_N128.L1` (job 31167105); plm: `luna-reference-solutions/athenak/jeans_wave/a7_plm_rk2_128.npz` `unstable_N128_L1` (AthenaK jobs 31167512); ppmx: `luna-reference-solutions/athenak/jeans_wave/a7_ppmx_rk4_128.npz` `unstable_N128_L1` (AthenaK jobs 31167513).
- A8: Not a scheme-accuracy row: AthenaK couples gravity to the energy through the mass fluxes (Mullen, Hanawa & Gammie 2020), LUNA by a cell-centred kick; floors 1e-8, CFL 0.3. Left out of the figure (MC / PLM 6.5e4 would set its log axis). Sources: minmod: `luna_tests/a07_a08_ak/luna_a8_all.json` `hllc-minmod_N256.dE_t3` (job 31167760); mc: `luna_tests/a07_a08_ak/luna_a8_all.json` `hllc-mc_N256.dE_t3` (job 31167760); vl: `luna_tests/a07_a08_ak/luna_a8_all.json` `hllc-vl_N256.dE_t3` (job 31167760); plm: `luna-reference-solutions/athenak/evrard/a8_plm_rk2_256.npz` `N256_dE_t3` (AthenaK jobs 31167525, 31167787, 31167791); ppmx: `luna-reference-solutions/athenak/evrard/a8_ppmx_rk4_256.npz` `N256_dE_t3` (AthenaK jobs 31167526, 31167790, 31168615).
- A15: Reference range Re 25..1600: a value at 1600 is a lower bound (AthenaK from 128^3, LUNA MC at 256^3), so 64^3 is the finest N with both finite. AthenaK Re_eff reduced by job 31083046 from the K(t) in `luna-reference-solutions/athenak/taylor_green/tg_ppmx_rk4_256.npz`. Sources: minmod: `luna_tests/summary/luna_tg.json` `minmod_N64.Re_eff` (job 31108104); mc: `luna_tests/summary/luna_tg.json` `mc_N64.Re_eff` (job 31108104); vl: `luna_tests/summary/luna_tg.json` `vl_N64.Re_eff` (job 31108104); ppmx: `luna_tests/phase0/tg_ppmx_rk4_reeff.npz` `N64_Re_eff` (AthenaK jobs 31078275, 31078276).
- A15: Reference range Re 25..1600: a value at 1600 is a lower bound (AthenaK from 128^3, LUNA MC at 256^3), so 64^3 is the finest N with both finite. AthenaK Re_eff reduced by job 31083046 from the K(t) in `luna-reference-solutions/athenak/taylor_green/tg_ppmx_rk4_256.npz`. Sources: minmod: `luna_tests/summary/luna_tg.json` `minmod_N128.Re_eff` (job 31108104); mc: `luna_tests/summary/luna_tg.json` `mc_N128.Re_eff` (job 31108104); vl: `luna_tests/summary/luna_tg.json` `vl_N128.Re_eff` (job 31108104); ppmx: `luna_tests/phase0/tg_ppmx_rk4_reeff.npz` `N128_Re_eff` (AthenaK jobs 31078275, 31078276).

What it shows. LUNA's MC limiter has the smaller error than AthenaK PLM + RK2 on the pure-hydro problems A1, A3, A4, A9, A10:
MC / PLM 0.13 (A1), 0.13 (A3; 0.18 at equal dt), 0.78 and 0.72 (A4 static, moving), 0.83 (A9), about 0.8 (A10). Most of that is the
limiter, which AthenaK does not offer: with AthenaK's own slope (van Leer) LUNA is 1.00x (A4 static), 0.93x (A4
moving), 0.91x (A9), about 0.9x (A10) and 0.71x (A3; 0.74x at equal dt) of AthenaK PLM. A1 is the exception: LUNA van Leer is 0.18x
and even minmod 0.55x on this 1D linear wave (not analysed further). LUNA's default minmod loses to AthenaK PLM on every 2D
problem: 6.7x (A3), 4.2x and 3.3x (A4), 1.43x (A9), about 1.4x (A10). AthenaK's PPM family has the smaller error on the smooth
problems and the shocks: MC / PPMX 3.1e4 on the linear advection A3 (PPMX fourth order there), 1.78x and 1.34x on A4
(PPM4: MC / PPM4 2.5x static), 1.35x on A9, about 1.2x on the A10 implosion (PPM4 the same), 1.25-1.9x on A14, and AthenaK reaches 2.9x LUNA MC's Re_eff at 64^3
(A15). On A1 PPMX sits at the EPS^2 floor (1.63x is a floor, not a scheme gap). The one problem where LUNA has
the smaller error than PPMX is the RT eigenmode A13: MC 0.014x in L1 and 0.026x in the growth rate, minmod 0.37x
and 0.89x. A11 (Kurganov-Tadmor 3, 6): on config 3's exact shock strips LUNA HLLC has
about half the error of AthenaK HLLC (MC / PLM 0.46, MC / PPMX 0.47 at 2048), but AthenaK with HLLE matches LUNA: a
Riemann-solver difference, not a scheme advantage. Outside the core the schemes are close on config 3 (MC / PLM 1.05,
MC / PPMX 1.20), and on config 6 PPMX leads (MC / PPMX 1.5 at 1024) until its slip lines roll up from 2048; MC / PLM
0.73 there, van Leer / PLM 0.91, minmod / PLM 1.64. McNally KH (A5): LUNA MC has the smaller mode-amplitude
error than PPMX (0.24-0.53x at 512-2048 in every metric) and ties the PLM family at 1024, while PPMX has the
smaller field error (1.3-1.5x in V_y, 1.5-2.4x in rho, L1, at 1024). Self-gravity (A7, A8, upstream AthenaK multigrid): on the
stable Jeans wave MC is 0.30x AthenaK PLM (van Leer 0.57x) and PPMX ~500x below MC; on the growing mode AthenaK
PLM has the smaller error (MC 1.7x), LUNA's error there being temporal (MC 1.19e-3 at CFL 0.4, 7.5e-4 at 0.2);
on the Evrard energy budget AthenaK's flux-based gravity energy coupling holds abs(dE/E0) at 3.7e-6 (PLM,
256^3) against LUNA's 0.24 (MC): a coupling difference, not a scheme-accuracy one. Static potential (A6, isothermal
atmosphere, pgen `luna_ic_a6`): AthenaK keeps the hydrostatic state better, MC 11x PLM and 1.9x PPMX in the density
drift at 2048 (van Leer the same as MC; AthenaK PLM converges at 2.9, everything else at 2.0), and LUNA's drift falls
with dt while AthenaK's does not; in the E + rho phi budget MC is 0.10x PLM, PPMX holds it to ~1e-11. The fair summary is "competitive and differentiable":
second-order LUNA with MC at or below AthenaK's second-order scheme, and behind its PPM schemes except on A13.

## Self-similar verification ladder (from `~/Downloads/hydro tests.pdf`)

The memo is LLM chat output: every solution, exponent and parameter comes from the papers (Lazarus
1981, SIAM J. Numer. Anal. 18, 316; Ramsey, Kamm & Bolstad 2009, the Guderley problem revisited;
Velikovich, Giuliani & Zalesak 2018, JCP 374, 843; Coggeshall 1993; Coggeshall & Axford 1986; Jang,
Liu & Schrecker 2025, converging and converging/diverging shocks), never from the memo; its DOIs are
checked when each is implemented. LUNA is Cartesian without geometric source terms: cylindrical
problems run in 2D, spherical in 3D, so every radial test also measures grid anisotropy.

Have: Riemann problems, LeBlanc, Gresho and low Mach, Sedov 3D; Noh only inside the carbuncle test.
To add, in order:
1. Noh as its own exact test, 2D and 3D: L1/L2 of the radial profile, symmetry error, wall heating,
   shock position [IC: from the carbuncle test]. Cheap; the baseline for the rest.
   Done (`tests/test_hydro_hard_noh_multid.py`, A100 + CPU-8, FP64, CFL 0.8): quadrant / octant [0, 1.7]^d,
   BOUNDARY=2, inflow warmed from r = 1 and ramped to rest from 1.5 (a cold ramp goes to negative pressure,
   MC / van Leer even in a ramp warmed behind its edge); exact inflow checked on [r_s + 0.05, 0.3]
   (2nd order). L1 rho over r < 0.3, 2D N = 256 -> 2048: HLLC minmod 2.33e-2 -> 6.21e-3 (order 0.64), MC
   0.61, van Leer 0.60, HLL 1.82e-2 -> 2.23e-3 (1.01); 3D 64 -> 256: HLLC 7.90e-2 -> 2.48e-2 (0.84), MC 0.78,
   van Leer 0.79, HLL 0.94. L2 rho orders 0.3-0.55. Shock radius within 0.3-0.65 h of t/3 on average; HLLC runs
   ahead along the axes (1.5 h at 2048, plateau +3.5 % there) and its plateau angular std stalls at ~1.2 %
   (2D) / 3.5 % (3D), HLL 3e-4 / 5e-3 converging at order 1 (strict xfail per HLLC limiter, the carbuncle).
   Wall heating at the origin, rho / 16: HLLC minmod 0.78, MC 0.89, van Leer 0.85, fixed with N, extent
   7-10 / 1-7 / 2-9 cells (2 %); HLL 0.94 -> 0.97, a shallow dip over 11 cells. 3D rho / 64 at the origin
   at 256: 0.62 (minmod), 0.76 MC, 0.70 van Leer, 0.89 HLL; the 3D plateau is still 4 % low at 256 (first order), so no 3D extent.
   Diagonal mirror symmetry exact (2D GPU) / <= 2.6e-12; mass and energy to round-off.
2. Generalized Noh (Velikovich, Giuliani & Zalesak, JCP 374, 843-862 (2018), DOI checked; CC-BY open
   access, full text still to obtain): finite initial pressure and shock strength, nearly arbitrary
   EOS, cylindrical and spherical; isentropic expansion and compression, cavity collapse, accretion
   shock; three stages in one run (expansion, compression or cavity collapse, accretion-shock
   expansion), not five as the memo had; verified there with a high-order Godunov Eulerian code
   (abstract, via an archived page; the M0 parametrization is from Huete et al., JFM 927, A35 (2021)).
   Semi-analytic (ODEs in r/t): tabulate and interpolate per cell; start at t > 0, large domain or
   exact inflow at the edges. Companion: Velikovich & Giuliani, PRE 98, 013105 (2018). Highest
   practical value.
   Deep check: the gamma-law finite-pressure form is written out in Huete et al., JFM 927,
   A35 (2021), eqs. 2.4-2.11 (uniform rho_0, p_0, inflow -v_0 r-hat, M0 = v_0/c_0; upstream ODEs in
   r/(v_0 t), Rankine-Hugoniot onto a core at rest) [researcher read]. A pure-Python RK4 solver
   (scratchpad gnoh2.py, to move into tools/) reproduces classic Noh to 5 digits for gamma 5/3 and
   1.4, cylindrical and spherical (checked by Claude); at gamma 5/3, M0 = 1.5: cylindrical xi_s =
   0.77736, rho_s = 4.93870, p_s = 4.23486; spherical 0.97318, 7.15184, 7.27063 (derived, to check
   against Huete Fig. 2a / the JCP paper before gating). PRE 98, 013105 is EOS-focused (Al, Cu) and
   gives no three-stage ICs; the JCP three-stage test is still unread. Velikovich, Murakami, Taylor &
   Giuliani (Phys. Plasmas 2016, arXiv 1603.05149; checked) ran it in Athena on Cartesian grids from
   the exact state at t0 = 2: grid-seeded perturbations decay slower than the physical modes, so
   perturbation decay is an observation, not a gate. ExactPack's `noh` hard-codes the post-shock
   pressure as rho_0 u_0^2 4^g / 3, right only for gamma = 5/3 (checked: noh1.py:59-61); use
   (gamma-1)/2 rho_s u_0^2.
3. Guderley converging shock, 2D and 3D, initial state from the self-similar ODE solution with its
   eigenvalue alpha; R_s(t), fitted alpha, similarity collapse, the sonic point; continued through
   collapse to the reflected shock. GPU resolutions. Checked: Ramsey, Kamm & Bolstad,
   LA-UR-09-05853 (OSTI 970025), also Int. J. Comput. Fluid Dyn. 26, 79 (2012), verify RAGE (cell-
   centred Eulerian FV, 1D spherical) and find ~first order in L1 before focus, sublinear after;
   exponents to 9-10 digits for gamma = 1.4, 5/3, 2, 3, 6 (cyl., sph.), the literature agreeing only
   to 8-9 digits, hard below 1.4; a reflecting far wall sends a spurious rarefaction in, so compare
   only where it has not arrived (agent's reading of the preprint, to re-read when implementing).
   Jang, Liu & Schrecker, ARMA 249, 24 (2025; arXiv 2310.18483): existence for gamma in (1, 3],
   cylindrical and spherical, sonic degeneracy behind the shock; SIAM J. Math. Anal. 57, 190 (2025;
   arXiv 2403.12247): extension past collapse with a finite outgoing shock for gamma in (1, 3], unique
   with a physical state at the origin for gamma in (1, 5/3]. So gamma = 1.4 and 5/3.
4. Coggeshall smooth similarity solutions, 2D/3D: time-dependent smooth exact flows for the formal
   order, L∞ and anisotropy beyond the isentropic vortex and Gresho.
   Guderley ladder (provenance: C = checked by Claude, R = researcher read, U = unverified):
   G1 surrogate spherical shock tube, gamma 1.4, no exact initialization, shock at the centre at
      t ~ 14.05, run to 20 (Ramsey & Shashkov LA-UR-12-22751) [R]; runs now.
   G2 converging Guderley, 3D spherical and 2D cylindrical, gamma 1.4 and 5/3, exact self-similar
      start; R_s(t), fitted exponent, similarity collapse (Ramsey et al. tables [R], Lazarus 6.4/6.5
      via ExactPack [C]).
   G3 the sonic point behind the shock: position and the error there (Jang, Liu & Schrecker 2025) [C].
   G4 through collapse to the reflected shock, gamma in (1, 5/3] only [C]; against RAGE's 1D rates
      (~1 before focus, < 1 after) [R].
   G2-G4: GPU resolutions; exact inner state or a large domain with the edge excluded from norms.
   Done, G2 at gamma 1.4 (A18, `tests/test_hydro_hard_guderley.py`, exact `src/luna/hydro/exact_guderley.py`,
   2026-10-22, A100 FP64, HLLC, CFL 0.8):
   alpha by shooting through the sonic saddle: 0.835323191953 (cyl.), 0.717174501489 (sph.), 0.815624901431
   and 0.688376822923 at gamma 5/3, within 5e-10 of RKB Table 1 (0.6883740859, quoted in the A18 brief, matches
   no entry: the sphere at 5/3 is 0.688376823); R from Lazarus' adiabatic integral equals its ODE to 9e-9, the
   radial Euler residual is < 6e-8, ExactPack's state() agrees to 1e-8. Surrogate: the exact state at t = -1
   (R_s = 1) in [0, 2]^d, reflecting faces, run to R_s = 0.15; the far walls' rarefaction is bounded by the C-
   characteristic from r = 2 (>= 0.6 clear of the L1 band); the origin cell never sees the shock (A12).
   Cylindrical (2D quadrant), minmod, N = 64..2048 per side: alpha fitted from R_s(t) with t_c = 0: 0.8732,
   0.8460, 0.8376, 0.8366, 0.8357, 0.83559 (error 3.8e-2 .. 2.7e-4); t_c free 0.8229 .. 0.83572 (4e-4 at 2048,
   +1.2e-3 at 1024: the three-parameter fit is ill-conditioned at this level). MC at 2048 0.835399 (t_c = 0),
   van Leer 0.835460. L1 behind the shock (band [0.75, 1.5] R_s, over the post-shock values), minmod rho at
   R_s = 0.8 / 0.2: 0.0234 / 0.0915 (64) .. 0.00127 / 0.00462 (2048), orders 0.82-0.96 (256..2048, all of rho,
   u_r, p and 4 times); MC 0.000984 / 0.00312, van Leer 0.00104 / 0.00345 at 2048 (MC near the shock: the C8
   face fix on main may shift it). The shock sits 0.2-0.5 cells behind the exact radius at every N (the
   estimator on the exact cell averages: < 0.1), the wedge spread of R_s grows toward collapse to 0.6-0.8 cells
   (exact: < 0.5); the x <-> y mirror is round-off that the converging flow amplifies (5e-15 at 256, 1e-11 at
   2048); mass and energy to round-off. Spherical (3D octant), minmod, 64..256: L1 rho 0.0196 / 0.145 ..
   0.00652 / 0.0446; alpha (t_c = 0) 0.7161, 0.7081, 0.7109 (not monotone: the shock goes from 0.36 cells
   behind at R_s = 0.8 to 0.27 ahead at 0.15 at every N), t_c free 0.7364, 0.7260, 0.7214 (first order). Not
   done: gamma 5/3 runs, G3 (sonic point), G4 (through collapse), FP32. Teeth: the IC run at gamma 5/3 fits
   alpha 0.030 and L1 rho 0.239 at N = 64. pytest: CPU-8 3 passed, 4 GPU-only skipped, 9.7 s (job 31386045);
   A100 7 passed, 343 s (job 31384086).
5. Variable-density Guderley, rho_0 ~ r^omega (G5): the error map over (gamma, omega, m, dx), where the
   scheme converges and where it does not. Primary source to identify [U].
6. Perturbed self-similar states (a multidimensional perturbation of 2 or 3), then the Lazarus
   branches with secondary shocks: research benchmarks, after 1-5.
Radiation (M1/M2 exist): ExactPack `suolson` (Su & Olson: non-equilibrium Marshak wave, one-group
diffusion) and `radshocks` (equilibrium and non-equilibrium diffusion radiative shocks), where M1/M2
reach the diffusion limit; Coggeshall & Axford 1986 for one-temperature similarity flows. After the
hydro ladder. Later: cavity collapse (a vacuum free boundary: tests the floors, after the floor work). The memo covers no gravity; the
gravity list below stands. Lazarus 1981 (checked: SIAM J. Numer. Anal. 18, 316-371; erratum 19, 1090
(1982)): converging shocks and collapsing cavities, cylindrical and spherical, any gamma > 1,
nonanalytic branches with one or arbitrarily many secondary shocks; "the evidence for approach to a
unique self-similar solution is not convincing".

Heat conduction (the goal; LUNA has none yet; open: the module): its gate is ready
before the module, references generated from ExactPack, in order: `heat` (linear diffusion, 1-3D, planar/cylindrical/spherical) -> Coggeshall cog8-18 (nonlinear
conduction lambda = lambda_0 rho^alpha T^beta coupled to hydro, 1D radial) -> `rmtv` (Reinicke &
Meyer-ter-Vehn: a thermal front ahead of a hydro shock); multidimensional with nonlinear conduction:
Coggeshall & Meyer-ter-Vehn, J. Math. Phys. 33, 3585 (1992; checked), a 2D axisymmetric P2 flow and a
3D spiral.

## ExactPack (LANL, github.com/lanl/ExactPack, BSD-3, checked at 9bacc47, 2025-09-24)

A Python library (numpy, scipy) of exact and semi-analytic verification solutions. Use: generate
references offline with a `tools/` script into luna-reference-solutions (`*.npz`) with the ExactPack commit in the
metadata, as for `tools/hydro_reference.py`; no test-time dependency. Its solvers, read in the code:
- `riemann`: exact 1D Riemann problems. Independent cross-check of `luna.hydro.exact_riemann` on Toro
  1-5, LeBlanc, Einfeldt (cheap).
- `noh` (planar, cylindrical, spherical; any gamma): reference for ladder item 1 (Noh 2D/3D).
- `noh2`: Noh's uniform collapse (Noh 1987, p. 92): a smooth adiabatic compression; a smooth radial test.
- `sedov` (planar, cylindrical, spherical): cross-check our Sedov reference.
- `guderley`: converging shock, cylindrical and spherical; exponents interpolated from Lazarus 1981
  Tables 6.4 and 6.5 with the 1982 erratum (`interp_laz.py`), Ramsey et al.'s method (`ramsey.py`):
  the reference for ladder item 3.
- `cog1`-`cog21` (no cog15): Coggeshall 1991 radial (planar, cylindrical, spherical) solutions of Euler
  with radiation-diffusion heat conduction, not multidimensional. cog8-18 need conduction (lambda0,
  alpha, beta): not applicable. cog1-7 and cog19-21 have no conduction parameter; whether their heat
  flux vanishes is to check against Coggeshall 1991, and several need gamma < 1 or gamma = 5 (cog3,
  cog5, cog21). Usable ones are radial similarity flows, some with shocks (cog19-21).
- `riemann2D_2section_steadystate` (Glaz & Wardlaw 1985): a steady 2D Riemann problem; a 2D exact test.
- `suolson` (non-equilibrium Marshak wave), `radshocks` (equilibrium and non-equilibrium diffusion
  radiative shocks): radiation tests, after the hydro ladder, if M1/M2 match their models.
- With the conduction module: `heat`, `rmtv`, cog8-18 (see Heat conduction above).
- Not applicable: `nohblackboxeos` (general EOS), `blake`, `ep_piston` (solids), `dsd`, `kenamond`,
  `mader`, `ehep`, `sdrz` (explosives and burn).
This corrects ladder item 4: ExactPack's Coggeshall solutions are 1D radial, mostly with conduction.
The smooth multidimensional ones are in Coggeshall, LA-UR-93-2505 (1993, OSTI 10180432; read):
"both heat conduction and the energy source S are neglected"; solutions 1-7 2D axisymmetric, 8-11 2D
in (R, phi), 12 3D (gamma = 5/3). E.g. solution 8 (gamma = 2): rho = rho_0 (R/t)^2, u_R = R/(2t),
u_phi = v_0 R/t, T = (4 v_0^2 + 1)/(16 Gamma) (R/t)^2, polynomial in x, y. All are homologous
(u ~ r/t): the box edges must be held at the exact, time-dependent solution, the same boundary
infrastructure as DMR; rho = 0 on the axis in solution 8, so the domain avoids R = 0.
Coggeshall & Axford 1986 (Phys. Fluids 29, 2398) is 1D, one temperature (equilibrium diffusion, LTE),
not T_m != T_r; its piston-driven shock with a thermal precursor is a numerical example.

## On apudev (viper)

Every test, check, bench and smoke run goes to `apudev`; never JAX on the login node (XLA's CPU
backend dies there with `pthread_create() failed`, which looks like a code bug).

- Partition: `-p apudev --account=<your_account> --constraint=apu`, 15-minute cap, at most
  `--cpus-per-task=48 --gres=gpu:2` per node. Always `-N1 -n1`: without `-n1` srun starts two tasks,
  each running all of pytest and preallocating GPU memory. One node was bad: exclude it with `--exclude=<node>`.
- When apudev is backed up, a short single-node job goes to `-p apu --time=00:30:00`; long runs stay
  on `apu`.
- Environment: `module load rocm/7.2 gcc/14`, then one of three venvs (checked 2026-10-06):
  - `jax_env2` (jax 0.10.2): the nightly's and the one-GPU default.
  - `venv_jax_rocm7.2` (jax 0.10.0): every multi-GPU run with gravity. The sharded
    gravity hangs on step 2 on 2 GPUs under 0.10.1 and 0.10.2 and runs under 0.10.0 (jobs
    11952449/59/60, 2026-09-23); not re-checked since `AMD_SERIALIZE_COPY=3`.
  - `amuse_env010` (jax 0.10.0 + AMUSE): coupling tests and the parallel full
    suite below.

  Always with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `AMD_SERIALIZE_COPY=3` (see `docs/ENV.md`) and
  `LUNA_DATA` (chemistry).
- Backend: `tests/conftest.py` defaults to CPU with 8 virtual devices; the GPU runs set
  `JAX_PLATFORMS=rocm XLA_FLAGS="--xla_gpu_enable_command_buffer="` (as
  `submit/examples/nightly_tests.sbatch`). The two disagree; compare runs only under the same one and
  say which.
- One pytest process per file: a combined session on ROCm 7.2 can stall where each file alone passes.

One file on one GPU:

```bash
srun -p apudev --account=<your_account> --constraint=apu --exclude=<node> -N1 -n1 \
     --cpus-per-task=24 --gres=gpu:1 --time=00:15:00 bash -lc '
  module load rocm/7.2 gcc/14; source <jax_env2>/bin/activate
  export JAX_PLATFORMS=rocm XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
         XLA_PYTHON_CLIENT_PREALLOCATE=false AMD_SERIALIZE_COPY=3 LUNA_DATA=<data dir>
  cd <checkout> && python -m pytest tests/test_X.py -q'
```

Multi-device tests (`test_pm_blocks.py`, `test_cosmo_hydro_coupled.py`) need `--gres=gpu:2`; with
fewer devices they skip silently.

Full suite: ~48 min serial, so it runs in parallel on one apudev node (CPU backend, 8 workers, longest
file first from `tests/tiers.json`, a 14-minute per-file timeout):

```bash
git worktree add <scratch>/tree_<sha> <sha>
sbatch -J <name> --export=ALL,ROOT=<scratch>/tree_<sha>,OUT=<scratch> \
       submit/examples/parsuite.sbatch
# report: <scratch>/parsuite_<name>.json
```

Run it from a worktree of the exact commit, never the live checkout: the tests read the files, and a
branch switch or merge mid-run gave false failures (2026-09-30). A commit older than
`tools/parsuite.py` also needs `RUNNER=<checkout>/tools/parsuite.py` in `--export`. A `--jobs`
option in `tools/run_tests.py` would replace the runner.

## On viper (next GPU session)

- Full tier on the GPU (`python tools/run_tests.py --tier full --update-costs`), with `LUNA_DATA` set
  (chemistry) and AMUSE installed (coupling). 59 of 192 test files have no measured cost in
  `tests/tiers.json`: they run in the full and slow tiers but never in the fast
  tier (`UNKNOWN_COST = 1e9`, `tools/run_tests.py:47`), cheap guards such as `test_halo_local`
  included. Copy the re-timed costs from `tests/.costs.local.json` into `tests/tiers.json`.
- Re-measure the GPU ladders at powers of 2 with L2, HLL and conservation: `test_toro_1d_gpu`
  (800/1600/3200 -> 1024/2048/4096), `test_sod_3d_directions_gpu`, `test_smooth_ladder_gpu`.
- Pallas: does `enable_fp_fusion` reach Triton, and is the centred wall blast bitwise symmetric with it
  off (see Conservation and symmetry sweep).
- FP32 and mixed ladders (precision markers) at GPU resolutions.
- Evrard 64/128/256 (`test_evrard_energy_gpu`) and RT 128-512 (`test_rayleigh_taylor_linear_gpu`):
  measure, then bounds and orders.
- Rerun the gravity-related test files after the gravity-precision change (80 of 102 left; the list
  is every test file with SELF_GRAVITY or gravity in it; 22 passed on the laptop).
- High-resolution LUNA references for A5, A10, A11 (2D, no exact solution).
- Tier placement: A7 stable Jeans (~230 s per scheme) and A8, A13 at 128 belong in the full tier.

## Precision

One rule for the whole suite: every test runs in exactly one precision mode, declared once with
`@pytest.mark.precision("fp64" | "fp32" | "mixed")` (default fp64), set by an autouse fixture in
`tests/conftest.py` and restored after the test. fp64: x64 on, `enable_x64=True`; fp32: x64 off,
`enable_x64=False`; mixed: x64 on, `hydro_recon32=True` (FP32 reconstruction, FP64 fluxes and state).
`evolve` and `_run_short` refuse a cfg that disagrees with the active mode; tolerances follow the mode
(conservation 1e-13 fp64, 1e-5 fp32). Replaces the 128 import-time `jax_enable_x64` switches (global,
never restored) and the 21 module `_x64` fixtures; `cfg.enable_x64` alone was not honoured in tests.

## Conservation and symmetry sweep

Done: Linf (bound and order) on every smooth test: the smooth ladder (sound, advected
sound, Gresho, sine; not the square), the LeBlanc fan interior and the isentropic vortex (which also
gained L2 and HLL). Measured orders: minmod clips extrema, so Linf ~1.2-1.6 on waves, 0.72-0.85 on
Gresho (kinked profile), 1.86-1.98 in the fan interior, ~1.8 one-period on the vortex.

Every test checks what its setup conserves; one helper in `tests/_helpers.py` bounds mass,
momentum, energy and passive scalars for closed or periodic boxes, wired into every accuracy test.
- Mass, momentum, energy: round-off in closed/periodic boxes; flux budget for open boundaries.
- Angular momentum: truncation only in Cartesian FV; bound the drift and its order (Gresho,
  isentropic vortex); L = 0 by symmetry for a centred problem in walls.
- Passive scalars, species: sum(rho X) to round-off, X in bounds, fractions sum to 1; chemistry
  conserves H and He nuclei and charge.
- Precision ladders: an FP32 sound wave (as Cholla's `float32_sound_wave`) and the smooth ladders in
  fp32 and mixed mode (`@pytest.mark.precision`): where the FP32 round-off floor stops the order;
  passive scalars stay in [0, 1] (boundedness) in every mode.
- Entropy: non-decreasing globally and per cell in shock tubes; change converges to 0 in smooth
  isentropic flow.
- Gravity: periodic PM momentum (zero self-force); Evrard energy and Layzer-Irvine as budgets.
- AMR: conservation across levels (refluxing); CIC deposit mass exact.
- Radiation: gas + radiation energy.
- Symmetry: WC, Sedov and the centred wall blast stay mirror-symmetric to round-off. Not bitwise:
  every operator but HLLC is bitwise mirror-odd; HLLC is too when evaluated eagerly with S_M's
  numerator grouped and the mean star flux at S_M = 0, but under jit it breaks at 1e-15 (probe
 ): XLA CPU contracts a*b + c into an exactly rounded FMA (3000 of 3000 samples), with fast
  math off, at optimization level 0 and across `lax.optimization_barrier`; no flag in jaxlib 0.10
  turns it off. The core stays as is; bound 1e-13 (measured <= 6e-15).
  On viper: the fused Pallas kernel bypasses XLA fusion, but Triton contracts FMAs by default
  (`enable_fp_fusion`); check whether Pallas exposes the switch and whether the centred wall blast is
  bitwise symmetric with it off (Pallas scope: 3D, FP64, minmod, HLLC, periodic/zero-gradient).
- MHD, with the kernel: div B to round-off under UCT; helicities as budgets; Orszag-Tang.

## To add

Marked [IC] where an IC exists, [infra] where LUNA first needs new capability.

1D (cheap on GPU: resolution ladders to N ~ 1e5, FP32 and FP64):
- Done: LeBlanc as in LANL LA-14379 (`tests/test_hydro_hard_leblanc.py`, exact; minmod,
  MC, van Leer; L1 order ~0.9-1.0, positivity, contact overshoot 0.11-0.25) and Titarev-Toro
  (`tests/test_hydro_hard_titarev_toro.py`, vs `luna-reference-solutions/luna/titarev_toro/titarev_toro_mc.npz`, a 2^16 MC run
  checked against 2^15 by `tools/hydro_reference.py`; MC resolves the waves between 2048 and
  4096 cells, minmod beyond 4096).
- Finding: rarefaction fans from a Riemann problem at t = 0 converge at first order inside the
  fan (LeBlanc, Sod; every limiter), second order (1.9-2.5) when started from the exact solution
  at t > 0: the start-up error of the unresolved centred fan, not the sonic point. Shock-tube
  orders below 1 (Toro 1-5) partly come from this.
- Done: Einfeldt 1-2-0-3, 1-1-0-5, 1-1-2-5 (`tests/test_hydro_hard_einfeldt.py`, exact;
  1-1-2-5 on a 2D strip, v rides the contact; near-vacuum centre vs Liska-Wendroff and Guo), Shu-Osher
  (`tests/test_hydro_hard_shu_osher.py`, vs `luna-reference-solutions/luna/shu_osher/shu_osher_mc.npz`; IC as exact cell averages,
  shared with Titarev-Toro), Planar-Riemann-2D (`tests/test_hydro_hard_planar_riemann_2d.py`, exact;
  the 2D step on a y-uniform strip is bitwise the 1D step at equal dt).
- More from LA-14379 (exact): Peak, RCVCR / vacuum expansion, stationary contact, slow shock,
  stream collision.
- Done: Woodward-Colella (`tests/test_hydro_hard_woodward_colella.py`, vs
  `luna-reference-solutions/luna/woodward_colella/woodward_colella_mc.npz`, 2^16 MC, BOUNDARY=2; minmod and MC, L1 orders 0.67 and 0.81,
  mass and energy to round-off). Its first probe found the reflecting-wall leak (Fixed, above).
- Planar Noh: wall heating.
- Smooth problems to N = 65536, FP32 vs FP64: where the FP32 round-off floor stops the order.

2D:
- Done: slotted cylinder at GPU resolutions (`tests/test_advection_slotted_cylinder.py::test_gpu_ladder`,
  gpu tier): 512, 1024, 2048 after 1 and 10 turns (the paper runs 10 turns at N = 256), exact
  reference u(T) = u0. A100, FP64, CFL 0.8, L1 at 512 / 1024 / 2048 and the L1 order over them:
  minmod 1 turn 0.0581 / 0.0364 / 0.0227 (0.68), 10 turns 0.113 / 0.0783 / 0.0508 (0.57); MC 1 turn
  0.0289 / 0.0174 / 0.0107 (0.72), 10 turns 0.0587 / 0.0360 / 0.0223 (0.70); van Leer 1 turn
  0.0349 / 0.0203 / 0.0124 (0.75), 10 turns 0.0669 / 0.0412 / 0.0245 (0.73). L2 orders 0.33-0.41.
  As measured, not converging: Linf 0.77-0.85 at every N (the discontinuity is never resolved);
  maximum principle: minmod keeps 0 <= u <= 1; MC min down to -6.6e-3 and max - 1 up to 8.3e-3,
  growing with N after 1 turn; van Leer after 1 turn min -1.5e-9 (2048) and max - 1 1.1e-8 (1024),
  after 10 turns within 3.3e-12. Mass to round-off (<= 5.2e-13) while the field stays off the
  walls; otherwise outflow through the zero ghosts: minmod 2.8e-10 (512, 1 turn), 1.3e-4 / 4.7e-7 /
  2.2e-11 after 10 turns, MC 3.4e-11 (512, 10 turns). Cost, steps and A100 wall per run: 1 turn 4014 / 8035 / 16078
  steps, 0.16 / 1.3 / 9 s; 10 turns 40134 / 80347 / 160772 steps, 1.6 / 12.6 / 90 s (fits gpudev).
- Liska-Wendroff 2D Riemann configurations; Hui implosion (reflecting walls, symmetry);
  off-axis Sedov (isotropy).
- Done: off-axis Sedov (`tests/test_hydro_hard_sedov_offaxis.py`, exact Kamm 2000 closed form, now j = 1, 2, 3
  in `src/luna/hydro/exact_sedov.py`; cylindrical alpha 0.98385 / 0.56428 vs Kamm & Timmes 2007 0.9840 /
  0.5643). gamma 1.4, E = 1, p_bg = 1e-5, periodic unit box, T = 0.09 (R_s = 0.301); blast at (0.4637,
  0.5219), Gaussian deposit (sigma 0.004) as exact cell averages; analysis in 32 wedges about the true centre.
  A100 FP64 (jobs 31377015-17, 31377027, 31377019; pytest 31378415/16 GPU), 256..2048^2, off / on centre:
  minmod L1(rho)/rho2 in 0.4-1.2 R_s 0.0229 -> 0.00407 (order 0.83), MC 0.0194 -> 0.00286 (0.92), van Leer
  0.0203 -> 0.00318 (0.89); shock radius +0.95 / +0.59 / +0.69 cells at 2048 (constant in cells: first order);
  rms R(theta)/R_s at 2048: 2.3e-4 / 8.4e-5 / 9.6e-5. The offset costs nothing measurable: L1 off/on
  0.99-1.016, sR off/on 0.85-1.43 (no trend), m = 1 (centre drift) 0.003-0.007 cells = the estimator floor
  (0.0064 on an exactly isotropic field), mass/energy/momentum to round-off (<= 2.2e-16). Finding: minmod's
  m = 4 grid mode grows in cells, 0.037 -> 0.196 cells (256 -> 2048; ~1e-4 of the box, barely converging);
  van Leer 0.07-0.15, MC 0.011 at 2048 (estimator floor 0.003-0.007). 3D minmod (48 equal-area cones):
  128^3 / 256^3 sR 1.24e-3 / 8.9e-4 off, 1.12e-3 / 8.0e-4 on; L1 0.0327 / 0.0216 both. MC numbers predate
  the C8 MC face fix and may shift.
- Rayleigh-Taylor, rest of: 2D linear done (A13, Group A progress). Open: 3D single mode; the
- Rayleigh-Taylor, rest of: 2D linear done (A13, Group A progress). Open: the
  nonlinear LW run (T = 8.5, 128 x 512) as a conservation regression; seed-amplitude convergence.
  3D single mode done: `tests/test_hydro_hard_rayleigh_taylor_3d.py`. A13's stratification, g along x,
  height 2/3, all walls; v_x = W(x) cos(ky y) cos(kz z), |k| = 6 pi, in the minimal wall box; the exact
  compressible eigenmode (A13's shooting ODE with K^2 -> ky^2 + kz^2; at height 1 it gives A13's sigma to
  6e-14; sigma = 0.661767 here), seeded +-AMP as A13. Axis mode (5, 0) and oblique (3, 4) (a 3-4-5 triple:
  same |k|, same h) give the isotropy |sigma_obl / sigma_axis - 1|. FP64, A100 (jobs 31384675/6), 48, 96,
  192 cells per wavelength to T = 4: growth-rate error axis / oblique minmod 0.057/0.044, 0.016/0.012,
  0.0041/0.0034 (order 1.90/1.86), MC 0.0099/0.0086, 0.0010/0.0009, 1.5e-4/1.4e-4 (3.0); L1 minmod
  0.0316/0.0156 .. 0.0031/0.0015 (1.7), MC 0.0059/0.0030 .. 1.6e-4/8.2e-5 (2.6); isotropy minmod 0.014,
  0.0031, 7.3e-4 (2.1), MC 1.3e-3, 1.4e-4, 1.1e-5 (3.4): the oblique mode is the more accurate one, the
  gap closes faster than either error. E + rho g x order 2.0, background drift order 1 as A13, mass to
  round-off. (4, 3) is (3, 4) with y and z swapped to 6e-15 (round-off). CPU-8: 24 cells per wavelength
  to T / 4 (job 31383738). Teeth: stopped at t / 2 (CPU), g flipped (GPU). Bounds 1.3x in the file;
  file passes CPU-8 (job 31386293, 24 s) and A100 (job 31386294, 8 passed, 27 min).
- Richtmyer-Meshkov, 2D: a shock through a single-mode interface; the linear impulsive growth
  d eta/dt = k du A eta (Richtmyer 1960, post-shock A and eta; Meyer & Blewett 1972 for heavy to
  light), its convergence, the phase inversion for heavy to light; mirror symmetry; conservation or
  the exact flux budget (shock-tube ends). Formulas from the papers when implemented.
- Reshocked Richtmyer-Meshkov, 2D/3D: the same with a reflecting end wall sending the shock back
  through the mixing zone; mixing-width growth after reshock, its resolution dependence, HLL vs HLLC.
- Shock-driven RT with RM above; ablative RT needs conduction (the conduction gate).
- Cloud crushing (Klein, McKee & Colella 1994), 2D and 3D: a shock over a dense cloud traced by a
  passive scalar; cloud-crushing time t_cc scaling, mixed fraction and cloud mass against time, the
  resolution study in cells per cloud radius (known not to converge in detail: report and bound the
  integral measures); conservation and scalar conservation.
- Non-linear thin-shell instability: adiabatic colliding flows, rho = P = 1, gamma 5/3, M' = 2, interface
  x = 0.1 sin(2 pi y), -5 < x < 5 open, 0 < y < 1 periodic, to t = 1.2; their comparison is visual
  and self-convergence only. For LUNA: self-convergence of integral measures on a power-of-2 ladder,
  symmetry, conservation; needs per-axis boundaries. A cooling variant (Blondin & Marks 1996) after.
  "Valcke et al. 2010" is a KH-in-SPH paper, not an NTSI comparison.
- Moving Gresho: the Gresho vortex advected (add to `tests/test_hydro_hard_galilean.py`): the error
  against the static run, its N-dependence; angular momentum as in the smooth ladder.
  Done (with the low-Mach item below): `tests/test_hydro_hard_gresho_moving.py`, not in
  `hard_galilean` (own static runs, three limiters, GPU ladder). A4 already has p(0) = 5 moving along x.
- Shock-bubble (Haas & Sturtevant 1987; Quirk & Karni 1996) and 3D multimaterial collapse: the real
  problems have two gases with different gamma; LUNA has one gamma-law EOS plus passive scalars, so a
  multi-material EOS comes first (a feature, open). A one-gamma, density-contrast version of
  the shock-bubble can run now as a precursor.
- Shock-vortex interaction (in Liska & Wendroff 2003; reference: Inoue & Hattori 1999): an isentropic
  vortex through a stationary shock; the generated sound field against the reference; tests that shock
  capturing keeps smooth vorticity. Missing.
- Kelvin-Helmholtz, nonlinear: the smooth setup of McNally, Lyra & Passy (2012) with their converged
  reference (mode amplitude and kinetic energy against time); extends `test_hydro_accuracy_kh.py`,
  which already checks the linear growth of a smooth layer against `luna.hydro.exact_kh_linear`.
- Cloud crushing setup (G2), a LUNA regression only: `tests/test_hydro_hard_agertz_blob.py`, Agertz et al. (2007) sec. 4.1 (periodic
  {10, 10, 40} R, chi = 10, M = 2.7; no SPH noise in our IC: cell-averaged sphere). Cloud fraction (T < 0.9 T_ext,
  rho > 0.64 rho_cl) at 64, 128, 256 across, A100 FP64: at tau_KH minmod 0.37, 0.51, 0.57, MC 0.48, 0.56, 0.64
  (rising with N, MC above minmod, as their Fig. 8); gone by 1.75-2.25 tau_KH except 256 (minmod 0.015, MC 0.092 at
  2.5), the grid codes' result. Mass, energy, momentum to round-off; 256^3 x 1024 in 3 min (~800 Mcell/s).
  No analytic solution and no converged one after tau_KH (the break-up's phase changes with N, their Fig. 7): the
  bounds are LUNA's own runs (+-15 % to tau_KH), not a reference. To do: AthenaK from the same IC (pgen luna_ic)
  for the ablation phase (code to code, the difference falling with N) and a fixed destruction bound from the paper.
- NTSI setup from the code comparison of Valcke et al. (2010).
- Moving Gresho at M = 1e-2 with a bulk velocity >= c, at an angle to the grid (low Mach, Galilean,
  orientation together).
  Done: `tests/test_hydro_hard_gresho_moving.py`, gamma 5/3, bulk (30, 15) / T = 1.07 c along (1, 0.5),
  T = a quarter revolution, periodic (exact = IC); HLLC, CFL 0.8, FP64, one A100 (Raven), 128-1024:
  velocity L1 static / moving, minmod 0.0490 / 0.0876 .. 0.00327 / 0.00660 (orders 1.30 / 1.23), MC
  0.00772 / 0.0237 .. 0.000302 / 0.00139 (1.56 / 1.37), van Leer 0.0126 / 0.0319 .. 0.000494 / 0.00173
  (1.56 / 1.41). Moving / static grows with N (minmod 1.8 -> 2.0, MC 3.1 -> 4.6, vL 2.5 -> 3.5): the
  moving ladder converges slower; KE loss and core L_z loss ~2-4x static (minmod 1024: KE 0.0084 vs
  0.0038). Bounds ~1.3x measured, regression only. MC measured before the C8 face fix.
- Shell collapse: an exact homogeneous isentropic implosion (Kidder's solution, 1970s, citation to
  check) as the smooth counterpart of Guderley and cavity collapse, 2D/3D on the Cartesian grid.
Not added: "rotated Noh / rotated Gresho" (radial problems: the symmetry error already measures grid
orientation); rotated shocks are covered by Sod along x, y, z, xy, xyz. The LLM overview also had
LeBlanc's p and e mixed up (our IC follows LA-14379, checked) and put Noh's wall heating on LeBlanc.
Setups, references and published numbers for all of these from the papers when implemented (the
lists came from LLM summaries). A "full HEDP benchmark" is not a test target: not LUNA's science case.
- DMR (Woodward & Colella 1984) [infra: per-face and time-dependent boundaries]: L1/L2 vs a
  high-resolution own run, triple-point position, jet length.
- Forward-facing step [infra: obstacles]; Mach 80/2000 jets [infra: inflow].

Cooling and stratified (researched, notes in `research_notes/Hydro test suite literature catalogue/
cooling_stratified.md`; checked items marked):
- Kim & Ostriker 2015 SNR: terminal momentum ~2.95e5 Msun km/s n0^-0.16 (checked; 2.17e5 n0^-0.13 is
  p at shell formation), bound ~20 % at dx <= r_sf / 10; match their cooling normalisation
  (`cooling_ne_per_nH`, `cooling_mu`, `cooling_X_H`).
- Field 1965 thermal instability: isobaric sigma = (1 - alpha) / (gamma t_cool) for constant heating;
  bound against the dispersion relation (Field's paper itself not read: secondary, to check).
  Done: `tests/test_cooling_field_ti.py`. KI table (`cooling_ne_per_nH = 1`, mu 0.6, X_H 0.76) with constant heating,
  equilibrium at a table-bin midpoint, T0 = 998 K (alpha = 0.5922 from the interpolant, KI analytic 0.5922) and
  T0 = 80 K (alpha = 1.661, stable). Field's cubic in s = n tau, q = k c tau (tau = u / (n_H Gamma)):
  s^3 + alpha s^2 + q^2 s + q^2 (alpha - 1) / gamma = 0, re-derived and matching Klimchuk 2026 (arXiv:2602.10902)
  eq. 3; Field's own equation numbers not read (eq. 31 the isobaric limit per Klimchuk). Paper-independent check
  (`test_cubic_vs_linearised`): eigenvalues of the dimensional linearised 1D equations (rho1, v1, T1) with LUNA's
  own dT/dt(rho, T) differentiated by central differences equal the cubic's roots to <= 4.3e-9 (q = 0.3-100, 998 K
  and 79 K; job 31401004, CPU). 1D / 2D cooling equals the 3D embedding bitwise (after 6bcecbb). Limits checked: q = 1e4
  condensation root (1 - alpha) / gamma to 5e-9, q = 1e-5 isochoric root -alpha to 4e-10, small-q condensation
  q sqrt((1 - alpha) / (gamma alpha)). Exact eigenmode, amplitude 1e-4, periodic 3D grid one cell across (N x 1 x 1;
  N x N x 1 at angle atan 2), production step order in a jitted loop (equal to `evolve_staged` to 1.5e-15), rate
  fitted over 2 e-folds. A100 FP64, minmod, CFL 0.8, |s / s_exact - 1| at N = 16, 32, 64, 128, 256:
  q = 0.3: 2.8e-2, 1.9e-2, 1.1e-2, 5.7e-3, 2.9e-3; q = 1: 2.1e-2, 1.4e-2, 7.7e-3, 4.1e-3, 2.1e-3;
  q = 10: 3.3e-3, 2.0e-3, 1.1e-3, 5.6e-4, 2.8e-4; q = 100: 3.3e-4, 2.0e-4, 1.1e-4, 5.6e-5, 2.9e-5 (orders 0.81-0.88);
  stable decay q = 10 (N 16-128): 1.0e-2, 6.0e-3, 3.1e-3, 1.6e-3; angle (1, 2) q = 10 (N 32-128): 1.6e-3, 1.0e-3,
  5.9e-4; MC / van Leer at q = 10 within 15 % of minmod. The error is the time split: at equilibrium t_cool =
  T / |dT/dt| is infinite, so each cooling half-step is ONE forward-Euler step of dt / 2 whatever `cooling_dt_frac`
  (`src/luna/sources/cooling.py:274-279`). N = 64, CFL 0.8 -> 0.05: q = 0.3 (dt / tau 0.25 -> 0.016) 1.1e-2 -> 2.9e-4
  (order 1.28 in dt), q = 10 1.1e-3 -> 1.3e-5 (CFL 0.8 -> 0.1, order 2.07); the isochoric mode (q = 0.03, s =
  -0.590, decays at alpha / tau) 4.1e-2, 2.2e-2, 1.1e-2, 5.8e-3 at dt / tau 0.32 -> 0.04 (order 0.95). At fixed
  dt / tau ~ 1e-3 the spatial residual is -2.6e-3, -4.4e-4, +1.3e-5, +1.1e-4 at N = 16-128 (sign change). Teeth:
  dropping the first cooling half-step gives -0.50. Jobs 31383851, 31384025, 31386053; pytest job 31386440: A100 14
  passed + 2 xfailed (345 s), CPU-8 2 passed, 12 GPU-only skipped, 2 xfailed (10.8 s; the N = 16, q = 10 rate on 8
  shards equals the 1-GPU 3.280e-3).
  Finding: cooling is a silent no-op in 1D and 2D (reads U[1..3], U[4], writes U.at[4], the 3D layout,
  `cooling.py:233-234, 294`; nothing refuses it): strict xfail `test_cooling_in_1d_2d`. `evolve_staged` /
  production dispatch the cooling scan eagerly (job 31382945: 560 steps at 32 cells in ~260 s on an A100).
- Internal gravity waves in a walled isothermal layer (Dintrans & Brandenburg 2004): exact eigen-
  frequencies, e.g. omega = 0.572054 for (l, n) = (1, 0); the dynamics on top of the balance.

Open decisions:
- Remove the duplicated sound ladders now covered by A1 (`test_hydro_accuracy_smooth.py`,
  `test_regressions.py`).
- Production default limiter: minmod (LIMITER 0) is far behind MC on every smooth test (LW smooth 30x).
- `gravity_precision` is silently ignored for proper runs (`src/luna/setup/cosmo_common.py:50`): refuse the
  combination instead (standing rule: refuse, don't fall back).
- The joint verification table (35 claims, final section of the catalogue) still needs both checks.

Sinks and collapse:
- LUNA has no Bondi accretion mode: `ACCRETION_MODES = ("threshold", "federrath")`
  (`src/luna/species/sinks.py:21`), "bondi" is refused (`tests/test_sink_flash_rules.py:367`); sinks
  start inactive (`sinks.py:102-106`), so no sink can be placed at t = 0; eps = r_acc = 2 dx by
  default (`:63-64`). Bondi and Shu tests need a pre-placed sink (an IC or init option) first.
- Bondi (1952): lambda(gamma) from Bondi's Table I (isothermal e^1.5/4 = 1.12042; 4/3 0.707, 1.4
  0.625, 1.5 0.500, 5/3 0.250); Krumholz, McKee & Klein 2004 Table 1 gives Mdot errors vs r_B/dx
  (+1.1 % at 0.1 ... -24.4 % at 3.16, -2.3 % at 10); threshold accretion is off by orders of
  magnitude in the subsonic regime (Bleuler & Teyssier 2014), so with LUNA's modes a Bondi bound
  holds only where the flow is supersonic at r_acc. [researcher read; to check]
- Shu (1977): m0 = 0.975 (A = 2+); Krumholz et al. 2004 reach ~1 % in Mdot; Gong & Ostriker 2013 by
  figure. Federrath et al. 2010's m0 at A = 29.3 and 50 disagree with re-integration (138.9 and
  313.3 vs 133 and 279). [researcher read; to check]
- Larson-Penston: Hunter's constants (U(0) = 3.278, P(0) = 8.854, e^Q0 = 1.672); no published percent
  gate; needs ~12 decades of central density for the -3.28 c infall: AMR depth.

Isothermal and infall:
- Isothermal Riemann problems against the exact isothermal solver (no external reference exists:
  RAMSES `tests/hydro/isothermal` is a self-gravitating collapse with a checksum). Reference values
  recomputed by Claude with an independent solver (c = 1): iso-Sod (1, 0 | 0.125, 0) rho* =
  0.3457801278, u* = 1.061952175, S_R = 1.663202039; collision (1, +2 | 1, -2) rho* = 3 + 2 sqrt 2,
  S = +-0.41421; density ratio 1e4 rho* = 34.20882478; double rarefaction (1, -1 | 1, +1) rho* = 1/e.
  Runs through the full stage pipeline (the projection is a stage), with the exact budget.
- Bertschinger (1985) secondary infall, gamma 5/3 gas: lambda_s = 0.338976, inner limits D -> 2.60173
  lambda^(-9/4), P -> 2.40676 lambda^(-5/2), M -> 10.4069 lambda^(3/4) (researcher read, ADS scan;
  tables to transcribe); 3D verifications are by figure only (RAMSES 2002: "down to 2 fine cells").
  Pitfalls: seed mass, 3/5 delta for a Hubble-flow start, periodic images (keep r_ta << L), cold gas.
  Needs AMR depth for a quantitative 3D comparison.

Gravity solver literature (, researcher read, to check; notes gravity_solver_tests.md):
Enzo's orbit test is a direct clone target (32^3, test particle at 0.3 L, 200 orbits: energy scatter
0.276 % rms, 0.546 % max); PM pair force: mean force good beyond ~1.25-1.5 cells, CIC anisotropy large
below a cell (Efstathiou et al. 1985; RAMSES ~1 % at large r); Jiang et al. 2013 Table 2: source-term
self-gravity energy error ~O(1/N) (0.65 ... 0.05 from 22^2 to 362^2); Guillet & Teyssier 2011:
coarse/fine force L-inf first order on the level boundary, L2 ~ dx^1.6; Layzer-Irvine 0.5-2 % in
RAMSES (64^3-256^3), with Efstathiou's potential-energy correction for momentum-conserving PM;
Maclaurin per Ricker 2008 (O(dx^2) uniform), a uniform cube for a clean slope (Moon et al. 2019).
Gravity:
- PM zero self-force and exact momentum conservation (matched CIC deposit and interpolation).
- PM pair-force accuracy against 1/r^2 from sub-cell to half box, orientation scatter (anisotropy).
- Kepler / two-body orbits with LUNA's KDK: energy bounded and O(dt^2), angular momentum, phase.
- Self-gravitating gas: centre-of-mass drift in a periodic box (the source-term form is not exactly
  momentum-conserving; stress-tensor forms are), with the Evrard energy budget.
- Plummer and Hernquist spheres held over many dynamical times (Lagrangian radii, energy).
- Maclaurin spheroid potential (the IC exists) on the isolated and AMR solvers.
  Done (`tests/test_gravity_maclaurin.py`, FP64, A100): L1 order 2.0 at 32-256 (e=0.5: L1 1.2e-3 multipole faces,
  2.6e-4 analytic faces at 256); one-level AMR (eff. 64-256) order 2.0, L1 1.2e-3; isolated AMR via `solve_phi_hierarchy` only.
- Pulsating polytrope. Done: `tests/test_gravity_polytrope_pulsation.py`, n = 1 with Gamma_1 = 5/3, isolated
  self-gravity, flux energy coupling (auto), hllc-minmod, FP64, A100. The reference is the LAWE (Cox 1980 ch. 8) as
  linear finite elements (`tests/_polytrope_lawe.py`). It matches Pekeris (n = 0) to 1e-10 and Cox Table 8.1 (via
  Goodman's A403 notes) to 1.2e-3 (n = 1.5) and 1.8e-4 (n = 3); Omega^2 = 1.891565 at n = 1, inside Cox's bounds
  1 .. 1.913. No published n = 1 value was found. The perturbation is the LAWE fundamental mode (delta rho_c / rho_c =
  1e-2); the signal is perturbed minus unperturbed rho_c, fitted over 1.5 periods. Results at 64 / 128 / 256:
  - period error 2.5e-4 / 1.6e-3 / 6.6e-4, not monotone in N or in the window (64: 4.4e-3 over 1 period, 2.7e-2
    over 2), so no order;
  - |damping| / omega 6.5e-4 / 1.7e-3 / 3.5e-4, sign mixed, at the fit's noise;
  - unperturbed rho_c drift 3.3e-2 / 1.2e-2 / 4.1e-3, order 1.50;
  - |dE| / |W0| 5.8e-5 / 3.8e-5 / 2.0e-5 (kick: 5.7e-2 at 64 over 3 periods).
  Teeth: gamma = 1.55 gives a period error of 0.19, and that run lands within 2.8e-3 of the LAWE for 1.55.
  Open: after about 2 periods (t ~ 8, about 8 central crossings) rho_c climbs in both runs, +15 % at 64 and +30 % at
  128 by t = 12. It happens with kick too and with R = 0.25 (wider margin), so it is not the box. Not yet
  explained. A uniform hot ambient's pressure jump at R focuses a wave at the centre (+9 % at t = 1.5 at 64); the test
  uses P0 inside plus a hydrostatic isothermal atmosphere instead.
- AMR gravity at a coarse/fine boundary: force on a point mass or sphere near the edge.
- Layzer-Irvine as a budget on the DM growth runs.
- Poisson spatial convergence: power-of-2 ladder, L1/L2/Linf of phi and g, fitted order, periodic,
  Neumann (analytic) and isolated.
- Tasker et al. (2008) suite: the static and the translated King sphere (self-gravitating equilibrium,
  then Galilean); with the well-balanced work.
- Santa Barbara cluster (Frenk et al. 1999): gas + DM, profiles and global quantities against the
  12-code range (validation, no exact answer). Deferred: needs zoom ICs or deep refinement in the
  cluster; a uniform grid misses the core, and the comparison would measure that.
- Stratified and stellar low-Mach set of Wongwathanarat, Grimm-Strele & Müller (2016): hydrostatic
  equilibrium, Gresho, Taylor-Green, RT with scalar diagnostics.
- Poisson convergence ladders, periodic / Neumann / isolated, L1/L2/L∞ of phi and g.
- Kepler/two-body orbit energy over many orbits (KDK); PM momentum conservation.
- Evrard energy budget; Jeans growth rate vs linear theory with evolved gas;
  Maclaurin spheroid and Plummer vs analytic forces; Bonnor-Ebert held stable.

## Breakdown map

Per problem, a ladder over resolution x precision x limiter x Riemann solver, recording L1, L2,
orders, positivity margin and conservation, run on viper. The result is where LUNA breaks down:
input for the paper, and the first automated loop (`docs/history/` once run).
