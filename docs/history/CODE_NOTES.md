# Notes moved out of the code (2026-10-01)

Measurements and fix stories that used to live in docstrings and comments,
moved here when the docstrings were made lean. Dated numbers; the code and
the live docs win where they disagree.

## hydro

- **Barrier placement in the 3-D sweep** (`muscl_hancock._muscl_hancock_3d`).
  The y/z sweep inputs go through the `optimization_barrier` together with
  the x contribution. A barrier on dU alone let XLA fuse all three
  directions, keeping 27 face-sized arrays alive at once. Production step,
  FP64, one MI300A, bitwise equal: temporaries 46.3 -> 25.3 fields of N^3,
  512^3 Sedov 1.254 -> 1.259 Gcell/s. A barrier after y as well: 23.3 fields
  but -3.8%; a barrier only after y: 37.3 fields, +4.8%.
- **Kernel paths.** The fused concat-free 3-D step was +17% at 512^3 FP64 on
  MI300A. recon32 (FP32 reconstruction, FP64 fluxes) gave about 2x step speed
  at 512^3 FP64. Building the predictor one direction at a time saves about
  40 GB of working set at 1024^3 FP32.
- **A100 tuning, 2026-10-10** (Raven, one A100-SXM4-40GB, jax 0.10.1 with
  jax-cuda13 0.10.1).
  - Pallas launch shape matters more than anything else measured. The kernel
    alone at 256^3, default tile 1x4x64, Triton's default num_warps=4 against
    num_warps=8, bitwise equal: FP64 387 -> 1123 Mcell/s (2.9x), recon32
    1092 -> 1498 (1.37x), FP32 2583 -> 2876 (1.11x). The best tiles of the
    17-tile x {1,2,4,8}-warp sweep all give one cell per thread (1x2x64 w4,
    1x1x128 w4, 1x4x64 w8, 1x1x256 w8): FP64 1136, recon32 1515, FP32 2944.
    Two cells per thread is the slow case, worst in FP64; that points to
    register spills (twice the registers per value), not confirmed by
    counters: ncu cannot profile on Raven, DCGM holds the counters even
    after `hpcmd_suspend` on a shared node. num_warps=1/2 is 4-10x slower.
    The tile was chosen on MI300A (64-wide wavefronts): an NVIDIA rule of
    num_warps = tile / 32 does not carry over. Now `muscl_hancock_pallas._call`
    sets num_warps = tile / 32 on NVIDIA only (8 for 1x4x64), within 1-2.4 % of
    the best of 204 configs in every precision. Full step after it, Mcell/s at
    256, 384, 512^3: FP64 856, 904, OOM in the bench process (Sedov 512^3 in a
    clean process: 835 at 10 GB); recon32 1141, 1220, 1230; FP32 2088, 2289,
    2416: 1.2x (FP64), 1.3x (recon32), 1.6-1.85x (FP32) the XLA step.
  - ncu (exclusive node, `dcgmi profile --pause`; on a shared node DCGM holds
    the counters), kernel alone 256^3 after the fix. FP32: FMA pipe 72.7 %,
    DRAM 7.9 %, 180 registers, no spills, occupancy 12.5 %, IPC 2.64, ~1820
    FP32 add/mul per cell and 0 fused FMAs (ncu: up to +36-50 % from fusing).
    FP64: FP64 pipe 77.6 %, DRAM 5.4 %, 255 registers and still 10.5 M spill
    requests. Both are arithmetic-bound now: what is left is FMA contraction
    (Triton-Pallas does not fuse; changes rounding) and the redundant
    predictors and faces (7 predictors, 6 faces per cell).
  - Five ways past that, prototypes in the tune directory (kernel alone, 256^3 unless said):
    (1) split, predictor once per cell in pass A, faces in pass B: with an XLA pass A it
    loses (XLA's predictor alone 6-18 ms); with both in Pallas, A writing W_half only and B
    recomputing slopes ("split-lite"), FP64 1.37x, recon32 1.25x, FP32 1.23x (512^3: 1.46,
    1.21, 1.19x), bitwise under Triton, +5 fields of (N+2)^3, compile 33 -> 11 s. Writing the
    slopes too (20 fields) OOMs FP64 at 512^3; x-slab chunking of that loses the gain to copies.
    Now production on NVIDIA. (2) x-marching with carried planes: bitwise, but a 209 s compile
    at 64^3. (3) reciprocals in HLLC: +12-20 % (rounding changes at 1e-15 / 6e-7). (4) FMA via
    inline PTX (Triton-Pallas does not fuse): +2-6 % on top of (3). (5) a CUDA shared-memory
    marching kernel through jax.ffi (each predictor and face once, 80/124 registers, no
    spills): bitwise with Pallas in FP64 at -fmad=false, but 0.95-1.00x the tuned Pallas
    kernel: synchronisation and shared-memory traffic eat the halved arithmetic.
  - (3) in production, 2026-10-11 (`fluxes._hllc_core`, `_hll_core`): the
    reciprocals of rho_L, rho_R, gamma-1 and _nz(denom) are reused, 7 divisions
    per HLLC face instead of 11, 3 instead of 5 for HLL; no sums reordered, no
    FMA, every _nz kept. Full step on 0214bdf, before -> after in the same job,
    two jobs, Mcell/s at 256, 384^3: Pallas FP64 1070 -> 1254/1181, 1158/1147
    -> 1291/1269 (+10-17 %); Pallas recon32 1406/1294 -> 1535/1400, 1447/1406
    -> 1578/1527 (+8-9 %); XLA FP64 and recon32 +0-4 %, HLL XLA +0-1 %, all
    within noise. FP32 unchanged and bitwise equal after 10 steps, XLA and
    Pallas: both already divide in FP32 as a reciprocal times the numerator
    (the prototype's FP32 gain was its FMA and reordering). Differences:
    FP64 fluxes on random states normwise 2.1e-14 (HLLC), 2.2e-16 (HLL);
    10 Sedov steps 128^3 at most 5.8e-16 (FP64), 7.6e-16 (recon32) of
    max|U|; mass and energy conserved to 2.3e-16. One bound moved:
    `tests/test_hydro_cfl_stability.py` 3-D grid-scale wave at rest, the
    round-off floor of std(rho), 1.34e-7 -> 6.39e-7 against 1.8e-7; it is the
    E_star reciprocals alone (c only 1.46e-7, E only 1.34e-7).
  - Current kernels, 2026-10-11 (main 37ef4bd). Step,
    Mcell/s Pallas / XLA at 128, 256, 384^3: FP64 HLLC 1114/570, 1188/732, 1270/746; FP64 HLL
    1272/481, 1400/695, 1488/778; mixed HLLC 1396/796, 1456/905, 1545/955; mixed HLL 1595/929,
    1769/1090, 1816/1154; FP32 HLLC 2154/1095, 2677/1359, 2711/1451; FP32 HLL 2566/819,
    2947/1353, 2957/1453. 512^3: FP32 2803/1307, mixed 1572/OOM (FP64 OOMs in the bench process
    after the XLA attempt; Sedov 512^3 alone 1116). ncu, divergence alone at 256^3 (split path):
    the faces kernel is FP64-pipe bound, 85 % (HLLC) / 89 % (HLL), DRAM 20 / 27 %, 128 registers,
    occupancy 25 % (register-limited), no spills, 132 B/cell; FP32 faces FMA pipe 82 %, DRAM 26 %,
    0 fused FMAs (ncu: up to +50 %; FP64 up to +39 %, its fused DFMAs are the division and sqrt
    sequences). Predictor kernel 2.0 ms FP64, DRAM 59 %, 112 B/cell. Pallas moves ~245 B/cell per
    divergence (FP64); XLA ~1470 B/cell per divergence, DRAM 78 % (FP64) /
    87 % (FP32): XLA is memory-bound, Pallas arithmetic-bound at ~1/6 of the bandwidth limit.
  - Before that fix, production step (`tools/bench_hydro_roofline.py`), Mcell/s
    XLA / Pallas at 128, 256, 384, 512^3: FP64 471/336, 707/344, 719/351,
    OOM/333; recon32 579/627, 842/889, 893/931, OOM/942; FP32 783/1222,
    1279/1916, 1362/2165, 1242/2245. FP64 Pallas was 0.5x XLA on the A100
    (2x faster on MI300A, `docs/AUTODIFF.md`); FP32 Pallas 1.5-1.8x XLA.
    XLA FP64 is bandwidth-bound (384^3: 107.5 GiB in 78.8 ms, 1.46 TB/s).
    XLA needs 29 GiB (FP64) / 17.8 GiB (recon32) in one allocation at 512^3:
    there only Pallas fits.
  - XLA barriers (divergence alone, all variants bitwise equal): dropping the
    after-y and W_half barriers is +5.5% in FP32 (512^3 1372 -> 1448),
    recon32 (384^3 1058 -> 1121) and FP64 (384^3 820 -> 866) at the same peak
    memory (FP32 512^3 30.2 GiB, 11.8x the padded state). No barriers at all:
    recon32 384^3 +36% at +18% memory; FP64 -22%; FP32 512^3 OOM.
  - Weak scaling, FP32 XLA Sedov: 512^3 on 1 GPU 1211 Mcell/s, 1024^3 on 8
    GPUs (2 nodes, mesh 2,2,2) 9129 Mcell/s: 94.2 %.
- **Pallas in every dimension and limiter, 2026-10-12/14** (Raven, one A100-SXM4-40GB, jax 0.10.1). The A10 checker found
  `hydro_kernel='pallas'` ran XLA silently for 2-D and for MC/van Leer. The kernels are now
  dimension-generic and call XLA's per-cell code (`slopes.slope_limiter`, the jacobian rows, the
  flux cores). Pallas vs the XLA wide step, divergence alone, all 84 (dim, limiter, solver,
  precision, fused/split) cases at 4096 / 128x192 / 32x48x64 cells: 0 differing values (bitwise);
  CPU interpreter <= 2.0e-16 (FP64), 5.6e-8 (FP32) of max|dU|. The 3-D minmod path is bitwise
  main's (40 sha256: divergence fused/split, HLLC/HLL, FP64/FP32/recon32, scalars, the production
  step at BOUNDARY 0/1/2/4).
  - Tile sweep, divergence alone, num_warps = cells / 32, every tile bitwise the same. 1-D 2^24
    cells: fused tile 32..256 7.56-7.58 Gcell/s FP64, 64 best in FP32 (19.5; 1024: 10.8); split
    7.0 (FP64) / 11-15 (FP32): 1-D stays fused, tile 64. 2-D 4096^2 (21 fused, 64 split tiles):
    fused best 4x32 2742 Mcell/s FP64, flat 6.5-7.2 G FP32; split predictor 2x64, faces 2x64 FP64
    3259 (best 3279), 4x32 FP32 7485 (best): split 1.2x (FP64), 1.04x (FP32) the fused kernel.
  - Production step (`tools/bench_hydro_roofline.py`, DIM/LIMITER, BOUNDARY 1, median of 20),
    Mcell/s Pallas / XLA, minmod: 1-D FP64 at 2^16..2^24 117/194, 431/379, 963/1005, 2093/2167,
    3237/3424; FP32 105/264, 680/465, 1289/1516, 3394/3167, 5927/4392. 2-D FP64 at 1024..8192^2
    1545/1234, 1417/1542, 2082/1985, 2067/1964; FP32 3032/1943, 4020/2402, 4466/3050, 4593/2856.
    MC and van Leer within 10 % of minmod except FP64 van Leer 2-D 0.8x XLA (1526/1955 at 4096^2):
    its FP64 division, done for three slopes per axis in the faces pass. 3-D 256^3 FP64 1194/735
    (minmod), 1064/723 (MC), 886/709 (VL); FP32 2334/1341, 2269/1363, 2547/1341; recon32
    1414/906, 1411/901, 1417/902. The step costs ~2.9 ms more than the divergence at 2-D 4096^2
    FP64 (cons_to_prim, halo, update outside the kernel), which XLA fuses into its own.
- **OU driver normalisation** (`turbulence.py`), fixed: it used tr(P_k)
  instead of tr(P_k^2). At zeta=0.5 that gave 0.5 rms^2, so amplitudes were
  1/sqrt(2) too low.
- **OU kick, two .at adds** (`OUDriver.apply`), 2026-10-16/19, MI300A, 512^3, FP64 state, FP32 force,
  jax 0.10.0 / rocm 7.2, median of 7-10. Four chained `.at[i].add` cost 24.98 ms per kick (force field alone
  2.51 ms, applying a given F 5.06 ms). `U.at[1:4].add(stack(dm)).at[4].add(dE)`: 16.70 ms, bitwise equal
  on CPU and GPU (job 12282865). A stack + concatenate is 8.67 ms and bitwise on the GPU, but not on the
  CPU backend (FMA contraction differs), and it promotes a weakly typed FP64 state to FP32 (a weak FP64
  times the FP32 force is FP32): commit 4dbfa73 did that, every XLA-kick run from it was FP32 after the
  first kick. `.at` keeps U's dtype; with a weak U it does the add in FP32, so main's first kick rounded
  the momenta to FP32 once (max 2.7e-8, job 12282325). Since then `setup/evolve.initial_state` makes the
  run's state strongly typed (driven_turbulence_3d and others build U with jnp.full of a Python float).
- **OU kick in Pallas** (`hydro/turbulence_pallas.py`, picked by hydro_kernel), 2026-10-16, same setup.
  One pass over U per tile; F from per-pair 1-D tables (cos/sin of ky y + kz z by angle addition), never
  stored. 512^3: 6.90 ms against 8.89 ms for the XLA stack kick and 3.12 ms for a copy of U; max |dF| / rms F
  1.6e-6 against the XLA kick (summation order), rho bitwise, 2 shards = 1 device bitwise (job 12246898).
  Driven step 512^3: kicks 13.8 ms against 40.5 ms of hydro, step 62.3 ms (was 98.2; job 12246866).
  768^3 with a preallocated pool: peak 70.9 GiB (81.0 with the XLA .at kick), step 202 ms (320); the
  kick's XLA temporaries fall from 30.4 GiB to 0.08 (jobs 12247616-17). Without a preallocated pool 768^3
  fails on fragmentation, not size: the BFC regions cannot hold one more 16.9 GiB U.

- **`preserve_face_states`, RAMSES positivity floors, 2026-10-19** (viper MI300A, jax 0.10.0, branch guard-ad).
  One opt-in switch for mini-ramses's four floors: cons->prim rho >= smallr, e_int >= smalle rho
  (`umuscl.f90:1813, 1847`; `conversions.cons_to_prim_floored`); the face trace guard, a face rho < smallr
  or P < smallr smallc^2 takes the cell value (`umuscl.f90:422-441`), then the Riemann inputs
  rho >= smallr, P >= rho smallc^2 / gamma (`godunov_utils.f90:370-383`; both `slopes.ramses_face`); the
  CFL dt from the floored primitives. U is not floored. `hydro_smallr`, `hydro_smallc` in code units
  (default 1e-10). Off: no new op, bitwise main (16 XLA/Pallas variants, job 12286583); on, a smooth
  state is bitwise unchanged too. Separately, always on: the HLLC/HLL sqrt guard (double where on
  c^2; HLLC branch tests on `ok = c^2 >= 0`, F_R otherwise as before; HLL takes c = 0 there, it was NaN, and keeps
  its branch tests, as `ok` there broke the bitwise 1-D = 2-D strip of `test_hydro_hard_odd_even[hll]`) keeps the VJP finite.
  Cost, `tools/bench_hydro_roofline.py` MC mixed periodic 512^3 one GPU, two reps (job 12287046): Pallas
  38.0 ms main, 38.0 off, 48.4 on (+27 %); XLA 66.7 main, 67.4 off (+1 %, the sqrt guard), 65.6 on.
  Weak, `tools/step_breakdown.py` on T3 Mach 16 Pallas mixed, 512^3 per GPU (jobs 12287180, 12287323):
  step 99.3 / 99.1 ms main / off on 1 GPU, 110.2 / 110.5 on 8 (1024^3), weak efficiency 90 % both; on: hydro
  49.8 vs 40.1 ms on 1 GPU, step 120.2 ms on 8 (hydro 59.5 vs 49.0).
  Gradients (128^3 MC Mach 4, `dJ/dlnA`, `dJ/dzeta` vs central FD, jobs 12286586/12286667): FD agreement
  unchanged, ~1e-7 up to 1 t_turb, 1e-4..1e-3 at 2 t_turb with or without; grad/fwd 5.3-6.4 both.
  T3 (256^3 Pallas MC HLLC recon32 isothermal, acc 3 M^2, 5 turnovers): Mach 16 zeta=1 (12286737) and
  Mach 12 zeta=0.5 (12286754) survive with no conserved rho <= 0; Mach 16 zeta=0 off is non-finite at
  0.58 t_turb (12286790), on stays finite but conserved rho <= 0 appears at 1.9 t_turb (up to 5947 cells)
  and dt collapses to 1e-6 median (12286666); with `hydro_fofc` added (12287062) no rho <= 0 and median
  dt 6.6e-5, at 114 vs 22 ms per monitored step. Test `tests/test_preserve_face_states.py`.
- **`hydro_fofc`** (AthenaK FOFC with HLL, cascade second-order HLL -> first-order HLL) is kept off main
  on branch `fofc` until it has a test and a 512^3 memory check.

## time_integration

- **NaN dt** (2026-09-18). A negative pressure made the CFL dt NaN; the
  cross-device min (RCCL) drops NaN operands, so a 128^3 slab A/B ran 87 steps
  on two NaN ranks unnoticed. `compute_dt` now maps NaN to -inf so the
  finite-dt guards fire.

## gravity

- **Flux mode without a floor; Pallas mass fluxes; dark matter** (2026-10-15, Raven A100-40GB, jax 0.10.x,
  FP64; dossier updated). Results:
  - Each face's work is split half/half, except next to a cell the split would leave with < 0.5 of its e_th,
    where it goes to the donor cell. A conservative limiter backs this up.
  - Evrard runs with no floor at 32-256. |dE/E0| at t = 3 is 1.1e-6, 1.1e-5, 1.7e-5, 1.2e-5. The limiter moved
    2.5e-4 |E0| at 32 and nothing from 64.
  - Profiles against AthenaK PLM 256^3 are those of the half split (L1 ln rho at t = 3: 0.104, 0.049, 0.0067
    at 64-256). Donor-cell everywhere was 0.234 at 64: first-order heat.
  - Tried: donor where the half split cools by more than 0.1 or 0.02 of e_th. 0.02 is worse at 32 (the
    limiter moved 1.4e-2), and 0.1 is no better than 0.5.
  - Pallas: the hydro kernels return F_rho bitwise equal to XLA. A `pl.when` storing the last face cost +45 %.
    Storing both faces from every cell costs +4 % in the kernel and +3.4 % in the step at 256^3.
  - The whole step is 26.3 ms for flux and 21.3 ms for the Pallas kick (it was 31.4).
  - With dark matter the gas stays exact. The particles' own budget, dt-independent (1.35e-4 at 64), is the PM
    force not being the gradient of the CIC potential.
  - Isolated (64^3): the ~1e-4 is energy through the box faces. Booked, it is 1.4e-5 at t = 1, 8.6e-5 at
    t = 2 and 2.7e-4 at t = 3, after the outflow (unbooked 0.07, 0.20). A tighter solve does not help.
  - hllc-mc 64^3 still dies at t = 2.24, in the hydro step (box corner).
- **Energy through the face mass fluxes** (`gravity_energy='flux'`, 2026-10-14, Raven A100-40GB, jax 0.10.x,
  FP64). Mullen, Hanawa & Gammie 2020
  eq. 57-68 in LUNA's kick-drift-kick: the hydro step returns its Riemann mass fluxes, the closing half-kick
  adds -dt/(2dx) sum F dPhi with Phi the step average, minus the opening half-kick's work. Evrard
  |dE/E0| at t = 3: 1.2e-5 at 64^3 and 256^3 against 1.55 and 0.58 with the kick; at a 1e-10 solve 2e-11
  at t = 1 (round-off). The default kick is bitwise unchanged (A7, A8, turbulence hashes against main, GPU
  with deterministic ops and CPU). Not default: cold gas goes to negative pressure where the face work, split
  half to each cell, exceeds a cell's internal energy (Evrard ambient at 32^3, the infall at the bounce at
  64^3), so it needs a pressure floor; the floor's energy is the run's error. Tried first without: NaN at
  step 1 at 32^3.
- **In-place halos** (`poisson_bc.pad_inplace`, 2026-10-11, Raven A100-40GB,
  jax 0.10.1; gravity audit). The
  multigrid halos were `jnp.pad(mode="wrap"/"edge")` (dense) or three chained
  `ppermute` + `concatenate` (`_pad_axis`, sharded). XLA materialised a padded copy
  per stencil, so one RBGS iteration moved 2x the ideal bytes (1.545 GiB at 256^3
  FP64). Writing the ghosts into one padded buffer, x -> y -> z (the same values,
  corners included), is bitwise equal. Production warm solve (FP32 V(1,1) on FP64,
  to 1e-4, 3 cycles):
  - dense: 2.82 -> 2.30 ms at 128^3, 11.31 -> 6.85 at 256^3, 76.7 -> 42.1 at 512^3;
  - sharded, 256^3 per GPU: 17.9 -> 6.9 ms on a 1x1x1 mesh, 15.5 -> 9.7 on 2 GPUs,
    16.8 -> 11.7 on 4.

  The hydro halo's `optimization_barrier` costs +44% at 256^3 and +114% at 128^3
  here, so there is none. A size-1 mesh axis wraps locally instead of
  self-permuting.

  One exception: the root grid that a mesh gathers (`gathered_coarse`, solved
  redundantly on every device inside the shard_map program) keeps one `jnp.pad`
  (`make_halo_3d(..., inplace=False)`). In place it was slower there: 2 GPUs,
  256^3 per GPU, 10.62 against 9.33 ms; 512^3, 29.06 against 26.54 ms.
  Not understood.
- **Plain Pallas gas kick** (`kick_pallas.kick_plain`, 2026-10-11, A100). XLA
  lowers `kick_gas_body` to one concatenate fusion. Each of the 5 output slabs
  recomputes its inputs, which is 0.37 TB/s at 256^3 FP64 (4.68 ms). The Pallas
  kernel streams the 13 fields once and is bitwise equal:
  - FP64: 1.42 ms (1.23 TB/s, 3.3x) at 256^3, 10.9 ms (1.8x) at 512^3;
  - FP32: 0.87 ms (2.5x) at 256^3, 6.2 ms (1.5x) at 512^3.

  It is memory-bound. A sweep of 7 tiles x num_warps {default, 1, 2, 4, 8}
  stayed within about 5% of Triton's default launch on the 1x4x64 tile (FP32
  13% at most), so the default stays. A stacked XLA rewrite (momenta as one
  (3, ...) expression) was bitwise equal and 1.6x at 256^3, but 15-26% slower
  at 512^3, so the XLA kick is unchanged.
- **Sink-kick launch** (`kick_pallas._call`, 2026-10-11, A100, 256^3 FP64).
  - Triton does not turn the nearest-image `ddx / L` into a multiply. `* (1/L)`
    is bitwise equal, because only `floor` reads the quotient: 1.47x at box 3.
    Box 1 and box 4 are unchanged, since a power of 2 was already cheap.
  - The per-cell loop over slots is light. Several cells per thread win:
    num_warps=1 is 1.22-1.28x with 32-256 live sinks (the hydro rule tile/32 is
    0.84x), and neutral with few sinks (memory-bound).
  - Both together: 6.18 -> 3.24 ms (32 live), 47.3 -> 23.8 ms (256 live).
- **Fused sink-kick Pallas kernel.** The XLA path moved 27 GiB per call at
  512^3 and spent about 24 ms per call on the sink field alone (2026-09-25).
- **Fixed-point deposit.** A float CIC scatter changes 7% of cells under a
  mere particle reordering (512^3). The technique is from QuanTaichi
  (SIGGRAPH 2021) and OpenMM 2012; the exact-mass rounding-tree split and its
  use in PM are LUNA's. The integer scatter went from 984 to about 20 ms at
  1024^3 on 8 GH200 (slab path).
- **Deposit padding.** Zero-weight padding slots aimed at one cell serialize
  the atomics (2.6 s vs about 0.1 s at 512^3). Spreading them x-fastest costs
  5.7x a real particle (0.62 vs 0.11 ns/slot); z-fastest matches
  lattice-ordered particles.
- **`block_rows`.** Filtering the chunked read on x as well dropped the
  particles that crossed an x-slab face (2026-09-17).
- **Resort.** A cell-order resort recovers a 4-8x deposit/gather locality
  loss (WarpX-style sparse cadence). The jax 0.10.0 / ROCm 7.2 native argsort
  is wrong for n >= 2^16; before the radix fallback the resort lost 25% of
  the particles.
- **Sharded multigrid.** The communication-avoiding smoother measured
  -24% / -29% at 4 / 8 GPUs across nodes and was slightly slower within a
  node (XGMI). The mixed-precision cycle reaches the same converged residual
  at about 2/3 the cycle cost on one GPU and halves the halo and
  agglomeration bytes.
- **`_local_cell`.** The minimum-image fold fixes audit item 14: a particle
  that crossed the domain wrap since the last exchange was clipped into the
  wrong end of its block. A symmetric `(r + Ng/2) % Ng` fold is wrong when
  nloc > Ng/2.
- **Neumann adjoint gauge.** Pinning the gauge for the Neumann boundary too
  (as for periodic) moved the Neumann adjoint gate the wrong way, 2.82e-3 ->
  4.06e-2 (jax 0.10.0 and 0.10.2 alike): the error sits in the discrete
  operator's adjoint at the boundary, not in the null-space gauge.

## species

- **Sink field on the gas kick.** Skipping empty slot chunks: the kick took
  36.9 ms against 12.6 ms without the field (512^3). Unrolled over all slots,
  the field kept one grid of scratch per slot: 33 GiB at 512^3 with 32 slots.
  Per-slot scatter: 0.43 ms each.
- **Sink creation.** The old `_maybe_create` built about 15 full-grid arrays,
  ~12 ms of a 38 ms step at 256^3.
- **Sink moments.** The old moment kernel built whole-grid meshes: 36 GiB at
  1024^3, where the res2 1024^3 runs died.
- **Overlapping windows.** Splitting a cell reached by two sinks between them
  fixed audit C4.

## sources

- **Off-node winds made momentum (2026-10-14).** `feedback.wind_updates` and
  `inject_wind` pushed each deposit cell at v_star + v_term e with e the unit
  vector to its centre; the cells within the radius are symmetric about the
  star only on a grid node, so elsewhere the gas gained dm v_term <e> extra.
  Found by the conservation audit, strict xfail
  `test_an_off_node_wind_adds_only_the_stars_momentum`. Fix: e -> (e - <e>)
  rescaled to keep sum |e|^2 (`_zero_net`, `_zero_net_grid`; docs/TORCH.md).
  Measured, 24^3, two stars of 0.02 and 0.05 at 30 and 12, jobs 31171755 (one
  A100) and 31171758 (CPU, 8 virtual devices): |sum dp - sum dm v_star| 7.37e-3
  -> 5.9e-17 GPU / 1.7e-17 CPU (`inject_wind` the same); with add_heat
  dE - 0.5 dm (v*^2 + vt^2) +9.8e-4 -> -3.6e-15 (of 12.6). A star on a cell
  centre was worse, 2.1e-2 (3.5 % of dm v_term) -> 5.2e-17. On a node the
  residual was already 2.6e-16 (now 7.3e-17) and the deposit moves by 3.6e-16
  in momentum and 3e-15 in energy relative to max |dU|, not bitwise. Off a
  node the per-cell momentum moves up to 0.9 % of max |dU|, per-cell speed
  0.972-1.028 v_term, direction within cos 0.9986 of e_r (50 random stars,
  worst momentum residual 1.6e-16). The FLASH-Torch modes keep FLASH's
  weights and still carry net momentum off a node (job 31172054): 5.5e-4 to
  1.1e-3 of dm v_term in momentum mode, 3.1e-3 to 9.6e-3 in energy mode.
- **Radiation stage scratch.** A substep loop inside one program needed 33
  fields of N^3 of scratch, against 24 for one substep alone (256^3 and
  512^3); at 512^3 per GPU that 33 GiB request killed an 8-GPU 1024^3 run.
  Without the wait between substeps, 24 GiB of substep scratch plus a similar
  hydro scratch on an 82.5 GiB device killed 8-GPU 1024^3 jobs 11954521 and
  11954663 in their first steps.
- **Dark skip.** The cloud runs dark until its first star: 13 substeps of
  0.35 s at 512^3 per GPU, 96% of the step (2026-09-25).
- **Ionized CFL.** Before the bound, a split M1 step put the hydro 10x over
  its CFL around a new source and density went negative in step 1. The
  unsplit scheme was covered by accident: its 16-substep cap is about
  0.37 dx/c_i. The cloud's steps are 3-5x below dt_max and used to pay all 16
  substeps.
- **CC85 wind dt bound.** Without it, the M82 setup at 32 cells had negative
  pressure after step 1. A per-step device reduction for rho_min, and three
  device variants of `injection_rho_min`, stalled the ROCm runtime on
  2026-09-20; both are now done once, on the host.
- **`external_gravity` G.** `G_CODE` used to be the literal 43007.1 here and
  in `cosmology.music_ic`, implying Msun = 1.98832e33 g against 1.98892e33 in
  `units.py` (3.0e-4 apart). It is now derived from `units.py`.

## radiation

- **A100 launch, 2026-10-11** (Raven, FP64; M1 audit).
  The M1 sweep's 1x8x64 tile under Triton's 4 warps spilled (ncu: 255 registers, 10.2 M spill
  requests); on NVIDIA `m1_pallas` now uses 1x4x32 with 4 warps, one cell per thread, bitwise:
  coupled STARBENCH per hydro step 64.2 -> 45.8 ms (128^3), 460.2 -> 331.5 ms (256^3), XLA
  139.9 / 1163.7 ms; radiation is 88-92 % of that step. Split step alone at 256^3: XLA 99.8,
  Pallas 44.6 -> 32.0 ms; XLA does not fit 512^3 in 40 GB. Ionize 11.6 -> ~10.3 ms (XLA 19.7);
  it agrees with XLA to round-off (R within 1.1e-16), not bitwise. `m1_floor` as one concatenate:
  2.87 -> 1.74 ms at 256^3, bitwise. Open from the audit: predictor once per cell (~1.3x, not on top
  of the closure below; a bitwise prototype in the audit directory), the pad before each sweep
  (~26 % of the tuned step is XLA glue). The m1.py barriers: below.
- **Division-free Levermore closure, A100, 2026-10-11** (FP64, not bitwise: approved round-off change).
  `m1._closure`: chi = (5 - 2s)/3 with s = sqrt(4 - 3f^2) instead of (3 + 4f^2)/(5 + 2s), and
  q = 1/(c_hat E_m |F|_m) giving f = |F| |F|_m q and 1/|F|_m = c_hat E_m q (E_m, |F|_m the floored
  values, the E_FLOOR^2 softening under the sqrt kept): 3 -> 1 division per face state.
  Pallas sweeps x+y+z 12.88 -> 10.47 ms at 256^3 and 101.1 -> 82.3 ms at 512^3 (1.23x); Pallas split
  step 31.7 -> 27.3 ms and 247.8 -> 210.6 ms; XLA split step unchanged (memory-bound, 94.7 -> 95.1);
  coupled STARBENCH 256^3 per hydro step 330.2 -> 293.9 ms. Pallas stays bitwise with XLA. Against
  the old closure after one split step (256^3): max relative 3.1e-16 (E), 1.1e-15 (F), 73 % of
  cells moved; rk2 2.1e-16. Gate numbers unchanged to the printed digits (STARBENCH Stromgren
  +5.62 %, split diffusion +10.10 / +2.04 %, rk2 diffusion +3.02 %, beam crossing 0.4398).
- **Implicit ionization, one division per Newton step, A100, 2026-10-11** (FP64, approved round-off
  change). r = 1/(1+u) shared by g and g' in `ionize_implicit` and `m1_pallas._ionize_call`: XLA
  19.7 -> 16.3 ms and 154.0 -> 128.4 ms, Pallas 10.3 -> 8.4 ms and 76.4 -> 64.5 ms (256^3, 512^3).
  Against the old step: R within 8.1e-17 relative, x within 2.2e-16, a handful of cells; Pallas and
  XLA agree as before (x bitwise, R 7.9e-17). Gate numbers unchanged to the printed digits.
- **Radiation speed, second audit, A100, 2026-10-11** (Raven A100-SXM4-40GB, FP64, jax 0.10.1).
  "Bitwise": hashes of the M1/M2 split steps, the STARBENCH stage on one GPU and on a (2,1,1) mesh and
  the non-ionizing stage, against main b12c319 on the GPU and on the CPU (XLA and the interpreter).
  CPU hashes depend on the job's core count (XLA CPU, 1e-16 in a few cells): compare within one
  allocation shape. Coupled STARBENCH, Pallas, ms per hydro step, one job each size; 128^3 is
  host-bound and moves with the process (main 40.2-50.8 ms across jobs), so medians of three runs.
  - Substep program: the ionizing substep advances (R, x) only, the two-temperature EOS once per half
    step, w an argument; a substep is queued while the previous one runs up to `LAG_MAX_CELLS` (2^25
    per device; above it the wait of "Radiation stage scratch"). Bitwise. Coupled 256^3 295.0 -> 272.4
    ms; half step 256^3 163.1 -> 148.8 ms, peak 7.63 -> 5.81 GiB (no pass-through copies of U and w);
    the lag alone 1.04-1.05x (256^3) and 1.21-1.30x (128^3), half steps in one process.
  - M1 no pad: x and y sweeps read the unpadded state (scalar edge clamp or wrap of the plane index,
    tile 4x1x32 for y), the sweep before a z sweep writes z ghosts and `_ghost_call` fills them in
    place. Bitwise. Split step 256^3 28.07 -> 23.83 ms, temp 1.01 -> 1.02 GiB; coupled 256^3 248.4 ms,
    128^3 47.3 -> 38.7 ms (with the substep program).
  - Predictor once: `_predict_call` then `_faces_call` up to `PREDICT_ONCE_MAX_CELLS` (2^25 per
    device). Bitwise. Split step 23.83 -> 22.66 ms (same process 1.05-1.07x), temp 1.02 -> 2.14 GiB;
    coupled 256^3 238.3 ms, 128^3 38.1 ms (within the noise there). Off at 512^3, where a random-state
    split step with all M1 changes takes 162.6 ms against main's 210.6, temp 8.06 against 8.03 GiB, peak
    20.0 GiB both. The STARBENCH stage and the coupled run at 512^3 do not fit one 40 GB A100, before or
    after.
  - No final `m1_floor` after the split step (XLA and Pallas). Round-off: STARBENCH 64^3 after 12 half
    steps R within 2.1e-16 of max|R| per component (XLA; Pallas 1.3e-22), x 2.2e-16; 128^3 to t_end
    front 1.4028 pc both, L1(rho) 6.9e-16, L1(P) 1.9e-15. Split step 22.66 -> 20.95 ms; coupled 256^3
    224.6 ms (1.31x main, peak 6.69 -> 6.64 GiB), 128^3 36.1 ms (1.31x).
  - M2 no pad, one cell per thread on NVIDIA (1x1x128, 4 warps). Bitwise. Trace 37.44 -> 24.28 ms,
    learned 56.37 -> 42.71 ms (256^3); temp 1.27 -> 2.54 and 1.72 -> 2.99 GiB: two z-padded states are
    live across the z z pair (the audit's "+ghosts only" was wrong), ~+10 GiB at 512^3.
  - XLA M2 sweeps without the predictor-flux barrier. Bitwise. 128^3: trace 37.52 -> 35.24 ms,
    learned 43.43 -> 41.70 ms, temp 1.12 -> 1.12 and 1.35 -> 1.37 GiB.
  - `m2_split.floor_cells` with one reciprocal of m0 for seven divisions. Round-off: one split step
    within 1.8e-15 of max|U| per component (64^3, both paths), crossing 0.0919 unchanged; STARBENCH
    with `m2_learned` at 128^3 to t_end front 1.3334 pc both, L1(rho) 2.2e-8, L1(P) 9.5e-8 (the FP32
    network amplifies the last bit). Fused 24.2 -> 22.9 / 42.6 -> 41.5 ms (256^3), XLA 35.2 -> 28.9 /
    41.7 -> 35.2 ms (128^3).
- **M1 sweep.** The XLA sweep at 512^3 moved 148 GiB per sweep (18x the
  field); the fused Pallas sweep was 4-5x faster on an MI300A (2026-09-25).
- **M1 `_transport_from_padded` barriers** (256^3): inputs through the
  barrier took temporaries 37 -> 20 fields of N^3; materialised fluxes
  43.3 -> 32.8 ms per substep (the face-flux alternative was 26.4 -> 39.3 ms);
  the closure-scalar barrier a further 3.5%. A100, 2026-10-11 (FP64, bitwise): without that
  closure barrier and the Hancock sweep's predictor-flux barrier the XLA split step went
  94.9 -> 79.1 ms at 256^3 and 311.2 -> 264.0 ms at 384^3, temp 3.52 -> 3.39 and
  11.86 -> 11.43 GiB; the rk2 step, which shares the closure, 55.0 -> 55.1 and 187.1 -> 186.4 ms.
  The face-flux barrier stays (+37 % without it). MI300A not re-measured.
- **Implicit ionization.** The fused kernel with early tile exit: 37 -> 22 ms
  at 512^3. The unrolled XLA solve instead of `fori_loop` saved 9-18 ms per
  substep at 256^3. Keeping exact roots fixed cut the worst photon error
  after 60 iterations from 1.7e-6 to 8e-8. Sequential split ionize gave peak
  x 0.077 / 0.20 / 0.39 for 1 / 2 / 3 alternations.
- **Learned M2.** A random output layer plateaued at rms closure error 0.47,
  hence the zero ("off") init. The unbounded 3-D head went non-finite around
  step 100 at 64^3. `third_moment`: 16 blocks fastest at 256^3 (606 ms/step
  against 659 at 64 blocks and 631 at 8); block=0 is about 1.4x faster where
  it survives. The stacked tensor in `_rebuild` replaced 46 scatters, which
  segfaulted ROCm. XLA network hidden layers take 16 GiB each at 512^3, hence
  NET_BLOCK.

## nuclear

- **Burn thresholds, 2026-10-19** (Raven A100, jax 0.10.1, FP64). The 512-cell C/O white-dwarf
  pilot (`scripts/examples/stellar/white_dwarf_explosion.json` at 69ee9fb, job 31301165) stalled at
  t = 0.0197 s, step 337: dt fell from 1e-4 to 1e-17 s and 21 halvings were refused. Cell 88
  (rho = 10 g/cm³, T = 1.0e6 K) failed the aprox13 integration at every dt, since the rates refuse
  T < 1e6 K (`src/luna/nuclear/light_rates.py:18`); hydro and EOS recovery were valid everywhere.
  Mass and energy budgets 1.5e-15 and 1e-14. Hence `burn_min_temperature`/`burn_min_density`.
- **One jit per reactive step, 2026-10-19** (Raven, jax 0.10.1, CPU, FP64). `make_step` returned the
  Python glue between jitted pieces; the A100 sat at 3 % mean utilisation on the 512-cell pilot
  (1.3 s/step). Jitting the whole step is bitwise on `tests/test_stellar_golden.py` he_planar and
  smooth_periodic but not co_spherical: max relative 2.9e-7 after 12 steps (one retry), presumably
  XLA fusing the glue and Rodas3's step control amplifying the last bit. The golden file is
  generated from the jitted step.

## tools

- **`tools/rm_linear._bessel_psi`, 2026-10-11** (Raven, numpy 2.4.6, scipy 1.16.3). `scipy.special.itj0y0(x)[0]`
  (int_0^x J0) is right to 1e-14 up to x = 10, off by 5e-10 at x = 20 and by ~1e10 from x = 20.1 on
  (quadrature and the Struve identity agree to 1e-15). The Bessel history therefore blew up past
  home-fluid time 20: the light -> heavy A14 case (home fluid b, tau_b = 1.68 tau_a) gave psi/psi0 ~ 1e10 at
  tau_a >= 15, so its L1 was 1.9e10 and both light -> heavy gates failed; its teeth passed vacuously. Every
  route-check of `python tools/rm_linear.py` past tau_a = 20 failed the same way. Replaced by
  int_0^x J0 = 2 sum_k J_{2k+1}(x) over the K orders already computed; Bessel vs Bromwich over the window
  1.3e-15 / 3.6e-15 (was 5.5e-10 / 9.6e9), light -> heavy L1 0.872 / 0.619 (minmod), 0.627 / 0.417 (MC), as measured on viper.
  Jobs itj0y0.31078207, rm_fix.31078243, rm_teeth.31078482.

## 2026-10-30: the face report's cost (audit C8)

Each dense step now returns the count of unphysical Hancock face states (`integrators._face_report`), so a run
can refuse them or log them. Pallas writes an int8 face bitmask per cell, and the step reduces it once. The
location (`nonzero`) runs only on a step that has such a face. Measured on an A100-SXM4-40GB in FP64 with
minmod and HLLC, `tools/bench_hydro_roofline.py` with REPORT=0/1, median of 20 runs, two repeats, job
31663811: Pallas 384^3 46.60 -> 47.17 ms (+1.2 %), 256^3 14.76 -> 14.86 ms (+0.6 %); XLA within +-0.5 %.
512^3 FP64 runs out of memory on a 40 GB card with or without the report. Adopted 2026-10-30. An
in-kernel per-tile reduction would remove the per-cell store if this ever matters.
