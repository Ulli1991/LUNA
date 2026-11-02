# The hydro time step

LUNA's hydro step is unsplit MUSCL-Hancock: one Hancock predictor at the cell centre that uses all
directions at once, then one Riemann solve per face. Its stable time step is set by the *sum* of the
signal speeds over the directions, not by the largest one. LUNA therefore uses the RAMSES form of the
CFL condition:

    dt = CFL / max over cells of  sum_d (|v_d| + c) / dx_d          (CFL = 0.8 by default)

With cubic cells this is `CFL dx / (ndim c + sum_d |v_d|)`, the formula of RAMSES (`cmpdt` in
`hydro/godunov_utils.f90`, default Courant factor 0.8). It is the only rule: the measured stability
limit is `CFL` = 1.0 in every dimension, so the default 0.8 keeps a 20% margin. An old config that
set a per-axis `CFL` (0.3, 0.4) should simply use the default; `cfl_rule` is retired and fails with
that message.

Every hydro dt uses it (`time_integration/integrators.cell_dt`): the dense step's `compute_dt`
(and through it the AMUSE stepper, the cosmological runs and the autodiff rollout), the AMR driver
(`compute_dt`, `compute_dt_subcycle`), the ionised-gas dt bound of the radiation stage, and the
cold-start bound of the CC85 wind stage (the rule at rest, at the sound speed the first
half-injection gives the emptiest cell). `validate()` refuses `CFL` above 1.

## Why: the per-axis rule allowed an unstable step

The per-axis default was CFL 0.4. In 3D that is RAMSES 1.2 at rest, above the unsplit limit: 1e-8
noise in gas at rest grows to O(1e7) and the low-Mach Taylor-Green vortex blows up.

**Linear (von Neumann) analysis.** Linearised 3D Euler, exact upwind flux (which HLLC is for a
linear problem), first-order and unlimited centred slopes (minmod lies between; both give the same
limit). Largest stable dt in both units:

| background u / c | per-axis limit | RAMSES limit |
|---|---|---|
| (0, 0, 0) | 0.33 | 0.99 |
| (0.5, 0, 0) | 0.48 | 1.12 |
| (0.5, 0.5, 0.5) | 0.43 | 1.28 |
| (1, 1, 1) | 0.42 | 1.26 |
| (2, 0, 0) | 0.68 | 1.13 |
| (2, 2, 2) | 0.38 | 1.13 |
| (5, 0, 0) | 0.78 | 1.04 |
| (5, 5, 5) | 0.33 | 0.99 |

The per-axis limit moves between 0.33 and 0.78 with the flow, so a single safe value has to be 1/3
and wastes dt in axis-aligned flow. The RAMSES limit stays at about 1.

**Nonlinear scans (the production step).** Minmod, HLLC, FP64, the shard_map step on one MI300A
(Pallas kernel in 3D), periodic 128 (1D), 64^2 (2D), 32^3 (3D). The perturbation is white noise
(amplitude 1e-8, and 1e-3 in 3D) or a grid-scale diagonal sound wave (a quarter of the grid
frequency) on uniform gas moving at u; 600 steps. Unstable = std(rho) grew more than 10x; every
unstable run grew by 1e2 to 1e8 or went to NaN, every stable one decayed. RAMSES `CFL`:

| dim | background u / c | stable up to | first unstable |
|---|---|---|---|
| 1D | 0, 0.5, 1, 2, 5 | 1.00 | 1.05 |
| 2D | (0, 0) | 1.00 | 1.05 |
| 2D | (5, 0), (5, 5) | 1.05 | 1.10 |
| 2D | (0.5, 0), (1, 0), (2, 0), (1, 1), (2, 2) | 1.10 | 1.20 |
| 2D | (0.5, 0.5) | >= 1.20 | - |
| 3D | (0, 0, 0) | 1.00 | 1.02 |
| 3D | (5, 0, 0), (5, 5, 5) | 1.05 | 1.10 |
| 3D | (0.5, 0, 0), (1, 0, 0), (2, 0, 0), (2, 2, 2) | 1.10 | 1.20 |
| 3D | (0.5, 0.5, 0.5), (1, 1, 1) | >= 1.20 | - |

The measured limit is RAMSES `CFL` = 1.00 in every dimension, set by gas at rest (in 1D at exactly 1
the grid-scale wave neither grows nor decays). The default 0.8 keeps a 20% margin everywhere; at 0.8
and 0.9 the 3D noise and waves still decay after 3000 steps for every background above. In the old
units: per-axis 0.4 is stable in 1D and 2D (RAMSES 0.4 and 0.8 at rest) and unstable in 3D at rest
and at (5, 5, 5) (RAMSES 1.2); per-axis 0.33 was the 3D limit at rest.

`tests/test_hydro_cfl_stability.py` holds this: the default damps noise in 1D/2D/3D at rest and in
fast axis and diagonal flow, `CFL` 1.2 grows it in every dimension and in fast diagonal flow (the dt
of the old per-axis 0.4 in 3D), and `validate()` refuses CFL > 1.

## What it changes

In 3D at rest the default dt is 0.8/3 = 0.267 dx/c, against 0.4 dx/c before: 1.5x the steps (the old
default was unstable). For fast axis-aligned flow the RAMSES dt is the longer one: at u = 5 c it is
0.8/8 = 0.1 dx/c against 0.4/6 = 0.067 dx/c. In 1D the default doubles dt (0.8 against 0.4). In 2D
it is unchanged at rest (0.8/2 = 0.4) and up to 2x longer in fast flow.

A Sedov blast is sound-speed dominated, so its step count scales with 1/CFL at rest: the 3D Sedov
accuracy test at 256^3 took 2673 steps at per-axis 0.4, 4032 at RAMSES 0.8 (1.51x, the expected
1.2/0.8) and 8100 at RAMSES 0.4 (3.03x, a run whose script was not yet rescaled).

The hydro accuracy and hard tests (branch `hydro-hard` with this change, default dt, against the
same tests at per-axis 0.4; `MEASURED` values, CPU-8 and MI300A) all pass. The changes:
- 1D (Toro 1-5, sound wave): dt doubles; L1 falls by 10-30% (Toro) and 3x (sound wave).
  MUSCL-Hancock's error falls as the 1D Courant number nears 1.
- Boosted Sod (1D) and KH (2D): L1 10-37% lower; the boost resolution penalty 25-50% lower (Sod at
  100 c_s: 9.0 against 13.6; KH at 10 c_s: 1.73 against 2.29).
- 3D Sod along a diagonal: L1 +5-9%. Sedov 32^3: shock radius error 0.83 against 0.80 cells;
  64^3-256^3 (two MI300A) within 1%.
- Taylor-Green (it ran at per-axis 0.3): the same to 4 digits. Isentropic vortex: L1 within 7%,
  phase error mostly 5-25% lower (32^3 after 100 periods: 0.018 against 0.014 cells).
- Carbuncle and Noh: within 4%, except the HLLC wall at Mach 20 (row spread +8% on CPU, -11% on the
  GPU; HLLC carbuncles either way).
- Polytrope (BOUNDARY 4, gravity): 32^3 max |drho_c| 0.203 against 0.175 (bound 0.228); 64^3 and
  128^3 within 1-8%.
