# Test suite against the literature, 2026-10-07

Each literature test was compared with the paper it cites and with newer standard versions of the same
problem. Seven reading agents worked on the paper texts; every item under "Checked" was re-read by hand
in the paper text (quote and location given) and in the code. "For Uli" items could not be settled from
text available here or need a decision. "Dropped" lists agent claims that were wrong.
Paper texts and extracts: the session scratchpad (`rtlit/`, `dl_*/`, `papers_rs/`); not in the repo.

## Ranked: fix first

1. Moving Gresho bound uses the wrong grid's row (`test_hydro_hard_lw_gresho.py:107`).
2. Linear tests that never check they are linear: RT (fixed, below), KH (flattens like RT did), sound waves.
3. Self-referenced shock references (Shu-Osher, Titarev-Toro, Woodward-Colella) and published L1
   numbers left unused (LW Table 3.2, Greenough & Rider Table XIX).
4. STARBENCH D-type 8-10 % off the benchmark codes; the "converged" pair share one cell size.
5. Pancake gates stop before the caustic; the pressure floor hides the temperature check.
6. Evrard runs periodic and energy-only; the "~50 % expected" describes the old coupling.
7. Evolved Riemann ladder is Toro's exact-solver table, not the scheme-test set LW and Toro use.
8. Sink tests re-implement LUNA's own formulas; the coupling sink test is skipped wholesale.

## Rayleigh-Taylor (A13)

Checked:
- LW 2003 sec. 4.6 (pp. 1008-1009): "(x, y) ∈ (0, 1/6) × (0, 1) ... g = 0.1 in the −y direction ...
  interface ... y = 1/2 + 0.01 cos(6πx) ... smoothed out ... reflecting"; Fig. 4.5: 100 × 400, T = 8.5;
  pictures only. No gamma in sec. 4.6 (sec. 3.1: "all other tests use γ = 1.4", 1D). Box matches.
- P_I = 2.5 and gamma 1.4 are Springel 2010 sec. 8.8 ("P0 = 2.5 (with γ = 1.4)"), not LW. Docstring fixed.
- Athena 2008 sec. 8.3 runs (Lx, Ly) = (1/3, 1), v_y = 0.01 sin(6πx), pictures only.
- GIZMO sec. 4.4.2: logistic Δ = 0.025, i.e. a tanh of width 0.05; our 0.025 tanh is twice as sharp (ours).
- COSMOS sec. 3.3: "The computed growth rate is found to be 14% slower ... may be due to the relatively
  large initial amplitude". Fornax: "perfect agreement with linear theory up to t≃1.75" (figure only).
- No paper read runs RT as a linear convergence test with norms and orders. A13 is beyond practice.
- AMP = 1e-6 was nonlinear at 512 (hllc-mc L1 3.03e-5 -> 1.89e-5, step order 0.68); 1e-7 and 1e-8 agree
  to 3 digits (1.19e-5, step order 1.73). Not the limiter: MC, van Leer and unlimited all flatten.
  Fix in progress: seed +AMP and -AMP, use (U+ - U-)/2 (no drift, no even orders), AMP = 1e-7.

For Uli:
- Reckinger, Livescu & Vasilyev 2016, JCP 313, 181: the only lead for eigenfunction-seeded compressible RT.
  Paywalled; does it report growth-rate or norm convergence to compare with?

Dropped:
- Agent rule "k xi(T) <= 0.1 x finest error": too strict; at AMP = 1e-7 it fails yet 1e-7 = 1e-8 to 3 digits.

## Riemann problems and shock tubes

Checked:
- `src/luna/hydro/exact_riemann.py:8` evolves Toro's exact-solver table (all x0 = 0.5, test 1 plain Sod to
  0.25). LW Table 3.1 uses the scheme-test set: test 1 (1, 0.75, 1 | 0.125, 0, 0.1) x0 0.3 T 0.2 (sonic
  point), 3a stationary contact u = -19.59745 x0 0.8, 4 = Toro 5 at x0 0.4, 5, 6, peak. LW Table 3.2 per
  scheme relative L1 % (test 1: PPM 0.5 .. CFLFh 1.5; 3a: WAFT 2.6 .. CFLFh 10.3) is then usable.
- Einfeldt-Strong is Toro 2 (1, -2, 0.4 | 1, 2, 0.4): same states, identical bounds. Replace with
  LA-14379's vacuum case.
- Wave speeds are Davis min/max in HLLC and HLL (`fluxes.py:56-57`, `:133-134`). Athena: Einfeldt for HLLE
  (Stone 2008 eqs. 52-54), Batten et al. for HLLC. The flux tests pin the Davis choice.
- Stone 2020: "the HLLE approximate Riemann solver is chosen in order to suppress the Carbuncle-like
  instability ... The H-correction scheme in the Athena code ... suppresses these instabilities".
  The xfailed HLLC carbuncle is known HLLC behaviour with published cures.

For Uli:
- Switch to Einfeldt (HLL) / Batten (HLLC) speeds? A solver change: roofline and weak ladder, all bounds move.
- `test_hydro_hard_einfeldt.py:50` cites "Guo et al." with no reference anywhere. Which paper?
- Quirk duct seed (alternating ±1e-6) and LW Noh to T = 2 on 400² (we stop at 0.6): not re-read by hand.

## Shock problems against a stored reference

Checked:
- Shu-Osher, Titarev-Toro, Woodward-Colella use LUNA's own HLLC+MC at 2^16 cells. Greenough & Rider:
  "We choose to use a third method for generating a highly resolved solution to avoid potentially biasing
  of the results". Add an independent reference per problem.
- G&R Table XIX (WC blast, PLMDE, relative L1 vs N = 6400 PPMDE): 1.133e-1, 6.298e-2, 3.777e-2 at
  200/400/800. G&R: Shu-Osher "convergence rate is essentially zeroth order" for N < 1600.
- Sedov accuracy has no published counterpart (Kamm-Timmes: no standard setup); no velocity norm, no order.

For Uli:
- WC84 Table I (agent: MUSCL 0.162, 0.0996, 0.0553, 0.0401; PPMLR 0.0133 at 1200): the scan is garbled
  here, please read from your copy.
- Titarev-Toro text says "Mach number 1.1"; the agent's Rankine-Hugoniot check of their states gives 1.30.
- Sedov KE fraction: docstring says ~0.30, agent computes 0.283 for gamma 5/3 (not checked).

## Smooth and linear

Checked:
- Athena++ `hydro_linwave.py:104,112`: ratio err64/err32 <= 0.325 (HLLE 0.33, 0.34), i.e. order >= ~1.6,
  plus absolute bounds and exact L/R equality. `test_hydro_accuracy_linear_waves.py:66-67` and
  TEST_PLAN quote "ratio <= 0.25, order >= 2" (Athena's comment, not its threshold). Minmod 1D passes.
- Galilean KH boost is along x, parallel to the interfaces (`test_hydro_hard_galilean.py:84`); Robertson
  and Springel boost across them. The interface-advection case is untested.

For Uli (agent findings, not re-read by hand):
- Vortex is Shu's periodic box-10 version (Spiegel 2015), not YSD's; stationary Yee vortex L1 ladder
  (Pakmor 2016) missing; 100-period rows test dissipation only.
- No linearity check on the sound waves; a pointwise ±A check before extending ladders.
- Slotted cylinder: slot depth compared with a maximum-principle-preserving MUSCL curve using our MC;
  check min/max every step, not only at the end.
- `test_hydro_accuracy_smooth.py` duplicates A1 (sound) and A4 (Gresho at t = 1, 0.8 revolution).

## Vortex, low Mach, instabilities

Checked:
- LW Table 4.6 (moving Gresho), 160 × 40 worst of eight: vorticity 83 (CFLFh), density 0.77 (WAFT),
  TKE 30.9 (LL). Test (`:107`) uses (100, 1.12, 71.6): the 80 × 20 column, and 100 is JT not the worst.
  CLAW (MUSCL-type) at 160 × 40: 37, 0.29, 8.3. Static row (45, 0.23, 26.1) is right.
- KH linear (`test_hydro_accuracy_kh.py:9-10`): 0.0296, 0.00957, 0.00625 at 64/128/256, orders 1.63, 0.61.
  Seed is not the eigenmode, no linearity check. Likely the RT cure applies (not run).
- Low-Mach GPU variant runs t = 1, 0.8 revolution (`test_hydro_hard_low_mach.py:72`).
- Miczek et al. 2015 eq. 14: p0 = ρ0 u²/(γ M²) − 1/2; `gresho_vortex.py:24` omits the −1/2.
  Miczek Table 1 (Roe, 40²): KE kept 94.47 / 74.52 / 49.24 % at M = 0.1 / 0.01 / 0.001.
- Taylor-Green starts at ρ = 1 (`tests/_tg_reference.py:53`); HiOCFD C3.5: "The initial temperature
  field is taken uniform".

For Uli:
- Taylor-Green: follow C3.5 (uniform T) or keep ρ = 1 as a low-Mach test against the incompressible
  spectral reference?

## Gravity and balance

Checked:
- Springel 2010: "a relative error as large as ∼ 50% for the poorest resolution ... 24464 cells are inside
  the initial radius" (old cell-centred coupling). astronomix 2026: "simple source term ... up to 18%",
  flux-based "below 10−8"; flux-based crashed at 32³ without a positivity limiter.
- Berberich et al. 2019 (Comput. Fluids) sec. 8.3, isothermal atmosphere, 128 cells, t = 2, Std-O2:
  ρ 4.60e-4, ρu 7.24e-4, E 3.20e-3. The second-order reference A6 should cite.
- AMR Jeans flag uses the adiabatic sound speed (`amr/driver.py:203`); Truelove is isothermal.
- Polytrope sits in a uniform ambient (`test_hydro_hard_well_balanced.py:12`), not hydrostatic.

For Uli (agent findings, not re-read by hand):
- Evrard periodic L = 4R vs isolated: the agent's "6.5 % force at R, 2.2x initial energy".
- Jeans gates density only; Athena++ gates all five conserved variables.
- Mullen 2021 r⁻⁵ hydrostatic envelope as the modern polytrope setup.

## Cosmology

Checked:
- Standard pancake (Springel 2010, Hopkins 2015, Trac & Pen 2004): zi = 100, zc = 1, λ = 64 h⁻¹ Mpc,
  Ti = 100 K, shown after the caustic; Springel eq. 130 adiabatic T(x, z).
- `test_cosmo.py:34` runs a = 0.3 -> 0.6 (caustic at 1.5), one resolution, `LIMITER=1` (MC).
- `cosmo_p_floor = cs0²/γ` (`test_cosmo.py:45`, `run_pancake.py:40`) is enforced on P (`cosmo_floors.py`),
  lifting every underdense cell above its adiabat: the temperature check cannot be made (derived, not run).

Dropped:
- Agent: `test_cosmo.py` runs a = 0.2 -> 0.4. The code says 0.3 -> 0.6.
- Agent: coupled L1_GAS = 1e-2 is "4x the 2.3e-3 the gallery quotes". No 2.3e-3 found in the repo.

For Uli:
- DM growth: the agent's Garrison-2016 estimate (discrete deficit 0.987, matching the recorded
  0.984/0.987) would make the 3 % tolerance blind to a 1-2 % force error. Not checked.

## Radiation and sinks

Checked:
- Bisbas 2015 Table 3, mean R_IF at 0.14 Myr: 1.343 pc (1D codes), 1.324 pc (3D). LUNA 1.454 pc
  (`docs/RADIATION.md:113`): +8.3 %, +9.8 %.
- The "converged" runs 144³/3 pc and 240³/5 pc share dx = 0.0208 pc; Bisbas 3D: 128³ octant over
  4 R_St, dx = 0.0098 pc.
- Strömgren parameters match Bisbas sec. 4.1; the test is one equilibrium radius with an unresolved front.
- Sink Jeans check: E_th = Σ e_int dV (`species/sinks.py:389`), 1.5 P dV for γ = 5/3; separation default
  4 dx = 2 r_acc (`:64`, `:66`); E_grav uses the total potential (`:392`).
- `test_coupling_sink_formation.py` is skipped wholesale (points at the deleted `sources/sinks.py`).
- `test_wall_m3_ic.py:24` uses T = 40 K; runs and docs use 46 K.

For Uli:
- `docs/TORCH.md` says the sink rules follow FLASH's code. Does `Particles_sinkCreateAccrete.F90` use
  e_int and 2 r_acc? If so LUNA matches Torch and only the docs need the citation.
- STARBENCH prescribes an isothermal ionized EOS (agent); LUNA uses γ = 5/3 with a pressure reset.
- Iliev 2006 Test 1 (r_I(t)) and Test 3 (shadowing) are missing; M1's beam weakness is documented.
