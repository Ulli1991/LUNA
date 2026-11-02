# Thermochemistry: chemcool networks 5 and 17 in JAX

`luna/chemistry/` is a JAX port of **network 5** of Simon Glover's chemcool
(the SGChem module of Arepo; the copy ported is GIZMO's
`cooling/chemcool/`, now maintained by U. Steinwandel). Network 5 is Glover & Mac Low
(2007)'s H2/H+ network plus the approximate CO chemistry of **Nelson & Langer
(1997, ApJ 482, 796)**, "NL97". Phases 1 to 3 are done: a Fortran reference,
a function-by-function port of the right-hand side, and a batched,
differentiable stiff integrator that replaces DVODE (`luna/ode/rodas3.py`, the last
section). Network 17 (primordial H/He + NL97 CO + D+/HD) is ported the same
way; its section follows network 5's, and the integrator drives both.
Coupling into LUNA as a stage is phase 4.

## What the network is

State per cell: `y = (x_H2, x_H+, x_CO, e)`, abundances per H nucleus and the
internal energy density [erg cm^-3]. Everything else follows from them:
x_HI = 1 - 2 x_H2 - x_H+; all carbon not in CO is C+; x_e = x_H+ + x_C+ + x_Si;
free oxygen x_O = x_O,tot - x_CO; He is neutral with He:H = 0.1.

- **H2** forms on dust (Hollenbach & McKee 1979) and is destroyed by
  collisions with H, H2 and e-, by cosmic rays and by Lyman-Werner photons,
  shielded by dust and by its own lines (Draine & Bertoldi 1996).
- **H+** from collisional and cosmic-ray ionisation; recombines in the gas
  (case B) and on grains (Weingartner & Draine 2001).
- **CO** (NL97): C+ + H2 -> CH_x at k0 = 5e-16, then CH_x + O -> CO, the
  fraction beta = k1 x_O / (k1 x_O + Gamma_CHx / n) that is not
  photodestroyed first; CO is photodissociated at 1.235e-10 G0 e^{-2.5 A_V}.
- **e**: de/dt = -(28 cooling terms of `cool_func` + 6 chemical terms). The
  non-zero ones for network 5: gas-grain transfer, H2 rovibrational lines,
  CO rotational and vibrational lines (three isotopologues), cosmic-ray and
  photoelectric heating, OI 63/145 micron and CII 158 micron fine structure,
  SiII (if Si is present), Compton, H2 collision-induced emission, Lyman
  alpha; H2 formation/dissociation/pumping energy, H ionisation and
  recombination energy.

## The port

| module | Fortran | what |
|---|---|---|
| `tables.py` | coolinmo.F, cheminmo.F, spline.F | the rate and cooling tables network 5 reads (numpy, built once), Glover's cubic spline for the H2 and CO data, the e/kn -> T table |
| `thermo.py` | calc_temp.F | T and gamma from e and x_H2 |
| `photo.py` | calc_photo.F, photoinit_ism.F | H2 self-shielding, dust attenuation, chi(A_V), CO shielding (optional), the photo rates, for given TreeCol columns |
| `cooling.py` | cool_func.F, cool_util.F | the 28 cooling terms, dust temperature, CO cooling lookup, H2 opacity |
| `rates_net5.py` | rate_eq_simple.F (CHEMISTRYNETWORK == 5) | rate coefficients, `rhs(y, params) -> (ydot, aux)` |
| `../ode/rodas3.py` | evolve_abundances.F + dvode.F (replaced) | batched Rodas3 over dt, every cell at once; custom VJP; shared with nuclear burning |
| `networks.py` | chemcool_consts.h, evolve_abundances.F | networks 5 and 17 as Rodas3 `Network`s: tolerances, bounds, first-step floors |
| `gow17.py` | (GOW17, Visser+09, WD01) | the `fixes="gow17"` corrections, below |

Everything is pure, jittable and differentiable (`jax.jacfwd(rhs)` is tested
against finite differences), and needs `jax_enable_x64`. `aux` carries T,
gamma, the dust temperature, the 28 + 6 energy terms and `gross`, the sum of
|terms| of each component (the scale a net rate is accurate on).

Faithfulness choices that matter for 1e-10 agreement:

- **Single precision.** GIZMO compiles chemcool without `-fdefault-real-8`,
  so every undecorated literal is a 4-byte REAL: `BOLTZMANN`, `PI`,
  `PROTONMASS`, `GRAVITY`, `ELECTRON_VOLT`, `ABHE = 0.1`, the exponent 0.485
  of the H2-H2 dissociation rate, the OI polynomial coefficients, the chi(A_V)
  table, the molecular data... `tables.f32(x)` marks each one. Reading them
  in double moves the tables by 1.4e-4 relative (a teeth test).
- **Linear-in-T table interpolation** on the 10000-node log grid, with the
  Fortran's two edge conventions (rate_eq clamps above 1e8 K, cool_func
  extrapolates).
- **The dust temperature** is a Newton iteration from the previous value,
  stopped at the first step below 1 %; the Fortran carries T_dust as hidden
  state between calls. The port takes `params.tdust` in and returns the new
  one in `aux`, in a `while_loop` of at most 100 masked iterations that ends
  when every cell has stopped (forward-mode differentiable, as jacfwd needs).
  `params.dust_rtol` (default 1e-2) sets the stop; the integrator uses 1e-12.
- **Photo rates** follow `iphoto = 6` (GIZMO's TreeCol path): columns have a
  trailing pixel axis, each pixel weighted 1/n_pix.

Deliberate difference: cheminmo bakes the dust-to-gas ratio into the H2
formation table at start-up (`chtab(7)`); the port multiplies the current
ratio in. Identical when the ratio is constant, as in every reference run.

## Reference data

Fortran: `net5_ref` (read-only copy of the 43 source
files, md5-verified unmodified; gfortran 14.1, `-O2 -ffp-contract=off`, no
`-march=native`, default 4-byte REAL as in production). Everything that
builds and drives it is kept with the data, in
`$LUNA_DATA/chemistry/harness/net5/` (in neither repository): the config header
(CHEMCOOL, CHEMISTRYNETWORK=5, TREE_RAD, TREE_RAD_H2, TREE_RAD_CO, NSIDE=1),
a stub for the C++ `chemcool_fatal`, the one-zone driver, the dump driver and
the run scripts. **No line of the Fortran was changed.**

`energy-table.txt` (e/kn -> T, gamma; read by calc_temp.F) is not in the
chemcool source tree (GIZMO's .gitignore: "distribute via FTP"). The original
is Glover's chemcool file (GIZMO cooling module; all six
copies on raven identical, md5 f76eba8d484d3a390e2c28686b75d213). The port reads it as exact
doubles from `$LUNA_DATA/chemistry/energy_table.npz` (see Data files below).

Before the original turned up, a stand-in regenerated from calc_temp.F's own
e(T) relation with gamma = 1 + T/(e/kn) was used
(`tables.energy_table_regenerated`; the Fortran outputs from it are kept in
the reference directory as `*regenerated_table*`). Its T differs from the
original by up to 3e-4 (near 20000 K), its gamma by up to 15 % (median
2.3 %). Gamma enters network 5 only through the Jeans length (the CII escape
column, and the H2 opacity above 1e9 cm^-3), and still moved the references:
over the 168 runs, T at the end by up to 8.5 % (n = 1e3, G0 = 1.7, A_V = 1,
T = 40 K, where CII dominates; median 1.8e-5), x_H2 and x_CO at the end by
up to 2.6e-4, x_H+ by up to 3.4e-3; anywhere along the trajectories, x_H2
and x_CO by up to 4.2e-3 and x_H+ by up to 3.9e-2.

- **One-zone trajectories**, `net5_trajectories.npz`: n_H = 1..1e5 cm^-3,
  T0 = 20, 100, 1e3, 1e4 K, G0 = 0, 1.7, 10 Habing, columns thin /
  A_V = 1 (N_H = 1.87e21, N_H2 = 5e20, N_CO = 1e15) / A_V = 5 (9.35e21,
  4.5e21, 1e17), solar C and O (1.41e-4, 3.16e-4), zeta = 3e-17 s^-1,
  dgr = 1, x_H2 = 1e-6, x_H+ = 1e-4 at start; 168 cases (G0 = 0 only thin),
  250 log-spaced outputs from 1 yr to 1 Gyr, static gas. Each output holds
  x_H2, x_H+, x_CO, e, T, T_dust, ydot, the 28 cooling and 6 chemical terms,
  and the photo rates. All 168 ran; n >= 1e3 is in equilibrium to 1e-11 over
  the last decade, n = 100 to ~1e-2, n <= 10 shielded or dark cases still
  evolve at 1 Gyr (H2 formation time at n = 1 is ~Gyr).
- **Per-function dumps**, from the unmodified routines at sampled inputs
  that cross every branch (`make_inputs.py`, fixed seed): T-gamma lookup,
  chi(A_V), H2 shielding, calc_photo (also built with `-DCO_SHIELDING`),
  calc_dust_temp, co_cool, H2 opacity, the tables themselves, cool_func (3000
  states, n up to 1e11, T up to 3e8 K) and rate_eq (1500 states). A compact
  subset is `$LUNA_DATA/chemistry/chemistry_net5_golden.npz` (1.3 MB).

Rebuild (with the original `energy-table.txt` copied into
`<dir>/gizmo/cooling/chemcool/`; `H=$LUNA_DATA/chemistry/harness/net5`):
`sbatch --export=ALL,REF=<dir>,TOOLS=$H $H/run_ref.sbatch` (91 s on apudev),
then `python $H/collect_golden.py <dir>`.

## Notes on the code (tables, photo rates, the EOS table)

- `tables.py`: the Fortran tabulates every temperature-dependent coefficient on a log grid
  of 10000 temperatures (coolinmo.F, cheminmo.F) and interpolates linearly in T; the module
  rebuilds exactly the rows networks 5 and 17 read, from the same formulas, plus the
  spline-fitted CO cooling tables and the e/kn -> T table, as plain numpy arrays.
  gfortran reads an undecorated literal (`0.485`, no `d`) as a 4-byte REAL: the port writes
  `f32(0.485)`; an all-literal expression such as `-22.190316 + 1.5728955` is even evaluated
  in single precision (`_f32_sum`). Files named in the code are in gizmo/cooling/chemcool/.
- `chem_table`: ch7 is returned per unit dust-to-gas ratio (the Fortran bakes in the value at
  cheminmo time). `chem_table17`: flags as GIZMO sets them, iflag_ad = iflag_mn =
  iflag_3bh2a = iflag_3bh2b = 1.
- `co_tables`: rotational 5..2000 K x log N 14.5..19.0 (L0, LTE, n_1/2, alpha in the data's
  log form), vibrational LTE 100..4000 K x 13.0..19.0; `d*` are forward differences per 1 K.
- `energy_table`: Glover's original energy-table.txt (md5 f76eba8d484d3a390e2c28686b75d213),
  exact doubles in `$LUNA_DATA/chemistry/energy_table.npz`. `energy_table_regenerated`, the
  stand-in used before: inverts calc_gamma_temp's e/kn = x/(1+abhe-x) E_H2(T) +
  (1+abhe-2x)/(1+abhe-x) 1.5 T (first crossing over the eh2-table dip at 20000 K), gamma =
  1 + T / (e/kn); T agrees with the original to 3e-4, gamma differs by up to 15 %.
- `thermo.temperature_gamma_table` (calc_temp.F:167-217): linear in ekn between integer ekn,
  linear in x_H2 between the 0.1-spaced blocks; x_H2 < 0.1 blends with the atomic
  T = (2/3) ekn; outside 1 <= ekn <= 5e4, or x_H2 = 0: atomic gas.
- `photo.py` (calc_photo.F, photoinit_ism.F): the far-UV field G0 (Habing; Draine is 1.7) is
  attenuated by dust along each TreeCol direction and the H2 (optionally CO) lines shield
  themselves; GIZMO uses 12 pixels (NSIDE = 1), each weighted 1 / n_pix (iphoto = 6). CO
  photodissociation NL97 (ApJ 482, 796); H2 self-shielding DB96 eq. 37 (ApJ 468, 269) per
  pixel column (TreeCol + local); CO self- and H2-shielding Lee+96 (A&A 311, 690), compiled
  only with CO_SHIELDING (no GIZMO configuration sets it; calc_photo.F:409-476). chi(A_V)
  (calc_photo.F:683, Mathis+83 ISRF, isrf_option = 1) is linear between integer A_V and
  constant above 200. `photo_rates` returns k_h2_dust (before self-shielding), k_co, g_dust
  = G0 e^{-2.5 A_V} and chi; av_local (the smoothing-length column) enters A_V but not the
  LW dust factor; G0 = 0 zeroes everything (calc_photo.F:122).

## Data files

No data file is in the LUNA repository. The tables the port loads and the
Fortran reference the tests compare against, and the Fortran harness that
makes it, live in `$LUNA_DATA/chemistry/`;
code and tests stop with a clear error if it is missing.

| file | what | made by |
|---|---|---|
| `net5_raw.npz` | chemcool DATA statements and data files | `tools/chem_net5_tables.py <chemcool dir>` |
| `energy_table.npz` | Glover's energy-table.txt | `tools/chem_net5_tables.py --orig-energy-table <file>` |
| `chemistry_net5_golden.npz` | Fortran per-function reference | `harness/net5/collect_golden.py` |
| `chemistry_net5_trajectories.npz` | DVODE one-zone trajectories, packed | `tools/chem_integrator/pack_reference.py` |
| `cie_tables.npz` | WSS_CIE_COOL tables (wss_z_collis.txt, HeI/HeII fractions) | `tools/chem_net5_tables.py --cie <chemcool dir>` |
| `chemistry_cie_golden.npz`, `chemistry_cie_trajectories.npz` | WSS_CIE_COOL Fortran reference | `harness/cie/collect_golden.py` |
| `co_shield_visser09.txt` | Visser+09 CO shielding grid (Leiden `shield.03.5.69-557-36.dat`, 12CO block) | verbatim copy |

Point `LUNA_DATA` at the data directory: `export LUNA_DATA=<dir>`.

## Agreement (measured, FP64, CPU backend)

| function | max relative error vs Fortran |
|---|---|
| table values, all ported rows | 5.1e-13 |
| rate coefficients at arbitrary T | 4.4e-13 |
| T, gamma from e | 0 |
| chi(A_V) | 0 |
| H2 self-shielding | 1.0e-15 |
| photo rates (and with CO shielding) | 4.2e-16 (1.7e-14) |
| dust temperature, R_H2 | 6.0e-16 |
| CO cooling lookup | 8.3e-15 |
| H2 opacity | 9.1e-15 |
| 28 cool_func terms + T_dust | 4.5e-12 |
| RHS: ydot / gross, 34 energy terms, T_dust | 1.8e-13 (also jitted) |
| RHS at all 42168 trajectory states | ydot 4.2e-15, cooling 5.0e-14, chemical 3.0e-13 |
| jacfwd vs central differences (5 states) | <= 3.0e-9 |

The one exception to relative agreement: above ~1.5e7 K the gas-grain table
is denormal; XLA flushes it to zero, the Fortran returns ~1e-288 erg/s/cm^3.
Every test also has teeth: 19 deliberate breaks (a table row shifted by one
node, rates 1 % off, the lookup's single-precision 0.1 read in double, the
regenerated energy table instead of the original, ...)
each make the matching comparison fail (`tests/test_chemistry_net5.py`).

## Things in the Fortran that looked wrong (flagged, not fixed)

1. **`eh2-table.txt` has a spike**: its T = 20000 K row is 30000 (1.5 T)
   between 59957.9 and 60003, so e(T) is not monotonic there.
2. **Without WSS_CIE_COOL there is no atomic cooling above 1e4 K except
   Lyman alpha**: `iflag_atom = 2` zeroes the He and bremsstrahlung rows
   because the SD93 table "includes them", but cool_func then sets that table
   term `rates(3) = 0`. Hot gas cools only by Lyman alpha and fine structure
   (and, ionised and lit, is heated by the photoelectric term: see below).
3. **G0_dust is never set in GIZMO's default path** (only with G0_VARIABLE or
   M1), so the ISRF does not heat the dust there. The reference sets
   G0_dust = G0.
4. **H2 formation on dust uses the start-up dust-to-gas ratio** (chtab(7) is
   filled once in cheminmo), although dust_to_gas_ratio is set per particle.
5. **Hidden state in the RHS**: calc_dust_temp starts Newton from the last
   T_dust and stops at 1 %, so rate_eq's result depends on call history at
   the 1e-2 level in T_dust. Its Newton derivative also omits the
   dust-to-gas factor of the gas-grain term.
6. **co_cool indexes the vibrational column with the rotational one**
   (cool_util.F:131). Harmless only because cool_func always passes equal
   columns.
7. **CO_SHIELDING is in no configuration**: GIZMO's CO photodissociation is
   dust-shielded only, no CO self-shielding or H2 line overlap. With it, the
   two pixel means are multiplied, not the per-pixel products.
8. **In iphoto = 6 the LW dust factor ignores the local column** (dl n_H),
   while A_V for CO, photoelectric heating and chi includes it.
9. **chi(A_V = 200) exactly stops the program** ("Index = 200... Whoops.").
10. **Dust can cool below the CMB** (no 2.73 K floor): dark, dense reference
    cases reach T_dust ~ 1.5-3 K.
11. Small ones: `temp_2` reads one element past its block at e/kn = 5e4
    exactly (weight 0); rate_eq's `ipar` for T <= 0 does not zero ydot.

## Corrections from GOW17 (`fixes="gow17"`)

Gong, Ostriker & Wolfire 2017 (ApJ 843, 38; "GOW17") show three things in
chemcool's networks that are wrong, not only approximate. They are fixed behind a switch so the
faithful port stays available for comparison with the Fortran:
`rates_net5.rhs(y, p, fixes="chemcool" | "gow17")`. `"chemcool"` is the
default of the function and is bitwise the port above. Only the CO row of
ydot changes; H2, H+ and the energy are untouched at a given state. The new
rates live in `luna/chemistry/gow17.py`; `tests/test_chemistry_gow17.py`
checks each against the published formula or table and breaks each one.

1. **CO photodissociation** (both networks). chemcool: 1.235e-10 G0
   e^{-2.5 A_V} (photoinit_ism.F:101, vDB88), with CO self- and H2 shielding
   only under `CO_SHIELDING`, which GIZMO never sets. GIZMO's network 5 does
   get a CO column (`TREE_RAD_CO`, set by cool.h for every network but 1, 2
   and 4), but calc_photo.F:510-516 applies it to the CO rate only under
   `CO_SHIELDING`; the CO column is used nowhere else (CO line cooling takes
   its own LVG column, x_CO n / |div v|). Fixed: GOW17 eq. 8 and Table 2 #16,
   `2.4e-10 chi <e^{-3.88 A_V} Theta(N_CO, N_H2)>`, chi = G0 / 1.7, the mean
   over pixels of the per-pixel product. 2.4e-10 and 3.88 are one pair from
   Heays et al. 2017 (A&A 602, A105), Table 18, CO, "ISRF" and
   gamma_exp,ISM. The 3.53 in GOW17's text after eq. 25 is the older value of
   van Dishoeck et al. 2006 that Visser et al. 2009 (Sect. 3.4) quote with
   their 2.6e-10; it belongs to the old rate, so 3.88 is used.
   Theta is Visser, van Dishoeck & Black 2009 (A&A 503, 323): their Leiden
   file `shield.03.5.69-557-36.dat` (b(CO) = 0.3 km/s, T_ex = 5 K,
   12C/13C = 69, the parameters of their Table 5 that GOW17 use), 12CO block,
   kept verbatim as `$LUNA_DATA/chemistry/co_shield_visser09.txt` with its source and md5.
   It is Table 5 on a 5x finer grid and equals Table 5 at all 48 of its nodes
   (tested). Interpolation: bilinear in log Theta against (log N_CO,
   log N_H2), columns clamped to the grid (below 1 cm^-2 counts as none).
   N_CO and N_H2 are inputs per pixel, like the H2 shielding columns, plus the
   local x n dl. Where LUNA's columns come from (a local Jeans or Sobolev
   length first) is a separate decision.
2. **Cosmic-ray destruction of CO** (network 5 has none, so all carbon ends
   in CO in dark gas). Network 5 has no He+, so He+ is put in equilibrium
   (its lifetime, estimated from the GOW17 sinks, is ~1/(8e-14 n) s, ~4000 yr
   at n = 100):
   x_He+ = 1.1 xi x_He / [n (a_gr,He+ + (k7 + k25) x_H2 + k14 x_e + k8 x_CO)],
   and CO is destroyed at k8 x_He+ n + 90 xi f_mol per CO, with
   k8 = 1.6e-9 (He+ + CO -> C+ + O + He, T1 #8), k7 = 1.26e-13 e^{-22.5/T}
   (T1 #7), k25 = 7.2e-15 (T1 #25), k14 the HS98 case B fit (T1 #14),
   a_gr,He+ the WD01 grain recombination (T2 #4, with chemcool's phi), 1.1 xi
   the CR ionisation of He (T2 #8), 90 xi f_mol the CR-induced photons
   (T2 #12, f_mol = 2 x_H2 / (x_HI + 2 x_H2)). The products count as C+
   (network 5 has no neutral C). CR ionisation of CO (T2 #10) is left out:
   in GOW17 it makes HCO+, which recombines to CO.
   **zeta vs xi.** chemcool's zeta is the total rate (H at zeta, H2 at
   2 zeta). GOW17's xi_H is primary; their total is xi (1.5 x_HI + 2.3 x_H2)
   per H atom and twice that per H2 (T2 #6-7, App. A.1 item 3). Equal total
   ionisation per neutral H nucleus gives zeta = xi (1.5 x_HI + 2.3 x_H2) /
   (x_HI + 2 x_H2) in neutral gas: xi = zeta / 1.5 in atomic gas (xi = 2e-16 <-> zeta =
   3e-16), zeta / 1.15 in molecular gas (`gow17.xi_from_zeta`).
3. **He+ grain recombination for network 17** (chemcool's net 17 has He+ +
   CO, rate_eq_ulli.F:865, but no grain sink for He+, GOW17's mechanism 2).
   `gow17.he_grain_rate(g_dust, T, n, x_e, dgr)` is the WD01 rate with the
   coefficients chemcool tabulates (cheminmo.F:690-694, chtab 135/136) and
   rate_eq.F:364-371's form, for network 17's He+ sinks
   (`-he_grain_rate(...) * x_He+`). Not wired yet: `rates_net17.rhs` has no
   `fixes` switch (its `fixes17` is a different set of fixes).

Not changed, on purpose: NL97's lumped CO formation (k0, k1, Gamma_CHx;
GOW17's CH_x + H alone would starve it without the OH_x route), the
H2/H+ chemistry, the cooling.

**Effect sizes** (measured, one-zone, not yet validated against GOW17's
full network: that needs the phase-3 integrator and a slab test against
GOW17 Fig. 1-2). The 168 reference conditions, static gas, integrated to
1 Gyr in both modes with scipy BDF on the RHS (`tools/chem_gow17_onezone.py`;
the chemcool mode reproduces the Fortran's end state to 1.5e-4 in x_H2 and
x_CO and 8e-3 in T at n >= 1e3). x_CO(gow17) / x_CO(chemcool) at 1 Gyr,
T0 = 20 K, n = 1, 10, 1e2, 1e3, 1e4, 1e5 cm^-3:

| columns, G0 | zeta = 3e-17 | zeta = 3e-16 |
|---|---|---|
| none, 0 (dark) | 0.002, 0.04, 0.48, 0.93, 0.99, 1.00 | 9e-5, 0.002, 0.04, 0.50, 0.93, 0.99 |
| A_V = 5, 1.7 | 0.02, 0.06, 0.50, 0.93, 0.99, 1.00 | 0.002, 0.004, 0.05, 0.50, 0.93, 0.99 |
| A_V = 1, 1.7 | 26, 28, 28, 22, 6.5, 1.6 | 13, 14, 16, 16, 6.2, 1.6 |
| A_V = 1, 10 | 29, 30, 30, 30, 19, 4.5 | 23, 24, 26, 26, 18, 4.5 |
| none, 1.7 or 10 | 0.87 at every n | 0.87 |

So: in dark gas the cosmic rays now set a density floor for CO (half the
carbon in CO at n ~ 1e2 for zeta = 3e-17, ~1e3 for 3e-16; GOW17's eq. 25
puts it near 15 and 100-130 cm^-3, so network 5 still forms CO more slowly
than GOW17, an interpretation of the eq. 25 numbers); at A_V ~ 1 CO is 1.6-30x more abundant (dust exponent and
self-shielding); unshielded CO is 13 % lower (2.4e-10 / 1.7 vs 1.235e-10).
x_H2 moves by at most 5 %. T at 1 Gyr moves by up to -52 % in the dark gas
that loses its CO (n <= 1e2 at zeta = 3e-17, n <= 1e3 at 3e-16; 59 -> 28 K
at n = 10, -23 % at n = 1e3; the freed carbon becomes C+, which our
interpretation is cools more there than the CO it replaces) and by under
10 % elsewhere. The n <= 10 cases are not in
equilibrium at 1 Gyr in either mode.

**Proposed default for LUNA runs: `"gow17"`.** The three corrections replace
rates that are outdated or missing, not calibration choices, and the
production analogue (CO in dark and translucent gas) is exactly where they
act. `"chemcool"` stays the mode for any comparison with GIZMO or the
Fortran. Before science use, the slab test above should pass.

# Network 17: primordial H/He, NL97 CO and D+/HD

Network 17 is S. Glover's primordial H/He network with the NL97 CO model,
to which U. Steinwandel added the D+/HD channels of network 1
(`rate_eq_ulli.F`). It is the network for low-metallicity gas: He and the
deuterium chemistry matter there, and C, O and dust scale with Z down to 0.

## What the network is

State: `y = (x_H2, x_H+, x_CO, x_He+, x_He++, x_D+, x_HD, e)` per H nucleus
(TRAC_NUM 7, NSPEC 8). Derived: x_HI = 1 - 2 x_H2 - x_H+, x_HeI = abhe -
x_He+ - x_He++ with **abhe = 0.079** (a REAL*4 literal, as for network 1),
x_DI = abund_D - x_D+ - x_HD, x_C+ = x_C - x_CO, electrons
x_e = x_H+ + x_He+ + 2 x_He++ + x_C+ + x_D+ (no Si).

- **Three equilibrium species** (num_eqb_species = 3): H-, H2+ and HeH+.
  rate_eq solves them inside every RHS call by fixed-point sweeps: each
  sweep sets H- = formation/destruction (`comp_equilib`), corrects x_e and
  x_HI by the change, then H2+ (with the new H-), then HeH+ (with the new
  H2+), correcting x_e, x_HI, x_HeI after each. A result above the species'
  maximum (1, 0.5, abhe), or a zero denominator with formation, gives 0.
  Sweeps stop when all three changed by less than 1e-3 relative, from the
  second sweep on, or after 25. The corrected x_e, x_HI, x_HeI (not the
  inputs) enter every rate after the loop and cool_func.
- **H2**: dust (HM79), H- + H, H2+ + H, three-body (H, H2 and He as third
  body); destruction by H (MSM96 collision-induced + dissociative
  tunnelling, density-interpolated), H2, He (DRCM87), e-, H+ (charge
  transfer, interpolated to its LTE limit), He+, LW photons (DB96
  shielding), cosmic rays.
- **H+, He+, He++**: collisional ionisation (by e-, and of H by H and He),
  case-B recombination, grain recombination of H+, charge transfer with He
  and D, He+ + H2, cosmic rays (He at 1.09 zeta).
- **D+, HD**: D/H charge transfer, D+ + H2 <-> HD + H+, D + H2 <-> HD + H,
  three-body HD formation at the H2 rates, HD photodissociation (shielded);
  all off when abund_D = 0.
- **CO**: NL97 as in network 5, plus destruction by He+.
- **e**: the 28 cool_func terms (network 5's, with He as a collider for the
  H2 and CO lines) and 22 chemical terms (H2 dissociation and formation
  energies, H/He/He+ ionisation, H+/He+/He++ recombination, H- formation,
  UV pumping at 6.94 pumps per dissociation, dust H2 formation).

## The port

`rates_net17.py` holds what is network-specific: the rate assembly, the
network-17 branch of calc_photo (`photo_rates`), the equilibrium sweeps
(`equilibrium_species`, a masked `lax.scan` of 25 sweeps) and
`rhs(y, Params) -> (ydot, aux)`, the same shape as network 5's. Shared code
was extended, not copied: `tables.chem_table17()` (the 57 cheminmo rows
network 17 adds) and `tables.cool_table_hd()`, He as a collider and HD
cooling in `cooling.cool_func`, `photo.hd_self_shielding`, and the grain
recombination and UV-pumping helpers now shared with network 5.
`tables.ipow` multiplies integer powers in gfortran's order (its powi
chain); it also brought network 5's tables closer to the Fortran.

calc_photo runs once per hydro step in GIZMO, so the HD shielding and every
photo rate are fixed at the step start. `Params.photo_temp`, `photo_x_h2`,
`photo_x_hd` carry that state (`photo_state(y, p)` computes it as
chemcool.cc does, electrons = H+ only); left `None`, the RHS uses the
current state. H2 self-shielding is evaluated at the current state.

## Reference data

Fortran: `net17_ref` (copy of the 43 source files,
md5-verified unmodified; Glover's `energy-table.txt`, md5
f76eba8d484d3a390e2c28686b75d213). Build and drivers:
`$LUNA_DATA/chemistry/harness/net17/` (config CHEMCOOL, CHEMISTRYNETWORK=17, TREE_RAD,
TREE_RAD_H2, TREE_RAD_CO, NSIDE=1, plus OUTPUT_INDIVIDUAL_COOLRATES, which
only copies the 22 chemical terms into `lambda_chem`). Flags as for
network 5. **No line of the Fortran was changed**; one build change:
`rate_eq_ulli.F` is compiled with `-fno-automatic`, because its local `NH`
is read before it is assigned (finding 1 below). With it, `NH` holds the
previous call's 0.5 dl n_H; the dump driver calls rate_eq twice per state,
the one-zone runs have dl = 0. The port uses NH = 0.5 dl n_H.

- **One-zone trajectories**, `net17_trajectories.npz`, 288 runs, all
  finished: n_H = 1..1e5 cm^-3, T0 = 20, 100, 1e3, 1e4 K, G0 = 0 and 1.7
  Habing, thin and A_V = 1 columns (N_H = 1.87e21, N_H2 = 5e20,
  N_CO = 1e15; A_V = 1 at solar dust; G0 = 0 only thin), Z = 1, 1e-2, 1e-4
  and 0 solar (C = 1.41e-4 Z, O = 3.16e-4 Z, dust-to-gas = Z; Z = 0 takes
  calc_photo's dust-free branch). Fixed: D/H = 2.6e-5, zeta = 3e-17 s^-1,
  start x_H2 = 1e-6, x_H+ = 1e-4, x_D+ = 1e-4 D/H, no He+, He++, HD, CO;
  250 outputs from 1 yr to 1 Gyr. Each step mirrors chemcool.cc: calc_photo
  at the step start, its input clamps, evolve_abundances; then one
  diagnostic rate_eq records ydot, the 28 + 22 terms and the photo rates.
- **Per-function dumps** (`make_inputs.py`, fixed seed): the 66 chtab rows
  and the HD, He and H2-He cltab rows network 17 reads, constants, rate
  coefficients at 2600 temperatures, calc_photo (800 inputs, with dust-free
  and D-free ones), HD shielding (1500), cool_func with He and HD (3000),
  the equilibrium species (a marked copy of rate_eq_ulli's loop in the dump
  driver, since they are rate_eq locals) and rate_eq itself (2000 states,
  Z = 0..1, D on and off, invalid states). A subset, with every 25th
  trajectory output, is `$LUNA_DATA/chemistry/chemistry_net17_golden.npz`.

Rebuild: `sbatch --export=ALL,REF=<dir>,TOOLS=$LUNA_DATA/chemistry/harness/net17
run_ref.sbatch`, then `collect_golden.py <dir>` (same directory).

## Agreement (measured, FP64, CPU backend)

| function | max relative error vs Fortran |
|---|---|
| table values and slopes, 66 chtab + 11 cltab rows, constants | 5.7e-12 |
| rate coefficients at arbitrary T | 3.0e-11 |
| calc_photo, network-17 branch | 1.0e-15 |
| HD self-shielding | 8.8e-16 |
| 28 cool_func terms with He and HD, T_dust | 4.5e-12 |
| equilibrium H-, H2+, HeH+, corrected x_e, x_HI, x_HeI | 1.6e-13 |
| RHS: ydot / gross, 50 energy terms, T, gamma, T_dust, photo (also jitted) | 6.4e-13 |
| RHS at all 72288 trajectory states | ydot 2.4e-12, cooling 2.5e-14, chemical 1.1e-11, photo 1.1e-15 |
| jacfwd vs central differences (6 states) | <= 2.6e-8 |

The same checks after the merge with chem-integrator (pixel loop order,
eager tables, dust_rtol): chemcool mode tables 5.7e-12, rate coefficients
3.0e-11, cool_func 4.5e-12, equilibrium 1.6e-13, RHS 6.4e-13, all
trajectory states ydot 2.4e-12 / chemical 1.1e-11; the fixed mode against
the fixed Fortran (next sections) 3.0e-11 at most, the same rows.

The ch24 fit has a zero crossing; there the table agrees to 3e-11 only with
gfortran's multiplication order for the integer powers (numpy's `**` gave
3.3e-10). Sweep counts agree except in 3 of the 1000 golden states,
where 1e-3 x of some species is denormal: XLA flushes it to zero, so the
port sweeps on to 26, at the same abundances. Teeth: 15 deliberate
breaks (a row shifted, rates 1 % off, single-precision literals or abhe
read in double, network 5's 8.5 UV pumps, one sweep only, NH = 0, the HD
shielding weight "fixed", ...) each make their comparison fail
(`tests/test_chemistry_net17.py`).

## Things in the network-17 Fortran that looked wrong

Items 1-6 have a fix behind `fixes17="fixed"` (next section); the default
mode keeps them, as chemcool does.


1. **`NH` is used uninitialised in rate_eq_ulli.F**: the H2 self-shielding
   call (line 510) reads the local `NH` before line 602 assigns it
   (0.5 dl n_H). Production gets whatever the stack holds.
2. **HD cooling is never on**: rate_eq_ulli never sets `abundances(13)`,
   so cool_func's HD term (R19) is always 0, at any HD abundance. The port
   has the term (tested against cool_func with HD) but, like the Fortran,
   does not pass HD.
3. **HD self-shielding reaches 1.55**: in the iphoto = 6 loop of
   calc_HD_shielding_factor only the first term carries the pixel weight
   1/12; the second (0.05 exp(...)/sqrt(1+x) and its variants) is added 12
   times. The dumps reach 1.549, so optically thin HD is photodissociated
   1.5x faster than unshielded.
4. **With dust, H- photodetachment and H2+ photodissociation are zero**:
   for iphoto = 6 calc_photo sets df05 and df19 to 0 and never accumulates
   them, so ph4 = ph5 = 0 whenever dust_to_gas_ratio > 0; at exactly 0 the
   other branch gives them unattenuated.
5. **The energy table assumes He:H = 0.1**: the same energy-table.txt maps
   e/kn to T at abhe = 0.079. With calc_temp.F's own e(T) relation
   (`tables.energy_table_regenerated`) the two He fractions give T differing
   by up to 2.3 % (x_H2 = 0.5 block; median 0.8 %).
6. **ch100 (D+ + H2 -> HD + H+) is negative above 4.3e6 K** (the GP02
   quadratic in log T); hidden in production by the 5e5 K hot-gas bypass.
7. **`lambda_chem` is filled only with OUTPUT_INDIVIDUAL_COOLRATES**;
   without it chemcool.cc's cooling-rate diagnostic adds stale values.
8. **Photo rates are frozen at the step start** (the HD shielding uses x_H2,
   x_HD and T of the step start); H2 self-shielding is not.
9. Smaller: rate_eq_ulli uses its own double proton mass 1.6726d-24 for the
   Jeans length (elsewhere the REAL*4 PROTONMASS); validate_input clips He+,
   He++, D+, HD to 1, not to their pools; `no_dchem` is never set (zero in
   practice); the ch24 fit is negative (|value| < 1e-100, then replaced) at
   some low T; recombination cooling is not switched off below 100 K, unlike
   network 5; UV-pumping heating uses 6.94 pumps, network 5 8.5.

## The fixes (`fixes17="fixed"`)

`rates_net17.rhs(y, p, fixes17="fixed")` (also `photo_state`,
`photo_rates`, `chem_coefficients`; `networks.NET17_FIXED`) switches on
fixes for items 1-6 above. The default `fixes17="chemcool"` is the port
as before, bit for bit (35 arrays of rhs output, eager and jitted, network
5 included, compared against the merge commit). Fixes 1-4 and 6 are those
in the GIZMO version maintained by U. Steinwandel (commit a58f5faf, cooling/chemcool):

1. HD cooling on: `abundances(13) = abhd`; the port passes x_HD to
   cool_func's term 19.
2. `NH = dl n_H` before the H2 self-shielding call (`NH_LOCAL_FIXED`; the
   chemcool mode keeps 0.5 dl n_H, what -fno-automatic left there).
3. The pixel weight 1/n_pix multiplies both terms of the HD shielding fit
   (`hd_self_shielding(..., weight_both=True)`); thin HD shielding is 1.
4. df05 and df19 are accumulated over the pixels like the other dust
   factors (ph4 = H- photodetachment uses df05, ph5 = H2+ photodissociation
   df19); in GIZMO they are 1 under RADTRANSFER / TREE_RAY, a case the LUNA
   chemistry has no switch for yet (phase 4: under M1 all dust factors are 1).
5. The Newton EOS at the network's He/H = 0.079 (`eos="newton"`, LUNA's
   own; the fixed mode uses it unless `eos="table"` is passed).
6. ch100 (D+ + H2 -> HD + H+) clamped at 0 (`tables.chem_table17_fixed`).

**Reference**: `net17_fixed`, the same 43 files
with GIZMO a58f5faf's three changed ones, built as net17_ref but without
`-fno-automatic` (`make NOAUTO=`; NH is now assigned before use), same
dumps and the same 288 one-zone runs. Golden subset:
`$LUNA_DATA/chemistry/chemistry_net17_fixed_golden.npz`. The fixed mode with
`eos="table"` matches it as the chemcool mode matches net17_ref: every
check of `tests/test_chemistry_net17.py` runs in both modes at 1e-10, and
undoing any one fix (HD cooling off, NH = 0.5 dl n_H, the old shielding
weight, ph5 with df05 or ph4 with df19, ch100 unclamped) fails its fixed
comparison. One more sweep-count exemption: in one golden state the HeH+
formation term is denormal (1.5e-308) in the fixed run only; XLA flushes
it, the port sweeps on, the abundances agree to 4e-14.

**One-zone effect** (Fortran, chemcool vs fixed, `harness/net17/fixes_effect.py`;
T0 = 100 K, G0 = 1.7, dl = 0 so fix 2 does not act here; values at 1 Gyr):

| case | T [K] | x_H2 | x_HD |
|---|---|---|---|
| n = 10, thin, Z = 1e-2 | 2474 -> 2720 | 3.7e-7 -> 2.1e-8 | 2.1e-11 -> 1.5e-12 |
| n = 10, thin, Z = 1e-4 | 5258 -> 6844 | 5.8e-7 -> 7.1e-8 | 5.4e-11 -> 7.8e-12 |
| n = 10, A_V = 1, Z = 1e-4 | 309 -> 747 | 2.5e-3 -> 2.2e-4 | 2.5e-7 -> 1.5e-8 |
| n = 1e3, A_V = 1, Z = 1 | 37.5 -> 37.5 | 0.499 -> 0.499 | 5.9e-6 -> 2.4e-5 |
| n = 1e3, A_V = 1, Z = 1e-4 | 120 -> 56 | 9.5e-3 -> 4.1e-3 | 8.4e-6 -> 2.4e-5 |
| n = 1e3, A_V = 1, Z = 0 | 132 -> 58 | 6.2e-3 -> 2.6e-3 | 4.8e-6 -> 2.2e-5 |
| n = 1e5, A_V = 1, Z = 1e-2 | 38.5 -> 35.7 | 0.484 -> 0.485 | 2.1e-5 -> 2.6e-5 |
| n = 1e5, A_V = 1, Z = 1e-4 | 73.3 -> 29.8 | 0.147 -> 0.120 | 2.6e-5 -> 2.6e-5 |
| n = 1e5, A_V = 1, Z = 0 | 93.6 -> 29.2 | 0.056 -> 0.016 | 2.3e-5 -> 2.6e-5 |

Two effects dominate. At low n with a field, ph4 > 0 (fix 4) destroys H-,
the main H2 route without dust: x_H2 drops 3-18x and the gas stays warmer.
In shielded gas at low Z, HD now cools (fix 1) and the thin HD shielding is
no longer 1.55 (fix 3), so x_HD rises towards D/H and T falls by 2-3x
(down to the ~30 K HD floor at n = 1e5). Solar-Z gas is barely affected
except for x_HD (up to 4x). Fix 5 alone (LUNA integrator, fixed mode,
table -> Newton) changes these end states by <= 0.25 % in T (x_H2 = 0.5
gas, n = 1e3: 37.45 -> 37.36 K): in thermal equilibrium T is set by heating
= cooling, so the EOS moves the energy, not T; it matters for gamma and for
the T of a given energy in the hydro.

# Temperature by a Newton solve (`eos="newton"`)

`thermo.temperature_gamma(x_H2, ekn, abhe, eos)`. `eos="table"` (the
default everywhere, bit for bit the Fortran's lookup) reads Glover's
energy table, which is built for He/H = 0.1 only. `eos="newton"` solves
the relation the table encodes,

    ekn(T; a, x) = [x e_H2(T) + 1.5 (1 + a - 2x) T] / (1 + a - x),
    gamma = 1 + 1 / (d ekn / dT),

for T at the cell's own He/H `a` and x = x_H2, with a He-free table of the
H2 internal energy e_H2(T) (per molecule, / k).

**e_H2 from the table.** Each x block of energy-table.txt gives
e_H2 = (ekn (1+a0-x) - 1.5 (1+a0-2x) T) / x and, from gamma,
e_H2' = ((1+a0-x)/(gamma-1) - 1.5 (1+a0-2x)) / x, with a0 = 0.1 (the
double 0.1 fits better than the REAL*4 one). The five blocks agree: e_H2
to a median 2-4e-10 (max 9e-5 at 20-30 K), e_H2' to a median 3-6e-5. The
slope is the noisier one because the table's gamma is: it zigzags by ~1 %
at 20-30 K, and at 19970-20030 K it ramps from 3.3 to 3.0 while e_H2 is
already exactly 3 T (the blocks place that ramp at different T, so e_H2'
differs there by up to 55 % between blocks). Below 4 K e_H2 = 1.5 T (to 1e-13, the
table's rounding; the shipped table sets it exactly), above 2.003e4 K
e_H2 = 3 T exactly.

**The e_H2 table** (`$LUNA_DATA/chemistry/eh2_table.npz`, built by `tools/chem_eos/build_eh2_table.py`):
e_H2 and e_H2' of the x = 0.1 block (the one reaching the highest T) on
4700 nodes uniform in log T, 1e-3 dex apart, 1 K to 5.0e4 K. Log spacing
because the structure is in log T (rotational onset at tens of K,
vibration at thousands); 1e-3 dex resolves the 20-30 K region better than
the table's own samples there (0.66 K), and finer grids (5e-4, 2.5e-4 dex,
or 1 K and 0.5 K linear grids, all tried) do not lower the error at the
table's points, which is set by the table's own inconsistencies above.
At run time the interpolant is the cubic Hermite of phi = e_H2/T in ln T,
with slopes from e_H2' (phi' = e_H2' - phi): C1, exact in both linear
limits (phi = 1.5 below 1 K, 3 above 5e4 K), and gamma comes from the same
interpolant, so T and gamma are consistent.

**The solve.** G(u) = ln(x e_H2 + 1.5 (1+a-2x) T) - ln(ekn (1+a-x)) in
u = ln T, which is nearly linear (slope 1 + x phi'/(x phi + c)). Bracket
from 1.5 <= phi <= 3.49 (the table's range, widened by 1e-6), start from
the bracket's top, six Newton steps clipped to the bracket (updated by the
sign of G), then one more step outside `stop_gradient`. That last step
carries the implicit-function derivative dT/d(ekn, x, a) = -G_theta / G_u:
reverse and forward mode work and the integrator's custom_vjp sees it.
Limits: x = 0 and ekn < 1 (T < 0.67 K) come out atomic (T = 2/3 ekn,
gamma = 5/3) from the same formula; ekn <= 0 returns 2/3 ekn (flagged
`cold` by the RHS, as before). Above the table T keeps e_H2 = 3 T: the table
path switches to atomic gas at ekn > 5e4 (T jumps by up to 1.9x at
x_H2 = 0.5 there), the Newton path stays continuous.

**Measured** (`tools/chem_eos/validate_newton.py`, MI300A):

| | T max (where) | T median | gamma max | gamma median |
|---|---|---|---|---|
| a = 0.1 vs the table on its 250000 points | 8.0e-5 (x=0.4, T=19989 K) | 8.1e-9 | 4.3e-2 (T=19999 K) | 3.3e-6 |
| a = 0.079 vs the exact rescale (energy-table-abhe0079.txt) | 8.0e-5 (same place) | 8.3e-9 | 4.3e-2 (same) | 3.4e-6 |
| the table (a = 0.1) used at a = 0.079, for comparison | 2.4e-2 (x=0.5, ekn=5e4) | 8.1e-3 | 1.3e-1 | 2.7e-3 |

The maxima sit where the table disagrees with itself (the 2e4 K gamma
ramp; the 20-30 K zigzag gives T 6e-5, gamma 3.5e-3); the table lookup's own
interpolation error at its nodes is 2.6e-8 (x blending). Convergence:
|ekn(T)/ekn - 1| <= 3.9e-15 over 1e7 random (x, ekn, a) with ekn from 1e-3
to 1e8. Gradients against central differences: 1e-10 to 1e-9 relative.
Cost, 1e7 cells, jitted: lookup 0.050 ns/cell, Newton 0.311 ns/cell (6.3x;
both negligible next to one RHS evaluation).

Would we make it the default? For network 17 science runs, yes: the table
is wrong by 0.8 % (median) to 2.4 % in T at He/H = 0.079, and the Newton
path is continuous at the table edges and differentiable everywhere. The
default stays `eos="table"` (the Fortran comparisons need it);
`fixes17="fixed"` selects `eos="newton"` unless told otherwise.

One inconsistency is shared by both paths and left alone: ekn divides by
n_tot including electrons, while the relation above counts no electrons
(1 + a - x particles per H). Exact would be e/(k n_H) = x e_H2 +
1.5 (1 + a - 2x + x_e) T; it matters only for ionised gas with H2.

## Data files (network 17 and the EOS)

Read through `tables.data_path(name)` from `$LUNA_DATA/chemistry/`,
none in the repository:
`eh2_table.npz` (the Newton EOS), `chemistry_net17_golden.npz` and
`chemistry_net17_fixed_golden.npz` (the per-function references),
`net17_trajectories.npz` and `net17_fixed_trajectories.npz` (all 72288
trajectory states, optional), `chemistry_net17_trajectories.npz` (the
packed integrator reference) and `energy_table_abhe0079_every25.npz`
(every 25th row of the exact He/H = 0.079 rescale), next to network 5's
`net5_raw.npz` and `energy_table.npz`. The Fortran harness that makes the references
(drivers, Makefile, config header, run scripts) is in
`$LUNA_DATA/chemistry/harness/net17/`, in neither repository.

# Phase 3: the stiff integrator (`luna/ode/rodas3.py`)

DVODE (BDF, variable order, one cell at a time on the CPU) is replaced by
one batched solver that integrates every cell over the same interval dt, each
on its own adaptive substeps: `integrate(y0, params, dt) -> Result` with the
state at dt, the updated T_dust, a step size to start the next call with,
substep counts and a `failed` flag. It jits, runs inside `shard_map` with no
communication, and is differentiable in y0, every Params field and dt.

**Method: Rodas3** (Sandu et al. 1997, Atmos. Env. 31, 3459; the form of
KPP's `Rosenbrock`), a 4-stage Rosenbrock method of order 3 with an embedded
order-2 solution, stiffly accurate and L-stable (both solutions):

    (I/(h g) - J) k_i = f(y + sum_j A_ij k_j) + sum_j C_ij k_j / h,  g = 1/2
    y1 = y + 2 k_1 + k_3 + k_4,   error estimate k_4

with the exact Jacobian J = jacfwd(f) at the step start. Why Rosenbrock and
not an SDIRK: an SDIRK needs a simplified-Newton loop per stage, and in a
batch every cell waits for the slowest Newton. A first version used Kvaerno's
ESDIRK3(2); one attempt cost 38 ns/cell (4M-cell batch), of which ~24 ns were
Newton iterations of a few cells. Rodas3 does fixed work per attempt -- one
Jacobian, one LU, three RHS calls (stage 2 reuses f(y)), four solves --
11.8 ns/cell, and needs the exact Jacobian we have anyway. DVODE's BDF is not
an option on the GPU: its history (Nordsieck array) and order changes are
per-cell state that do not batch, and every restart (each hydro step) starts
it at order 1.

**Linear algebra**: the n x n systems (4 x 4 here, 8 x 8 for network 17) are
solved by an unrolled LU with partial pivoting written in jnp, on the matrix
scaled by the error weights. 2.1 ns/cell for factor + solve;
`jnp.linalg.solve` on the same batch costs 1400 ns/cell on the MI300A.

**Error control** as DVODE's: the weighted RMS of the error estimate with
weights rtol max(|y_n|, |y_n+1|) + atol, rtol = 1e-4 and atol = (1e-7,
1e-12, 1e-14, 0) for (x_H2, x_H+, x_CO, e) (`chemcool_consts.h`); step
factor 0.9 err^(-1/3) in [0.2, 5]. The first step is evolve_abundances'
dt_est (0.1 y/|ydot| over the species above their pool floor, 0.3 for e).
After each step, check_abundances: NaN, x < -atol or an overshoot of the
elemental bound by more than rtol reject the step (h / 2); smaller ones are
clipped. A step also fails if any stage state is invalid in the RHS.

**Batching and the step cap**: one `lax.while_loop` per cell under `vmap`;
cells that are done stop changing. Because a batched loop runs as long as its
slowest cell, the integration is done in tiers: every cell gets 2 attempts,
then the unfinished cells are gathered (`jnp.nonzero` with a static size) into
chunks of n/2 for 6 more attempts, n/8 for 16, n/32 for 48, n/128 for the
rest, up to `max_steps` = 2000. Measured at 256^3, dt = 1e3 yr: one pass
over all cells 1959 ns/cell, these tiers 326 ns/cell (other tier sets
346-517). A cell that hits the cap, or whose step underflows, returns
`failed` and the state where it stopped; nothing is silently wrong.

**The hidden dust temperature**: calc_dust_temp starts its Newton iteration
from the previous T_dust and stops at the first step below 1 % (flagged
item 5), so the Fortran RHS depends on its call history. The integrator
passes `dust_rtol = 1e-12` (a new Params field, default 1e-2 = the Fortran):
the RHS then solves T_dust to convergence from the carried value (two or
three iterations), is a smooth function of y, and takes one more Newton step
at the end so that T_dust carries its implicit-function derivative into
jacfwd (without it the Jacobian misses d T_d / d y: 2e-3 against finite
differences instead of 1.6e-9; a test). The dust loop is a `while_loop` that
ends when every cell has stopped (same iterates as before, forward-mode
differentiable).

**Differentiability**: a `custom_vjp` on the whole call. Under
differentiation the forward pass carries, per cell, the sensitivity matrix
S = d y / d(y0, theta) (theta = the Params fields asked for, all but the
hidden ones by default) through every accepted step,
S <- (d Phi/d y) S + [0 | d Phi/d theta], with d Phi the forward-mode
derivative of the Rosenbrock step map (its linear solves included) at the
accepted step size. The backward pass is one matrix product per cell
(y0_bar, theta_bar) = y1_bar S, plus f(y1) . y1_bar for dt. Nothing of the
step control or the loop is on the tape: gradient memory is one n x (n + m)
matrix per cell. Step sizes are treated as constants (the standard
discrete-sensitivity choice).

### Accuracy against DVODE

The 168 one-zone trajectories (250 log-spaced outputs from 1 yr to 1 Gyr),
integrated from the same initial states with a fresh start at each output,
as `evolve_abundances` does. Maximum relative error over all outputs where
the reference is above 1e-3 of its elemental pool (x_H2: 0.5, x_H+: 1, x_CO:
min(C, O)); e and T everywhere; and at the end state of the n >= 1e3 cases
(equilibrium):

| reference | x_H2 | x_H+ | x_CO | T | equilibrium T |
|---|---|---|---|---|---|
| DVODE rtol 1e-8, dust converged (the test's reference) | 9.8e-5 | 6.2e-5 | 8.1e-5 | 4.3e-4 | 5.4e-8 |
| DVODE rtol 1e-4, dust converged | 3.8e-3 | 1.2e-3 | 1.3e-3 | 9.4e-3 | 5.4e-8 |
| DVODE rtol 1e-4, dust 1 % (the original reference) | 3.7e-3 | 1.2e-3 | 1.4e-3 | 1.4e-2 | 1.0e-2 |

The targets (1e-3 at every output, 1e-4 in equilibrium) hold against the
first row and miss against the original reference. That is the reference,
not the integrator, as two diagnostic Fortran builds show
(`$LUNA_DATA/chemistry/harness/net5/run_variants.sbatch`: the same driver with RTOL 1e-8,
ATOL / 1e4 and/or the dust Newton stopped at 1e-12; no other change):

- DVODE at its production tolerance against DVODE at rtol 1e-8 (same RHS)
  differs by 3.7e-3 (x_H2), 1.2e-3 (x_H+), 1.3e-3 (x_CO) and 9.3e-3 (T):
  DVODE's own global error at RTOL = 1e-4 is above the 1e-3 target. Rodas3
  at the same rtol is 10-40x closer to the converged solution (first row).
- The 1 % dust Newton moves the equilibria: the original reference ends,
  for n = 1e5, at a T_dust 1 % away from the converged dust balance, and the
  gas temperature follows (1.0e-2 in T at the end, 1.4e-2 along the way).
  Its end states are steady states of the 1 %-Newton RHS (|de/dt| / sum of
  the |terms| <= 2.2e-5 with the T_dust it carried) and not of the
  converged one (up to 0.12). No integrator of the converged RHS can agree
  with them to 1e-4; the "carried" alternative (the Fortran's 1 % rule with
  T_dust carried between steps) was tried and reproduces the reference no
  better (T 9.4e-3, measured with the first, ESDIRK version), because the
  iterate depends on the call history.
- Rodas3 at rtol 1e-8 (atol / 1e4) against DVODE at rtol 1e-8, both with
  the converged dust: 2.2e-6 in x_H2, 7.3e-7 in x_H+, 4.8e-7 in x_CO,
  3.5e-6 in T, 5e-8 in equilibrium (the float32 storage of the packed
  reference). Two independent integrators agree.

`tests/test_chemistry_integrator.py` holds the default integrator to 1e-3 /
1e-4 against the first row (packed in `$LUNA_DATA/chemistry/chemistry_net5_trajectories.npz`, 0.6 MB); 55412 accepted
substeps for all 168 x 250 intervals, none failed, 2-9 per interval.

**Teeth** (each must fail the same comparison): explicit Euler in place of
the Rosenbrock step (488 of 1500 intervals fail at the step cap, errors up
to 2), the Jacobian's sign flipped (566 failures), the error estimate
disabled (no failures, but 3.0e-2 off).

### Gradients

Against central differences (1e-3 of each input, integrator at rtol 1e-8)
for F = sum of log T + log x_H2 + 1e3 x_H+ / (1e-4 + x_H+) after dt, from
mid-trajectory states of n = 1e2 (no field) and 1e5 (A_V = 1), w.r.t. y0
and the cosmic-ray rate zeta: the elasticities x dF/dx agree to 4.1e-6
(dt = 1e3 yr) and 2.2e-6 (1e5 yr) of the largest one; with sensitivities
that do not propagate through the steps the same check is off by more than
1e-2. On an 8 x 8 stiff linear chain (rates 1e-3 .. 1e6) the gradients
w.r.t. y0, a rate factor and dt match exp(M dt) to 1e-5.

### Cost on the GPU

One MI300A (apudev, jax 0.10.2 ROCm, FP64), states drawn uniformly from the
42000 outputs of the reference trajectories (each with its case's
parameters, T_dust guess 10 K), integrated over a hydro-like dt
(`tools/chem_integrator/bench_net5.py`; chunks of 2^22 cells via lax.map):

| cells | dt = 1e3 yr | dt = 1e5 yr |
|---|---|---|
| 64^3 | 657 ms (2.5 us/cell) | 2753 ms (10.5 us/cell) |
| 256^3 | 5.47 s (326 ns/cell) | 17.8 s (1061 ns/cell) |
| 512^3 | 43.9 s (327 ns/cell) | 151.7 s (1130 ns/cell) |

Accepted substeps per cell: dt = 1e3 yr mean 8.7, median 1, p90 23, p99 91,
max 155 (0.04 rejected); dt = 1e5 yr mean 21.3, median 5, p90 69, p99 134,
max 207 (0.46 rejected). The expensive cells are the dense (n = 1e5) gas
cooling from 1e4 K in the first outputs: e-folding time ~200 yr, H+ ~4 yr,
across thousands of table nodes. The sample over-weights such transients
(the outputs are log-spaced from 1 yr); equilibrium gas needs one step.

Per attempt and cell (4M-cell batch): RHS 1.4 ns, jacfwd 6.5 ns, LU 2.1 ns,
one Rodas3 attempt 11.8 ns. The 512^3 numbers are 3.3x the ideal
mean-substeps x 11.8 ns, the rest being the tiers' padding. 64^3 is
latency-bound (a while-loop iteration is a host round trip with command
buffers off).

**Against the hydro** (52 ms per step at 512^3 with Pallas): the chemistry
stage, as measured on this sample, costs ~850x a hydro step at dt = 1e3 yr.
It does not fit next to the hydro at every step at this cell count on one
GPU; see the phase-4 note below.

**Sharding**: inside `shard_map` over 2 GPUs the result equals one GPU
bitwise (2^18 cells, also with the tiers). That needed every per-cell sum to
be written as a loop (the RMS norm here, the TreeCol pixel mean in photo.py):
an XLA reduction is compiled differently for different batch shapes, and the
tiers make shapes that differ between 1 and 2 devices. On the CPU backend the
two programs still differ in the last bits (5e-12).

### Network 17

The integrator was network-agnostic already (the RHS, tolerances and pools
come in a `Network`); network 17 needed only its own: `networks.NET17`
(`NET17_FIXED` for the fixed RHS) with chemcool_consts.h's tolerances
(rtol 1e-4; atol 1e-7, 1e-12, 1e-14, 1e-14, 1e-14, 1e-14, 1e-12, 0 for
x_H2, x_H+, x_CO, x_He+, x_He++, x_D+, x_HD, e), check_abundances' bounds
(0.5, 1, min(C, O), abhe, abhe, D/H, D/H) and evolve_abundances' step
floors (CO, He+, He++, D+, HD against 1e-6 of their pools), the dust
temperature converged as for network 5 (`dust_rtol` is now a network-17
Params field too). Params fields left `None` (the photo state) are skipped
by `broadcast_params` and the sensitivities. The photo state is the
caller's: as GIZMO, compute it at the start of each hydro step
(`rates_net17.photo_state`) and pass it in Params.

`tools/chem_integrator/validate_net17.py` integrates the 288 network-17
one-zone runs (250 intervals each, photo state and chemcool.cc's input
clamps at each interval start, as the Fortran driver). Maximum relative
error at every 5th output (species above 1e-3 of their pool), "off
runaways" excluding the outputs where T changes by more than 25 % within
one interval, and at the n >= 1e3 end states:

| reference | x_H2 | x_HD | x_H+ | T | T off runaways | equilibrium T |
|---|---|---|---|---|---|---|
| DVODE rtol 1e-8, dust converged (the test's reference) | 1.6e-3 | 3.8e-3 | 3.6e-4 | 2.7e-2 | 5.3e-4 | 1.7e-6 |
| same, Rodas3 at rtol 1e-6 (atol / 100) | 1.3e-4 | 3.0e-4 | 4.2e-6 | 2.1e-3 | | 1.3e-6 |
| DVODE rtol 1e-4, dust 1 % (net17_ref, the original) | 1.6e-2 | 3.8e-2 | 1.1e-3 | 1.8e-1 | | 1.3e-2 |
| same, Rodas3 with the Fortran's 1 % dust stop | 1.6e-2 | 3.8e-2 | 1.1e-3 | 1.8e-1 | | 3.7e-3 |
| DVODE rtol 1e-4 vs DVODE rtol 1e-8 (no Rodas3) | 1.9e-2 | 2.9e-2 | 1.1e-3 | 1.2e-1 | | 1.3e-2 |

As for network 5, the production DVODE reference carries its own error
(last row) and the 1 % dust Newton moves the cold, dark equilibria (n =
1e5, T ~ 4 K, T_dust ~ 3.3 K: 1.3e-2). The largest differences are thermal
runaways: at n = 1e5, T0 = 1e4 K, low Z, T falls from 4300 K to 1800 K
within one output interval (ln T changes by 0.86), and a phase lag of a few
1e-3 in time is 2.7e-2 in T; at rtol 1e-6 it is 2.1e-3, so it is the
integrator's tolerance, not a bias. Medians are 3e-8 (T) to 4e-6 (x_HD).
`tests/test_chemistry_integrator_net17.py` holds the default integrator to
2e-3 off runaways (the measured 1.07e-3 is x_HD, a trace species whose
error accumulates over 1 Gyr) and 1e-4 in equilibrium, against the first
row (`$LUNA_DATA/chemistry/chemistry_net17_trajectories.npz`, 0.35 MB; the
Fortran variant is `harness/net17/run_variants.sbatch`). 0 failures;
1.5 substeps per interval on average, at most 63; 17.7 s for all 288 x 250
intervals on one MI300A. Teeth (`tests/test_chemistry_integrator_net17_teeth.py`):
explicit Euler (1090 of 1500 intervals fail at the step cap) and the error
estimate disabled (3.5e-2 off) both fail.
Fixed mode against net17_fixed (table EOS): the same picture (T 1.7e-1 at
the runaway, equilibrium 1.2e-2 from the dust; medians 3e-6).

Gradients through the fixed network with the Newton EOS (dt = 1e4 yr, one
shielded cell at n = 1e4, Z = 1e-4, d/d(x_H2, x_H+, x_HD, e, zeta) of sum(ln T + ln x_H2))
agree with central differences to 7.9e-7; with the EOS's implicit step
cut (no dT/de) the same check is off by 3.7 (the e column vanishes). Both in
`tests/test_chemistry_integrator_net17_grad.py`.

### Phase 4 (coupling into LUNA as a stage)

- A stage in the one time loop, between the hydro half-steps: the cell's
  y = (x_H2, x_H+, x_CO, e_int) from the conserved variables (abundances as
  passive scalars), `integrate` over the hydro dt, write back e_int and the
  scalars; carry T_dust and `Result.h` per cell (warm starts cut the first
  steps); columns from TreeCol when it exists, zero columns until then.
- Cost is the open problem: at 512^3 the measured chemistry is ~850x a
  hydro step (dt = 1e3 yr) on a transient-heavy sample. Before tuning, measure the substep
  distribution on a real simulation's cells (equilibrium gas is one step);
  then: a fused Pallas/Triton kernel for RHS + Jacobian (one attempt costs
  ~8 RHS evaluations, the jacfwd more than half of it), the hot-gas bypass of evolve_abundances (T > 5e5 K),
  chemistry every k hydro steps where the cooling time allows, and warm
  T_dust.
- The integrator is network-agnostic (`Network` spec: rhs, rtol, atol,
  bounds, hidden fields); network 17 needs its own spec of the same shape.

### Coupled cost: network 5 in a driven, self-gravitating ISM box (prototype)

A **measurement prototype**, not the phase-4 design: `luna/sources/chemistry.py`
(`chemistry = "net5"`) runs network 5 as the outermost stage of the step (nothing
before the hydro, `integrate` over the full dt after it: first-order splitting),
replacing the cooling stage, so only the chemistry changes the internal energy.

- **Advection.** LUNA's hydro carried no extra fields (the radiation and the
  ionisation state live in `state.sources` and are not advected). The species
  are now passive scalars S = rho x (x per H nucleus; x_HI and C+ derived)
  in `state.sources["chemistry"]["S"]`, advected inside the Pallas kernel: X
  is reconstructed and predicted like a primitive variable (minmod, v.grad X),
  and its face flux is F_rho X_upwind of the contact, which is HLLC's exact
  passive-scalar flux. Hydro with scalars only on that kernel
  (`step(U, S, dt)`); with none it traces the same program, and the default
  path is bitwise unchanged (128^3 driven + self-gravitating box, 6 steps,
  2 GPUs, before vs after: identical bytes). Cost of the 3 scalars at 512^3,
  1 GPU: 55.1 -> 87.6 ms per hydro step.
- **Shielding**: a local column N_H = n_H min(L_J, L_box) (L_J from the hydro
  gamma = 5/3), N_H2 = x_H2 N_H, N_CO = x_CO N_H, one pixel; A_V from N_H.
- **Warm starts**: T_dust and the integrator's step guess per cell are carried
  (not advected).
- **Floor**: internal energy below 1.5 (1 + x_He) n_H k (5 K), on the way in
  or out, is set to it and counted. A failed cell can stop at e = 0 and
  E - KE then rounds negative: the first 512^3 run went NaN one step after
  its first failed cell (step 1326). With the floor, a restart from step
  1273 ran to 1367 with no failed cell; it floored 1 cell at step 1326 and
  1-8 of the 1.3e8 cells per step from step 1356 on (the restart does not
  reproduce the trajectory bit for bit: the gravity solve starts cold).

Setup: 100 pc periodic box, 512^3 (dx = 0.195 pc), 1 MI300A, FP64, minmod,
HLLC, the Pallas kernel, CFL 0.4; uniform n_H = 1 cm^-3 at the network's own
equilibrium for that density and column (T = 7984 K, x_H2 = 1.7e-8,
x_H+ = 6.4e-3, x_CO = 4e-21, T_dust = 14.9 K; one-zone, 10 Myr chunks until
converged); OU forcing (k = 1..3, solenoidal weight 0.5, parabolic), rms
target 18.6 km/s = Mach 2 of the warm gas (T_ac = 2.7 Myr); periodic
self-gravity (G in pc, Msun, km/s); G0 = 1.7 Habing, zeta = 3e-17 s^-1,
solar C, O, dust-to-gas 1, no Si; the integrator's default dust solve
(converged, `dust_rtol` 1e-12, not the Fortran's 1 %); integrator batches
of 2^20 cells (2^22 runs out of memory at 512^3). Driver:
`tools/chem_coupled/run_turb_chem.py`; tables and figures `summarize.py`.

Per-step cost, medians per 0.5 Myr (the integrator with empty-tier
skipping, below):

| t [Myr] | steps | dt [yr] | hydro ms | gravity ms | forcing ms | chemistry ms (ns/cell) | chemistry % | cells > 1 attempt | max substeps |
|---|---|---|---|---|---|---|---|---|---|
| 0.0-0.5 | 87 | 5380 | 87.4 | 50.6 | 221 | 1740 (13.0) | 82.8 | 0 | 1 |
| 1.0-1.5 | 174 | 2850 | 87.8 | 50.5 | 221 | 1750 (13.0) | 82.9 | 0 | 1 |
| 2.0-2.5 | 206 | 2460 | 87.9 | 50.7 | 221 | 2250 (16.8) | 86.1 | 5.7e-5 | 9 |
| 2.5-3.0 | 232 | 2150 | 87.9 | 50.8 | 221 | 5400 (40.3) | 93.7 | 6.0e-4 | 27 |
| 3.0-3.5 | 246 | 2030 | 88.0 | 50.7 | 221 | 8910 (66.4) | 96.1 | 2.0e-3 | 38 |
| 3.5-3.7 | 48 | 1980 | 87.9 | 50.6 | 221 | 10900 (81.1) | 96.8 | 3.3e-3 | 38 |

Accepted substeps per cell: median, p90 and p99 are 1 at every step; the
mean is 1.00-1.01; the maximum grows from 1 to 38 as gas cools. One cell
failed (step 1326). Mass in T < 300 K / 300-5000 K / > 5000 K: 0.6 / 10.6 /
88.8 % at 2.5 Myr, 6.2 / 32.2 / 61.6 % at 3.5 Myr (n_H up to 1e3,
T down to 20 K). The cost is not the cells' work: 1.0056 attempts per cell
x 11.8 ns would be 1.6 s; the rest is the straggler tiers, which process a
fixed share of the batch however few cells are left.

Measured offline on the saved state at step 1273 (3.49 Myr, dt = 1997 yr;
`tools/chem_coupled/speedups.py`; one call, the step that state would take):

| variant | ms |
|---|---|
| phase-3 integrator (every tier scans all its chunks) | 16716 |
| + skip an empty tier, loop only over chunks with work (the run's) | 8733 |
| tiers ((2,1),(6,64),(16,256),(48,1024),(None,4096)) | 5377 |
| tiers ((1,1),(7,64),(16,256),(48,1024),(None,4096)) | 3882 |
| one pass, no tiers | 15318 |
| skip T > 5e5 K | no such cells here |
| skip cells with t_chem > 100 dt (97.2 % of cells; one explicit RHS step for them, error in e max 7.5e-4, p99 7.6e-5) | 186 (RHS) + 346 (the rest, compacted) |
| skip cells with t_chem > 10 dt (99.4 %; max 2.6e-2, p99 1.6e-4) | 186 + 106 |
| chemistry every 2 steps over 2 dt | 5079 per step |
| chemistry every 4 steps over 4 dt | 2890 per step |

At step 896 (2.71 Myr): 12602 / 4236 / 3129 / 2516 / 5543 ms for the first
five rows. t_chem is min over components of y / |dy/dt| (species above
their control floor), the "skip" rows' errors are against the full
integration. Masking skipped cells with dt = 0 instead of compacting them
saves nothing (the batch runs in lockstep). The every-k rows change the
splitting error, which is not measured here.

### The hybrid solver (`integrate_fixed`) and its fused kernels

The stage's solver (`ChemConsts.solver = "hybrid"`; `"adaptive"` keeps `integrate`):
every cell takes k1 = 1 Rodas3 step over the whole dt with no rejection and no while
loop; the cells whose embedded error norm exceeds `flag_err` = 0.1, or whose step was
invalid, are compacted on the device and take k2 = 4 substeps of dt/4 instead. Network:
`sources.chemistry.fixed_net`, the SymPy-generated `net5_core` (one pixel, no LVG, exact
Jacobian, the dust Newton as 4 masked steps at the 1 % stop). Pieces:

- `integrate.fixed_substeps` (lists of per-cell values: the same code runs vmapped in
  XLA and inside a kernel), `rosenbrock_jac / _stages / _finish`, `lu_factor_lists`
  (lu_factor's pivots and arithmetic). Options `k1`, `k2`, `flag_err`, `chunk` (first
  pass batch), `chunk2` (second-pass batch; the loop runs only over batches holding
  flagged cells).
- Compaction by a cumulative sum and a scatter that drops the unflagged cells. Not
  `jnp.nonzero`: its bincount scatter-adds every unflagged cell onto a few slots, and
  those atomics cost 474 ms at step 896 (125k flagged) and 47 ms at step 1273 (687k):
  the "second pass" cost of the first hybrid prototype. The cumulative sum takes 1.2 ms.
- Gradients: `integrate_fixed` has a custom VJP that recomputes each cell's VJP through
  its own k1 or k2 substeps (the second pass compacted again); it equals jax.grad of
  the same hybrid written without compaction (tests/test_chemistry_fixed.py).
- `luna/chemistry/fused.py`: the same solve in Pallas (Triton) kernels, FP64, one cell
  per lane. Per substep three kernels (f and J; LU and stages 1-3; stage 4, error and
  bounds). The T_dust returned (the next call's warm start) is the one solved at the start
  of the last substep, not re-solved at the end state. Splitting the f + J kernel (by
  rows or columns of J) was measured slower: each part repeats the lookups and keeps the
  register pressure (the spills come from the shared intermediates). One kernel per substep does
  not compile in reasonable time: Triton's coalescing pass walks the forward slice of
  every table gather, so its cost grows as (gathers x downstream ops); f + J alone
  takes 27 s, f + J + LU 65 s, the whole substep over 7 minutes. The kernels read the
  tables from one packed buffer (`tables.device_table` views). Their gradient is the
  plain one (custom VJP: `integrate.fixed_vjp`). Kernel weak-typed constants: a
  transcendental of a Python float lowers to its FP32 version in Triton, so the code
  keeps those as Python constants (`lookups.LN10`).

Measurements (512^3, one MI300A) are in the speed-up notes outside the repository.

Also to port later: the M1 hooks (ph38/ph42, already zero here), network 1
(WSS_CIE_COOL is ported, next section).

Credits: S. C. O. Glover, chemcool / SGChem (networks 5 and 17);
U. Steinwandel, the D+/HD extension of network 17; Nelson & Langer 1997 for the CO
model; the rate and cooling references are cited at each term in the source.

# WSS_CIE_COOL: hot-gas cooling (`cie_cool`)

chemcool's `-DWSS_CIE_COOL` adds the cooling of gas above ~1e4 K. The port
has it as an option, off by default (`rates_net5.rhs(y, p, cie_cool=True)`,
`cooling.cool_func(..., cie_cool=True, metallicity=Z)`, `networks.NET5_CIE`
for the integrator); with it off, every result is bitwise the one of the
code before (measured: RHS, cool_func, jacfwd, 59 trajectory outputs of all
168 cases and the gradients, on the CPU backend). The flag does four things:

1. **Term 3, metal-line cooling** (`wss_z_collis.F`): per element (C, N, O,
   Ne, Mg, Si, S, Ca, Fe) the collisional-equilibrium rates of Wiersma,
   Schaye & Smith (2009) at solar abundance, tabulated at 352 temperatures
   from 1e2 to 9.6e8 K, interpolated linearly in log rate, times n_H^2 and
   times each element's abundance over solar.
2. **He in CIE** (cool_func.F:391): He I, II, III fractions from a 51-point
   table in log T (1e4..1e7 K); all He I at or below 1e4 K, all He III at
   or above 1e7 K. They replace any He abundances the network carries.
3. **Terms 26-28 switched on** (coolinmo.F:1577): He I and He II excitation
   and the bremsstrahlung of H+, He+ and He++ (cltab rows 59-63, otherwise
   zeroed because the unused SD93 table "includes them").
4. **He becomes a collider at every temperature**: with item 2, neutral He
   (0.1 per H) enters the H2 rovibrational (cl64) and CO rotational
   (0.5 n_He) cooling, also in cold gas. Measured at n = 100, x_H2 = 0.4:
   H2 cooling x1.17 at 100 K, x1.37 at 1000 K, CO rotational x1.07; at
   n = 1e4, x_H2 = 0.49: H2 x1.01 and CO x1.05 at 300 K.

**Metallicity.** GIZMO fills Zmass with each cell's own element mass
fractions (chemcool.cc:252, capped at 10 Z_sun). LUNA tracks no elements:
it uses the WSS solar pattern scaled by one number, `Params.metallicity`
(Z / Z_sun, default 1, used only with the flag), so every element's ratio
is that number. At metallicity 0 term 3 is exactly 0 (He and free-free
remain). The caller sets it consistently with `dgr` and `abundc/o/si`, which
GIZMO scales with the same Z.

**Double counting.** None between the parts: term 3 holds metals only; H
excitation (term 25), He and free-free (26-28) are separate rows; the SD93
table cl6 is computed but rates(3) is overwritten (0, or the WSS rate), so
it never counts. The overlap is with the fine-structure terms 13, 16, 17
(OI, CII, SiII with the network's assumption of neutral O and all C as C+),
which stay on at all T, while WSS contains C, O and Si lines too: per
n_H^2 at x_H+ = 1, Z = 1, OI + CII + SiII are 2 % of the WSS total at
2e4 K, 0.07-0.3 % from 5e4 to 1e6 K and 4 % at 1e7 K (OI with neutral O at
1e7 K); below 1.2e4 K the fine-structure lines dominate (WSS is suppressed
there, item below).

**Quirks, ported as they are** (each in `tables.py` / `cooling.py`):

- abs(): 343 negative entries (net heating; N to 3600 K, Ne to 8100 K, Mg
  to 11200 K, Ca to 2600 K, Fe to 8500 K) count as cooling.
- The T < 1e4 K suppression is a factor exp(-(1e4/T)^2) (floor 1e-20):
  soft, 0.37 at 1e4 K, 0.78 at 2e4 K, 0.17 at 7500 K. It also suppresses
  above 1e4 K, and does not remove the metals below it.
- The lookup assumes a uniform grid (log T = 2 + i 6.98185/351, the range a
  single-precision literal); the file's temperatures deviate by up to 2e-5
  dex.
- Above the second-to-last node (9.16e8 K) the rate is clamped to that
  node; the last interval is never used. Below 100 K: the first node.
- The rates are CIE rates per n_H^2: they do not see the network's own
  x_e or x_H+, while terms 25-28 do.

**evolve_abundances' hot-gas bypass** (not ported, not part of the flag):
above 5e5 K the Fortran does not integrate the network; it sets x_H+ = 1,
x_H2 = x_CO = 0 and steps the energy alone by explicit Euler with
dt = 0.1 e/|de/dt|. Against DVODE through the full network at rtol 1e-8,
this moves the time to cool below 1e5 K by up to 7 % (median 0.4 %) on the
grid below. The integrator here integrates the network everywhere; its
reference is built with the bypass off (`-DCHEMCOOL_TNOCHEM=1d99`, a
compile-time switch, no source change).

**Reference and agreement.** `$LUNA_DATA/chemistry/harness/cie/` builds the unmodified
sources with `-DWSS_CIE_COOL` (`cie_ref`): per-function
dumps (the WSS and He tables, each element alone and the sum at 2124
temperatures 10 K..1e10 K with z = 0, 0.1, 1, 3; cool_func at 3000 states
from 1e2 to 1e9 K; rate_eq at 1500 states to 1e9 K) and a one-zone grid of
75 runs: n_H = 0.1..1e3, T0 = 1e4..1e8 K (x_H+ = 0.99 above 1e4 K),
z = 0, 0.1, 1, G0 = 1.7 thin, 1 yr to 1 Gyr. Max relative error, FP64 CPU:

| function | error |
|---|---|
| WSS tables, He tables, cltab 59-64 | 5.7e-14 |
| WSS per element and total | 2.9e-13 |
| 28 cool_func terms + T_dust | 8.1e-11 |
| RHS: ydot / gross, 34 terms, T_dust (also jitted) | 1.1e-11 |
| RHS at all 18825 states of the grid (with bypass / without) | ydot 4.4e-13 / 5.4e-13, cooling 3.1e-13 / 3.0e-13, chemical 4.4e-13 / 6.7e-13 |
| jacfwd vs central differences (5 hot/warm states) | <= 1.6e-8 |

The largest error is in Lyman alpha (term 25, not part of the flag) near
3e8 K: cool_func extrapolates its table linearly above 1e8 K, where cl52
falls with T, so the term turns negative (heating) above 3.07e8 K, a
cancellation the port follows to 8e-11. Also in the Fortran without the flag.

The integrator with `NET5_CIE` against DVODE (rtol 1e-8, dust converged,
bypass off): at rtol 1e-8 both agree to 3.8e-4 at every output (x_H+, e,
T) and to 8e-6 in the time to cool below 1e5 K. At the default rtol 1e-4,
Rodas3 cools 5.0e-3 early or late at worst (median 1.9e-5); DVODE at its
production rtol 1e-4 is off by 6.4e-3 (median 3.9e-4). At fixed output
times the runaway cooling amplifies such timing errors (T drops 1e6 ->
1e4 K within one output interval): 7.6e-2 for Rodas3 at rtol 1e-4, 0.21
for DVODE at rtol 1e-4. The test therefore holds the default tolerance to
DVODE's cooling-time accuracy and the end states (9.9e-9), and the tight
tolerance to 1e-3 at every output (`tests/test_chemistry_cie.py`).

Teeth (each fails its comparison): abs() dropped, the suppression factor
dropped, C and O or Mg and Si columns swapped, the last interval
interpolated, the log-T range read in double, metallicity ignored, the He
fractions dropped, neutral He dropped as a collider, cl59 or cl62 1 % off,
the Gaunt-factor literals read in double, term 3 and 26-28 left out of the
RHS; integrating the grid without the flag (most runs never cool).

**What it changes.** A 1e6 K, n_H = 1, Z = Z_sun parcel (G0 = 1.7, thin,
dust-to-gas 1, x_H+ = 0.99 at the start), measured with the port and the
Fortran: with the flag the cooling time e / Lambda of the ionised gas is
8.9e4 yr (Lambda = 1.55e-22 erg s^-1 cm^-3: WSS 1.53e-22, free-free
2.8e-24); it falls below 1e5 K after 6.15e4 yr (Fortran with its bypass:
6.18e4 yr) and settles at 7537 K. Without the flag nothing cools it: once
ionised, Lambda = -9.0e-25 (photoelectric heating 1.6e-24 beats OI and CII
fine structure), a heating time of 1.5e7 yr, and it reaches 8.1e8 K at
1 Gyr (Fortran: 8.04e8 K).
