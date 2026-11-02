# Test catalogue

The literature-driven verification tests by tier, with the leading reference for each. Setups, bounds
and status live in `docs/TEST_PLAN.md`; this file is where an ID such as A14 is defined.

## Tiers

The tier is what LUNA needs before the test can run (the author's prerequisite order).

| Tier | Needs |
|---|---|
| A | Nothing: runs on today's boundaries and physics |
| B | Exact, time-dependent, per-face ghost cells |
| C | A multi-material (two-gamma) EOS |
| D | Thermal conduction / physical viscosity |
| E | MHD |
| F | Zoom initial conditions |
| G | GPU resolutions (some also per-axis boundaries or deep AMR) |

Priority within a tier: P1 cheap with a published number to bound against; P2 a published number but
costly or needing a small feature; P3 a validation spread or a physics reference.

arXiv IDs, report numbers and links come from the literature audit; journal volumes and pages added
here without them are from memory and are to be checked before they are cited.

## A: runs today

| ID | Test | P | Leading reference | Test file |
|---|---|---|---|---|
| A1 | Linear-wave eigenmodes: sound L/R, entropy, shear; oblique; standing; AMR; FP32 | P1 | Stone et al. 2008, ApJS 178, 137 (Athena, sec. 8.1), arXiv:0804.0402 | `test_hydro_accuracy_linear_waves.py` |
| A2 | AMR seam pair: linear wave across levels; shock through a coarse-fine face | P1 | Athena++ `amr_linwave` regression test; Teyssier 2002, A&A 385, 337, sec. 3.5 | open |
| A3 | Liska-Wendroff smooth 2D advection | P1 | Liska & Wendroff 2003, SIAM J. Sci. Comput. 25, 995, sec. 4.1 | `test_hydro_accuracy_lw_smooth.py` |
| A4 | Gresho vortex in revolution units, static and moving | P1 | Miczek et al. 2015, A&A 576, A50, arXiv:1409.8289; LW 2003 sec. 4.10 | `test_hydro_hard_lw_gresho.py`, `test_hydro_hard_low_mach.py` |
| A5 | Smooth Kelvin-Helmholtz, mode amplitude M(t) | P1 | McNally, Lyra & Passy 2012, ApJS 201, 18, arXiv:1111.1764 (erratum E3: the quoted rate exp(4.384 t) does not belong to this profile, whose linear rate is 2.83; never bound against it) | `test_hydro_hard_mcnally_kh.py` |
| A6 | Periodic isothermal atmosphere in phi = sin(2 pi x) | P1 | Berberich et al. 2021, arXiv:2005.01811, sec. 3.1 | `test_hydro_hard_isothermal_atmosphere.py` |
| A7 | Linear Jeans wave and instability, fitted sigma | P1 | Athena++ `grav/jeans_3d` regression tests; sigma^2 = c^2 k^2 - 4 pi G rho0 | `test_gravity_jeans_wave.py` |
| A8 | Evrard collapse energy budget | P1 | Springel 2010, MNRAS 401, 791, sec. 9.1, arXiv:0901.4107 (Evrard 1988) | `test_gravity_evrard_energy.py` |
| A9 | Odd-even decoupling (Woodward-Colella blast on a strip) | P1 | Liska & Wendroff 2003, sec. 4.9, Table 4.4 | `test_hydro_hard_odd_even.py` |
| A10 | Hui/LW implosion, diagonal symmetry | P2 | Liska & Wendroff 2003, sec. 4.7 (Hui et al. 1999) | `test_hydro_hard_implosion.py` |
| A11 | Kurganov-Tadmor 2D Riemann configurations 3 and 6 | P2 | Kurganov & Tadmor 2002, Numer. Meth. PDE 18, 584 | `test_hydro_hard_kt_riemann.py` |
| A12 | Sedov per Kamm-Timmes: cylindrical 2D, spherical 3D, singular omega | P2 | Kamm & Timmes, LA-UR-07-2849 | open |
| A13 | Single-mode Rayleigh-Taylor, LW all-wall box, exact linear eigenmode | P2 | Liska & Wendroff 2003, sec. 4.6 | `test_hydro_hard_rayleigh_taylor.py` (not yet run in full) |
| A14 | Single-mode Richtmyer-Meshkov in a closed tube (theta-group states, our design) | P2 | Richtmyer 1960, CPAM 13, 297 (impulsive rate, unread); states from Thornber et al. 2017, arXiv:1706.09991 | open |
| A15 | Taylor-Green numerical Reynolds number | P2 | Wongwathanarat, Grimm-Strele & Mueller 2016, arXiv:1607.04272 (APSARA) | open |
| A16 | Adiabatic expansion; pressureless dust collapse | P2 | Enzo test problems (`run/`) | open |
| A17 | Keplerian disc in a softened potential, angular-momentum drift | P2 | Hopkins 2015, MNRAS 450, 53, arXiv:1409.7395 (GIZMO) | open |
| A18 | Guderley surrogate: spherical shock tube to focus, 2D | P2 | Ramsey & Shashkov, OSTI 1045963 | open |
| A19 | Stroemgren sphere (Iliev test 1) and M1 failure gates | P2 | Iliev et al. 2006, MNRAS 371, 1057; Rosdahl et al. 2013, arXiv:1304.7126 | open |

## B: exact per-face ghosts

| ID | Test | P | Leading reference |
|---|---|---|---|
| B1 | Noh 2D and 3D | P1 | Noh 1987, JCP 72, 78 (unread; via ExactPack `noh`, Gehmeyr et al. 1997) |
| B2 | Noh-2 uniform collapse | P1 | Noh 1987, JCP 72, 78, p. 92 (unread; via ExactPack `noh2`) |
| B3 | Kidder homogeneous isentropic compression (Cog6) | P1 | Kidder 1974, Nucl. Fusion 14, 53 |
| B4 | Kaeppeli-Mishra atmosphere and 3D polytrope; Berberich 2D polytrope | P1 | Kaeppeli & Mishra, SAM report 2013-05 (JCP 259, 199, 2014) |
| B5 | Coggeshall solution 8 (smooth 2D exact flow) | P2 | Coggeshall, LA-UR-93-2505 (OSTI 10180432) |
| B6 | Double Mach reflection on Kemm's [0,4] x [0,2] | P2 | Woodward & Colella 1984, JCP 54, 115; Kemm's domain |
| B7 | Glaz-Wardlaw / Hui steady two-section Riemann problem | P2 | Glaz & Wardlaw 1985 (via ExactPack `riemann2D_2section_steadystate`) |
| B8 | Shock-vortex interaction | P3 | Inoue & Hattori 1999, JFM 380, 81; Rault et al. |
| B9 | Generalized Noh, ideal-gas instance | P2 | Velikovich, Giuliani & Zalesak 2018, PRE 98, 013105 and JCP 374, 843 |
| B10 | Su-Olson Marshak wave; Lowrie-Edwards radiative shocks | P2 | Su & Olson 1996; Lowrie & Edwards 2008, Shock Waves 18, 129 (via ExactPack) |

## C to F: missing physics

| ID | Test | Tier | P | Leading reference |
|---|---|---|---|---|
| C1 | Shock-bubble | C | P3 | Haas & Sturtevant 1987, JFM 181, 41; Quirk & Karni 1996, JFM 318, 129 |
| D1 | Conduction ladder: ExactPack heat, Coggeshall 8-18, RMTV | D | P1 for the module | Reinicke & Meyer-ter-Vehn 1991; Coggeshall 1991 (via ExactPack) |
| D2 | Kelvin-Helmholtz with explicit diffusion, dye error vs Dedalus | D | P2 | Lecoanet et al. 2016, MNRAS 455, 4274, arXiv:1509.03630 |
| D3 | Taylor-Green at Re = 1600 | D | P2 | DeBonis 2013, NASA/TM (NTRS 20130011044) |
| D4 | Anisotropic-conduction ring | D/E | P3 | Sharma & Hammett 2007, JCP 227, 123 |
| E1 | Decaying MHD turbulence comparison | E | P3 | Kritsuk et al. 2011, ApJ 737, 13, arXiv:1103.5525 |
| F1 | Santa Barbara cluster | F | P3 | Frenk et al. 1999, ApJ 525, 554, arXiv:astro-ph/9906160 |

## G: GPU resolutions

| ID | Test | P | Leading reference |
|---|---|---|---|
| G1 | Theta-group narrowband Richtmyer-Meshkov, 3D (per-axis boundaries) | P1 | Thornber et al. 2017, Phys. Fluids 29, 105107, arXiv:1706.09991 |
| G2 | Agertz blob (LUNA regression only, no reference yet: `tests/test_hydro_hard_agertz_blob.py`) | P2 | Agertz et al. 2007, MNRAS 380, 963, arXiv:astro-ph/0610051 |
| G3 | Forced isothermal turbulence statistics | P2 | Federrath et al. 2010, A&A 512, A81, arXiv:0905.1060 |
| G4 | Guderley exact, 2D and 3D, through focus | P2 | Ramsey, Kamm & Bolstad, LA-UR-09-05853; Lazarus 1981 |
| G5 | Static and translated King sphere (deep AMR) | P3 | Tasker et al. 2008, MNRAS 390, 1267, arXiv:0808.1844 |
| G6 | Non-linear thin-shell instability (per-axis boundaries) | P2 | Hubber, Falle & Goodwin 2013, arXiv:1304.3456, sec. 3.2 (Vishniac 1994) |

## Proposed, not yet tiered by the author

Listed in `docs/TEST_PLAN.md` under "To add" until decided: detonation and Type Ia (XA-XW), engineering
CFD (VA-VE), radiative shocks with conduction, sinks and collapse (Bondi, Shu, Larson-Penston),
cooling and stratified (Kim & Ostriker 2015, Field 1965, Dintrans & Brandenburg 2004).
