# Third-party code and data

What LUNA contains, ports or reads that others wrote, with source and licence. LUNA itself is
BSD-3-Clause ([LICENSE](LICENSE)). "Unknown" means the redistribution terms are not established yet;
those rows must be resolved before a public release.

## Code in this repository

| what | where | source | licence |
|---|---|---|---|
| Block-AMR layer: JAX-AMR's design (fixed-shape blocks, NaN sentinel block, refine/coarsen and prolong/restrict semantics; Wen & Luo, arXiv:2504.13750) re-implemented; upstream code is not vendored (`src/luna/amr/DESIGN.md`) | `src/luna/amr/` | https://github.com/JA4S/JAX-AMR | MIT, notice in [src/luna/amr/LICENSE-JAX-AMR](src/luna/amr/LICENSE-JAX-AMR) (shipped in the wheel) |
| Helmholtz EOS (FP64), aprox13 network, screen5 screening, sneut5 / Itoh et al. (1996) neutrino losses: JAX ports following AMReX-Astro Microphysics at commit 99b9b47, which carries F. X. Timmes' public routines | `src/luna/eos/helmholtz.py`, `src/luna/nuclear/` (coefficients in `_neutrino_coefficients.py`, `alpha_rates.py`, `light_rates.py`) | https://github.com/AMReX-Astro/Microphysics; https://cococubed.com | Microphysics: BSD-3-Clause. The underlying Timmes routines and rate fits: **unknown** (published fits; no licence stated on cococubed) |
| Chemistry networks 5 and 17 (rate equations, cooling, `calc_temp` energy tables): ports of S. Glover's chemcool / SGChem Fortran (network 17 with D+/HD channels added by U. Steinwandel) | `src/luna/chemistry/` | chemcool, as distributed with GIZMO/Arepo-family codes; not publicly released on its own | **unknown** (port of non-public Fortran; needs the author's permission) |
| Kroupa (2001) IMF sampler, vectorised port of Torch's `imf_sample` with its quirks kept | `src/luna/species/imf.py` | Torch (Wall et al.), https://github.com/torch-sf/torch | **unknown** (Torch licence not checked) |
| AthenaK problem generator that reads LUNA initial data, written in AthenaK's pgen style for AthenaK | `tools/athenak/pgen/luna_ic.cpp` | AthenaK, https://github.com/IAS-Astrophysics/athenak | BSD-3-Clause (AthenaK's header, which refers to AthenaK's LICENSE; not included here) |
| Sutherland & Dopita (1993) solar-metallicity CIE cooling table, values typed in from the paper | `src/luna/sources/cooling.py` | Sutherland & Dopita 1993, ApJS 88, 253 | published data (cited) |
| Per-element cooling rates at solar abundance (Wiersma, Schaye & Smith 2009) and other published fits (Koyama & Inutsuka, Draine, Hummer, Gong, Ostriker & Wolfire 2017) as formulas or typed-in coefficients | `src/luna/chemistry/tables.py`, `src/luna/chemistry/gow17.py`, `src/luna/sources/cooling.py` | the cited papers | published data (cited) |
| Initial-condition setups taken from papers (McNally, Lyra & Passy 2012 Kelvin-Helmholtz; Boss & Bodenheimer 1979 as set up in GANDALF, Hubber et al. 2018; Federrath et al. 2010 forcing) | `src/luna/ic_generation/`, `scripts/validation/runs/` | the cited papers | parameters only, no code copied |

## Data in this repository

| what | where | source | licence |
|---|---|---|---|
| Learned M2 radiation closure weights | `src/luna/radiation/m2_closure_w32.npz` | trained by LUNA (`luna.radiation`); seed and training commit not recorded in the file | BSD-3-Clause (LUNA's own) |
| Figures | `code_paper_figs/`, `docs/` | LUNA runs; some panels show AthenaK or Arepo output from runs by the LUNA author | BSD-3-Clause (LUNA's own); the codes' outputs as below |
| `assets/lubibi.jpeg` | `assets/` | not referenced by any tracked file; origin not recorded | **unknown** |

## Data read from outside this repository

[luna-reference-solutions](https://github.com/Ulli1991/luna-reference-solutions): its `MANIFEST.json`
holds a licence note per file. Summary (84 files):

| directory | what | licence note |
|---|---|---|
| `athenak/` (36) | the LUNA author's AthenaK runs, reduced | AthenaK is BSD-3-Clause; no third-party data terms |
| `arepo/` (1) | A16 comoving cross-check, public Arepo | Arepo is GPL-3.0; the author's own runs |
| `arepo/` (20) | white-dwarf models and burn grid, Arepo2_WD branch | **unknown**: non-public Arepo branch |
| `ramses/` (1) | mini-ramses comoving cross-check | **unknown**: mini-ramses licence not checked |
| `microphysics/` (8) | AMReX-Astro Microphysics 99b9b47 outputs | Microphysics BSD-3-Clause; underlying Timmes routines **unknown** |
| `pencil/` (2) | McNally et al. (2012) Pencil Code curves digitised from the paper's figures | **unknown**: digitised published figures (cited) |
| `spectral/` (1) | Brachet et al. (1983) curve digitised from a figure | **unknown**: digitised published figure (cited) |
| `spectral/` (2) | Taylor-Green runs of euler_3d (the LUNA author's pseudo-spectral code) | **unknown**: that repository states no licence |
| `tables/` (1) | subset of the Timmes & Swesty (2000) Helmholtz table | **unknown**: redistribution not confirmed |
| `luna/` (12) | LUNA's own outputs (regression references) | BSD-3-Clause |

Not distributed by LUNA or the reference repository, read from `$LUNA_DATA` or a user path:
the full Helmholtz table `helm_table.dat` (Timmes), chemistry tables and Fortran reference
trajectories (`$LUNA_DATA/chemistry`, docs/CHEMISTRY.md), a Torch clone (`$TORCH_SRC`), AMUSE.

## Runtime dependencies

Installed by pip, not shipped: JAX and jaxlib (Apache-2.0), NumPy (BSD-3-Clause), h5py
(BSD-3-Clause); optional matplotlib (PSF-based Matplotlib licence), pytest (MIT), Jupyter
(BSD-3-Clause), and through the `cuda12` extra NVIDIA's CUDA wheels (NVIDIA EULA).
