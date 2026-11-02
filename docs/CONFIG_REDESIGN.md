# A cleaner Config: proposal, steps 1-4 done

**Status (branch config-problem, 2026-10-05).** Step 4 is implemented: the 67
fields one IC reads are its module's `Params` (`cfg.problem`, `problem.x=` on the
command line), and the shared ones went to the feature that reads them at run
time (`cc85_mu` -> `cooling_mu`, `cloud_mu` -> `sink_star_mu`, `cloud_seed` ->
`seed`; the M82 potential stays flat). 289 fields on main e10b784, 222 after.
A moved flat name fails with its new key (`user_config.MOVED`); the AMUSE worker
translates the old keys for one release. Step 5 is not done.

**Status (branch config-redesign, 2026-10-01).** Steps 1-3 of section 4 are
implemented: the dead fields and the aliases are out (a retired name fails with
its new place, `luna/setup/user_config.RETIRED`), validate() refuses every
unimplemented option, a config.py attribute Config does not know raises, the
16 experiment fields are the example tasks' options, and template_config.py is
sectioned by subject. 301 fields on main c145578, 276 after (45 core, 164
feature, 67 problem). Outputs are bitwise those of main; only the config record
(`config_json`, run.json) lost its retired keys and `run` reads "gas" for
"auto". Steps 4-5 are open. The text below is the proposal as reviewed.

Written 2026-09-30 on branch `cleanup-1` (main fcf7424). Every count below is
from that tree: the reader of each field was found by grep and then opened.
Counts marked ~ are by category and approximate to a few fields; the named
fields are exact.

## 1. What there is today

`luna/template_config.py` defines one frozen dataclass with **307 fields in
29 comment sections** (docs/CONFIG.md is generated from it). By what reads
each field:

| kind | fields | examples |
|---|---|---|
| core: read on every run | ~44 | `Nx`..`Lz`, `gamma`, `CFL`, `ORDER`, `LIMITER`, `riemann_solver`, `t_end`, output, restart, `units`, `MESH_SHAPE` |
| an optional feature | ~160 | radiation 26 + ionization 11, sinks ~20, AMR 19, cosmology/DM/floors ~30, gravity/Poisson ~16, cooling 7, particles 13, CC85, barotropic, OU, autodiff 9 |
| one test problem's IC only | 67 | `jeans_*` 7, `evrard_*` 4, `be_*` 5, `lane_*` 4, `mac_*` 4, `bb_*` 7, `cloud_*` 15, `disc_*` 6, `m82_disk_*` 4, `zel_*` 2, `turb_*` 2, `gresho_Mmax`, `kh_noise_amp`, `soundwave_single_mode`, `pulse_amp`, `hii_rho0`, `cc85_n/T_ambient` |
| shared by an IC and a source | 8 | `m82_stellar_*`/`m82_halo_*` (IC + external gravity), `cc85_mu` (IC + **cooling's** mean molecular weight), `cloud_mu`, `cloud_seed` (IC + Torch worker) |
| one example only (scripts/examples/autodiff) | 16 | `autodiff_emergence_*` 11, `autodiff_ic_*` 4, `autodiff_n_iter` |
| read by nothing | 3 | `production_mode` (set in 40 files, read nowhere), `check_memory`, `bb_rho_ambient` |
| aliases of another field | 4 (+1) | `CosmoDM`, `CosmoHydro` -> `run`; `dm_mixed_precision`, `dm_fp32` -> `gravity_precision`; `DECOMPOSITION` is never set and `MESH_SHAPE` overrides it |

**The sections do not describe the fields.** "Domain" holds 93 fields: the
grid, but also all of cosmology, DM, gravity, Poisson and seven problems' IC
parameters. "Riemann solver selection" holds the hydro kernel switches and
`gravity_kernel`. "M1 radiation" holds three Torch feedback fields. "Sink
particles" holds the Torch star fields, the whole `cloud_*` IC and `ic_dir`.
"Knobs declared 2026-09-16" is a date, not a subject (cooling, sinks, disc,
`dm_pdims`).

**Accepted, documented, not implemented** (validate() passes them):

| field | value | what happens |
|---|---|---|
| `ORDER` | 2 ("PPM") | runs PLM, silently; 23 files set ORDER=2. Any value but 0 is PLM; AMR ignores ORDER |
| `BOUNDARY` | 3 | validate() allows it; the hydro halo refuses it (only valid as `POISSON_BOUNDARY`) |
| `decomposition` | "particles" | documented, raises |
| `riemann_solver` | "hll" | 1D-3D since 2026-10-06; nothing in production uses it |
| `LIMITER` | 2 (van Leer) | implemented, used by no test or run |
| `amr_criterion` | "static" | implemented by the driver, rejected by validate(); unknown values fall back to density |
| `amr_reflux` | "iface" | 2-process hang open; only "repl" is used |
| `rad_ion_solver` | "split" | never used |
| `cooling_table` | "ki", "ki02", "ki_sd93" | undocumented spellings of two tables |
| `DECOMPOSITION` | anything but auto/slab | falls into "block" without an error |

Not checked by validate() at all: ORDER, DECOMPOSITION, decomposition, run,
rad_scheme, rad_order, rad_ion_solver, the three kernel switches,
cooling_table, gravity_mg_*, amr_reflux, particle_boundary.

**Two names for one thing:** snapshot cadence (`time_bet_snaps`,
`cosmo_snapshot_a`, `amr_n_snaps`), checkpoint cadence (wall time vs
`cosmo_checkpoint_every`), process grid (`MESH_SHAPE` vs `dm_pdims`), five
mean molecular weights (`cosmo_mu`, `cc85_mu`, `cloud_mu`, `cloud_amb_mu`,
`barotropic_mu`), warm V-cycles (`gravity_warm_vcycles`, `dm_warm_vcycles`,
`amr_grav_vcycles`), `DECOMPOSITION` vs `decomposition` (different meanings,
same word). Casing is mixed: UPPER (`SPATIAL_DIMENSION`, `POISSON_*`, `AMR`),
Camel (`CosmologicalIntegration`), lower (the rest).

**Hidden second defaults:** 10 `getattr(cfg, "x", default)` calls whose
default differs from the field's (e.g. `sink_max_count` 64 in the Torch
worker vs 4 in Config), inert only because every cfg is a full Config; and
~20 sentinel defaults (0/None meaning "compute it": sink radii, feedback
radius, `rad_c_true`, ...).

## 2. Where Config goes, and what a change breaks

| consumer | how | what a change breaks |
|---|---|---|
| run scripts, tests, tools | `replace(Config(), field=...)` in 43 scripts, 46 tests, 5 tools | every renamed or moved field they set (214 of 307 fields are set somewhere; 93 never are) |
| command line (`setup/cli.py`) | `field=value`, parsed by the annotation; nested dataclasses (`units`, `ou_config`) cannot be set | a nested Config needs dotted keys, `radiation.chat=3` (small: split on the dot, recurse) |
| user `config.py` (`setup/user_config.py`) | projected onto the template; unknown attributes WARN and are dropped | an old config.py with a moved field would run with the default. This must become an error that names the new place |
| `run.json`, HDF5 `config_json` | `json.dumps(asdict(cfg))` on every snapshot and checkpoint | **nothing reads it back into a Config** (cosmo_checkpoint skips it; grid restart compares only Nx..Lz, SPATIAL_DIMENSION, BOUNDARY, gamma as separate attributes). Only two tests read keys. A new layout changes the provenance attribute, so bitwise comparisons across the change must exclude it |
| Torch worker (`src/amuse_luna/interface.py`) | `set_config_value(key, float)`/`set_config_string` with raw field names; `$LUNA_CONFIG` names a config file | keys the coupling scripts send; only `scripts/coupling/amuse_spawn_test.py` sends any (12 grid/hydro keys). The Tier-1 runs use `$LUNA_CONFIG`, which goes through the same loader as main.py |
| `ic/ic_file.py` fingerprint | test_problem, N*, L*, gamma, `cloud_*`, units | moving `cloud_*` changes the fingerprint's keys, so a cached IC is rebuilt once |

Also worth fixing on the way (found by this survey): `run.json` is written
before `read_ics`, and the MUSIC reader then replaces the box, BOUNDARY, the
cosmology and the units (`cosmology/music_ic.py:285`), so run.json records
values the run did not use; the coupled path (`amuse_luna/stepper.build`)
calls neither validate() nor write_run_json.

## 3. Options

**A. Per-feature sub-configs.** `Config.hydro`, `.gravity`, `.cosmology`,
`.sinks`, `.radiation`, `.amr`, `.output`, `.autodiff`, each a frozen
dataclass; a feature is on when its sub-config is not None
(`radiation_enabled` disappears). Reads well (`cfg.radiation.chat`), the
section IS the type, validate() splits per feature. Cost: every one of the
~95 files that build a Config changes; `cfg.Nx` becomes `cfg.grid.Nx` in
~150 src readers; the CLI and the Torch keys need dotted names. Highest
churn, for the core fields that nobody finds confusing.

**B. Flat, fewer fields.** Keep one flat dataclass; delete the dead fields
and aliases, move the problem and experiment fields out, validate every
enumerated field, re-section. Leaves ~210 fields, still one long list, and
the feature fields (radiation's 37) still sit next to the grid.

**C. Hybrid (recommended).** Flat core (the ~44 fields every run reads keep
their names: `Nx`, `CFL`, `test_problem`, ... -- the top-30 most-set fields
are all core), plus:

- `cfg.problem`: the IC's own parameters, a frozen dataclass declared in the
  IC module (`ic_generation/jeans3d.py: Params(rho0, P0, amp, kx, ky, kz,
  eigenmode)`), selected by `test_problem`, defaulting to that module's
  `Params()`. 72 fields leave Config; a reader of `jeans3d.py` finds the
  problem's knobs in the same file.
- sub-configs only for the big optional features whose fields are read in
  one place: `radiation` (37 incl. ionization), `sinks` (~20), `amr` (19),
  `cosmology` (~30 incl. DM and floors). Their `*_enabled` flag becomes
  `is not None`. Smaller features (cooling 7, OU 2, CC85 6, particles 13,
  barotropic 4) stay flat with their prefix.
- experiment parameters are the examples' own task options (already how
  `movie=`, `forward_only=` work after cleanup-1).

## 4. Recommendation and migration, in small steps

Each step is one PR, suite green, outputs bitwise equal except the
`config_json` provenance attribute (compare.py already strips run.json
noise; add `config_json` to its ignore list for these PRs only).

1. **Free cleanups, no layout change.** Delete `production_mode`,
   `check_memory`, `bb_rho_ambient`; stop reading `be_rho_c` or stop
   overwriting it; make validate() check every enumerated field (ORDER in
   {0, 1}, BOUNDARY in {0, 1, 2, 4} for hydro, decomposition, run,
   cooling_table, rad_*, kernels, amr_reflux; drop "particles"; accept
   "static" or remove it from the driver); make `from_user_config` raise on
   unknown attributes instead of warning; replace the 10 conflicting
   `getattr` defaults with plain reads. Re-section template_config.py by
   subject (no field changes). Remove ORDER=2 from the 23 files that set it
   (it is PLM) or make it an error after they are fixed.
2. **Aliases out.** `CosmoDM`/`CosmoHydro` -> `run`; `dm_fp32`/
   `dm_mixed_precision` -> `gravity_precision`; `DECOMPOSITION` -> removed
   (MESH_SHAPE covers it). A moved name raises with its replacement (a small
   table in user_config.py and cli.py, kept one release).
3. **Experiment fields out.** The 16 `autodiff_emergence_*`, `autodiff_ic_*`,
   `autodiff_n_iter` become keyword options of the example tasks
   (scripts/examples/autodiff/), next to `movie=` and `forward_only=`.
   `autodiff_task` stays (it is how `python main.py` picks an example).
4. **`cfg.problem`.** One IC module at a time: declare `Params`, read
   `cfg.problem.x`, give the CLI dotted keys (`problem.amp=0.01`), update
   that problem's run scripts and tests. The shared fields (M82 potential,
   `cc85_mu`, `cloud_mu`, `cloud_seed`) move to the feature that uses them at
   run time, not to the IC. Start with jeans/evrard/BE/lane/mac (grav tests,
   few users), end with `cloud_*` (the Torch runs; coordinate with them).
5. **Feature sub-configs**, one per PR in the order radiation, sinks, AMR,
   cosmology. Decide after step 4 whether they are worth it: if step 4
   leaves a Config short enough to read, stop there (B+problem).

What breaks, all at step boundaries and all named in the error: old
`config.py` files that set a moved field; run scripts and tests (updated in
the same PR); CLI `field=value` for a moved field (the error names the dotted
key); Torch coupling keys for sinks/radiation fields in step 5 (the worker
maps the old flat keys for one release, since Torch scripts live outside this
repo). Checkpoints and snapshots stay readable: nothing reads config_json.

**In five lines:** keep a flat core (the ~44 fields every run reads keep
their names); move the 72 IC-only fields into each IC module's own `Params`
(`cfg.problem`) and the 16 experiment fields into the example tasks; delete
the 3 dead fields and 4 aliases; make validate() reject every unimplemented
option (ORDER=2, BOUNDARY=3, ...); add sub-configs for radiation, sinks, AMR
and cosmology only if the result is still too long, one PR per step.
