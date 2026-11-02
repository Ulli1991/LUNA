# LUNA community-release roadmap

Synthesis of the 2026-07-31 three-way audit (engineering standards,
community readiness, code quality/coverage). Verdict: substantially above
the academic-code baseline — CPU-runnable CI suite, exemplary vendoring,
provenance manifests, grad-student-ready TUTORIAL/ARCHITECTURE, clean core
modules — with gaps concentrated in release mechanics, config validation,
and a stale-claims layer. Plan: keep this repo as the lab; assemble a
curated release repo (single `luna/` package namespace) once Phase 1 lands.

## Phase 1 — before the release-repo export (S items, ~days)

TRUTH PASS
- [ ] Drop "MHD" from README tagline + pyproject keywords (no MHD exists).
- [ ] Rewrite README NaN-guard paragraph (retry/emergency-checkpoint were
      removed; current design is single-attempt + require_finite).
- [ ] Fix stale comments: SPATIAL_DIMENSION "3D not implemented",
      AMR "2D only", ic.py Zeldovich "not registered"; BOUNDARY=4 in README.
- [ ] gen_config_md.py: capture full comment blocks (not last line);
      remove dangling docs/multigrid_poisson_plan.md reference; regenerate.

GUARDS (the traps we ourselves hit)
- [ ] Config.validate(): CFL>0.5 with ndim>=2, enum checks (BOUNDARY,
      dm_decomp, cosmo_kick_scheme), cross-field requirements
      (CosmoDM->music_ic_path), units-required features.
- [ ] main.py: friendly error for missing config.py ("cp template_config...").
- [ ] IC error message generated from _IC_REGISTRY (lists 17 of 34 today).
- [ ] Multi-process restart: HARD ERROR until fixed (reader treats
      per-rank shards as global — real bug, found by inspection).

CONSISTENCY
- [ ] One switch per idea: LUNA_MG_MIXED env vs cfg.dm_mixed_precision ->
      cfg only. ENV.md rows for RECON32 / MG_CA / MG_SMOOTHER.
- [ ] Align divergent getattr defaults (dm_warm_vcycles 0-vs-1,
      dm_mixed_precision False-vs-True across runners).
- [ ] Hoist 8.2543e-3 (x11) into units.py; cooling.py imports constants
      from units.py (the two copies disagree in the 6th digit).
- [ ] Move LUNA_EMERG_*/KHC_*/M82_FWDONLY env reads out of
      runner_autodiff.py into run scripts; LUNA_KH_NOISE -> cfg field.
- [ ] Personal-path defaults in 3 scripts -> env-overridable.

## Phase 2 — release-repo assembly

- [ ] Export curated tree under `luna/` package (kills setup/main/utils
      namespace collisions). Scripts import from luna.*.
- [ ] Tag v0.1.0, CHANGELOG.md, Zenodo DOI -> CITATION.cff
      (date-released), version single-sourced via importlib.metadata.
- [ ] CONTRIBUTING.md (test tiers from conftest docstring, markers,
      black/ruff, mass-conservation rule, compact-docstring rule),
      issue/PR templates; ruff check + coverage added to CI.
- [ ] docs/OUTPUTS.md (snapshot/checkpoint/history schemas; honest Gadget
      scope: MUSIC in + DM writer out, gas snapshots LUNA-native) and
      docs/RESTART.md; test-problem catalog generated from _IC_REGISTRY.
- [ ] SCALING_DESIGN.md + this file -> docs/dev/.

## Phase 3 — first release-repo development cycle (M/L)

- [ ] Highest-value test: coupled Zel'dovich pancake through
      run_simulation_cosmo_hydro_registered_ic (gas+DM analytic + mass
      conservation). The coupled runner has ZERO tests today.
- [ ] Precision-parity gates: warm+mixed vs cold-FP64 residual (32^3);
      RECON32/FUSED step vs default on Sedov-32^3.
- [ ] Snapshot round-trips: write_snapshot_distributed->read_snapshot;
      write_gadget_dm_snapshot->read_music_dm_ic (mutual inverses).
- [ ] Dedup pass (~450-500 lines): cosmo-hydro helpers -> cosmo_common
      (~160 L, incl. third floor copy in amr/cosmo.py); runner_cosmo_dm
      three layouts -> one loop skeleton + layout object (~130 L); shared
      warm-seed/DC32 helper (both runners duplicate it — drift risk).
- [ ] Split monoliths: run_simulation_cosmo_hydro (780 L fn),
      run_simulation_distributed (416 L fn), runner_autodiff (core loop
      vs per-paper experiments).
- [ ] Type-hint the public seam (main.run, runner entries, kernel sigs;
      6% coverage today); mypy permissive config.
- [ ] Fix multi-process restart properly; wb_integrator docstring+test;
      docstrings for the 10 undocumented IC generators.
- [ ] Config field naming unification (19 UPPER / 143 lower / 4 Camel)
      with deprecation aliases for one release.

Per-package grades (auditor): hydro A-, cosmology A-, amr B+, gravity B+,
utils B, file_io B, time_integration B, ic_generation B-, setup C+ (the
runner monoliths are the main debt). Untested-behavior risk list and full
duplication map in the audit transcripts (2026-07-31 session).
