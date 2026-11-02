"""Each shipped run script's two-step smoke case, and the run and checks the step tests share."""
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUIET = ["max_steps=2", "write_snapshots=0", "enable_checkpointing=0", "PLOT_AFTER_SIM=0"]
TASK = ["n_steps=2", "n_iter=1"]


class ScriptCrashed(Exception):
    """A run script exited non-zero with the failure its case expects."""


class MassMismatch(AssertionError):
    """The gas mass after two steps is not what the boundary and the sources account for."""


@dataclass(frozen=True)
class Case:
    """One script: its argv, the group it runs in, how its mass is checked, a known crash."""
    argv: tuple
    group: str
    mass: str = "closed"            # closed | budget | cc85 | particles | none
    why_not: str = ""
    crash: str = ""                 # strict xfail: text the crash must print
    mass_bug: bool = False          # strict xfail: the mass check fails for `reason`
    reason: str = ""
    nonempty: tuple = field(default=())
    faces_core: tuple = ()          # (Nx, L, r_lo, r_hi, ndim): first-order faces (C8) required, all in that shell


CC85_F32 = "the CC85 injection adds a float32 update to the fp64 state, so the step runs in float32"

CASES = {
    "scripts.cosmo.run_cosmo_dm": Case(("mesh=16", *QUIET), "heavy", mass="particles"),
    "scripts.cosmo.run_cosmo_hydro": Case(("mesh=16", *QUIET), "heavy"),
    "scripts.examples.autodiff.blast_emergence_run": Case(("nx=16", *TASK), "autodiff", nonempty=("loss",)),
    "scripts.examples.autodiff.ic_inversion_run": Case(("nx=16", *TASK), "autodiff", nonempty=("ic_residual",)),
    "scripts.examples.autodiff.kh_corrector_run": Case(("nx=16", "n_steps=25", "n_iter=1"), "autodiff",
                                                      nonempty=("coarse_mse", "corr_mse")),
    "scripts.examples.autodiff.m82_autodiff_run": Case(("nx=8", *TASK), "autodiff", nonempty=("eta_recovered",)),
    "scripts.examples.autodiff.turb_autodiff_run": Case(("nx=16", *TASK), "autodiff", nonempty=("ic_residual",)),
    "scripts.examples.autodiff.turb_emergence_run": Case(("nx=16", *TASK), "autodiff", nonempty=("loss",)),
    "scripts.examples.autodiff.turbgrav_autodiff_run": Case(("nx=16", *TASK), "autodiff",
                                                           nonempty=("ic_residual",)),
    "scripts.examples.runs.boss_bodenheimer_periodic_minmod_run": Case(
        ("nx=16", *QUIET), "hydro", crash="IC requires cfg.units enabled",
        reason="the Boss-Bodenheimer IC needs cfg.units; this script runs in code units"),
    "scripts.examples.runs.boss_bodenheimer_run": Case(
        ("nx=16", *QUIET), "hydro", crash="IC requires cfg.units enabled",
        reason="the Boss-Bodenheimer IC needs cfg.units; this script runs in code units"),
    "scripts.examples.runs.cc85_wind_run": Case(("nx=32", "boundary_mass_budget=1", *QUIET), "hydro", mass="cc85",
                                                mass_bug=True, reason=CC85_F32),
    # nx=32: at nx=16 one cell spans the disc's inner edge (R_in=0.4) and the damping zone (R < 0.5)
    "scripts.examples.runs.disc_planet_run": Case(
        ("nx=32", *QUIET), "hydro", mass="none", faces_core=(32, 6.0, 0.0, 0.5, 2),
        why_not="damping zones relax the disc to its IC and the planet sink accretes"),
    "scripts.examples.runs.maclaurin_run": Case(("nx=16", *QUIET), "hydro"),
    "scripts.examples.runs.poisson_accuracy_run": Case(("nx=16",), "hydro", mass="none",
                                                       why_not="a Poisson solve, no gas"),
    "scripts.examples.runs.pulsating_polytrope_run": Case(("nx=16", *QUIET), "hydro"),
    "scripts.validation.runs.bonnor_ebert_run": Case(("nx=16", *QUIET), "hydro"),
    "scripts.validation.runs.boss_bodenheimer_iso_run": Case(("nx=32", "boundary_mass_budget=1", *QUIET), "hydro",
                                                             mass="budget"),
    "scripts.validation.runs.driven_turbulence_run": Case(("nx=16", *QUIET), "hydro"),
    # first-order faces at the cloud edge R = 1 only: a face's cell sees the jump within its slope stencil
    # (sqrt(3) dx) after the edge moved at most 2 x CFL 0.8 cells: R +- (sqrt(3) + 1.6) dx, dx = 0.25
    "scripts.validation.runs.evrard_collapse_run": Case(
        ("nx=16", *QUIET), "hydro",
        faces_core=(16, 4.0, 1.0 - (3 ** 0.5 + 1.6) * 0.25, 1.0 + (3 ** 0.5 + 1.6) * 0.25, 3)),
    "scripts.validation.runs.gresho_run": Case(("nx=16", *QUIET), "hydro"),
    "scripts.validation.runs.jeans_instability_run": Case(("nx=16", *QUIET, "write_snapshots=1"), "hydro"),
    "scripts.validation.runs.kh_run": Case(("nx=16", *QUIET), "hydro"),
    "scripts.validation.runs.m82_wind_run": Case(("nx=32", "boundary_mass_budget=1", *QUIET), "hydro", mass="cc85",
                                                 mass_bug=True, reason=CC85_F32),
    "scripts.validation.runs.sedov3d_amr_run": Case(("nx=16", "t_end=1e-4", *QUIET), "heavy"),
    "scripts.validation.runs.sedov3d_run": Case(("nx=16", "boundary_mass_budget=1", *QUIET), "hydro",
                                                mass="budget"),
    "scripts.validation.runs.sod3d_amr_run": Case(("nx=16", "nt=16", "t_end=1e-3", *QUIET), "heavy"),
    "scripts.validation.runs.starbench_hii_run": Case(("nx=16", "boundary_mass_budget=1", *QUIET), "heavy",
                                                      mass="budget"),
}


def group(name):
    """The script names of one group, sorted."""
    return sorted(k for k, c in CASES.items() if c.group == name)


def params(name):
    """pytest params for a group, a known crash marked strict xfail with its cause."""
    out = []
    for k in group(name):
        c = CASES[k]
        marks = []
        if c.crash or c.mass_bug:
            marks = [pytest.mark.xfail(strict=True, raises=ScriptCrashed if c.crash else MassMismatch,
                                       reason=c.reason)]
        out.append(pytest.param(k, marks=marks, id=k.rsplit(".", 1)[-1]))
    return out


def music_ic(path):
    """A small MUSIC gas + DM IC (the growing-mode one the cosmology tests use)."""
    from tests._music_growing_mode import growing_mode_ic
    growing_mode_ic(str(path))
    return str(path)


def run(name, tmp_path):
    """Run `name` in its own process on CPU; (return code, output, record)."""
    c = CASES[name]
    argv = list(c.argv)
    if c.group != "autodiff" and not name.endswith("poisson_accuracy_run"):
        argv += [f"output_dir={tmp_path / 'run'}", "clear_output_dir=1"]
    if name.startswith("scripts.cosmo."):
        argv.append("ic=" + music_ic(tmp_path / "ic.hdf5"))
    out = tmp_path / "record.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), ROOT,
                                                       os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "_run_script_smoke.py"), str(out), name,
                        *argv], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    text = p.stdout + p.stderr
    rec = json.load(open(out)) if p.returncode == 0 else None
    return p.returncode, text, rec


def check_core_faces(c, text):
    """First-order faces (C8) only where the case allows them: in the cells whose centre lies in the shell
    r_lo <= r < r_hi about the box centre, at most 2 * ndim faces per such cell per step, and at least one
    (else the opt-in is not needed); a case without faces_core has none (it would have been refused)."""
    lines = re.findall(r"^\[hydro\] step \d+ t=\S+: (\d+) first-order faces \(C8\)(.*)$", text, re.M)
    if not c.faces_core:
        assert not lines, f"first-order faces in a case that does not allow them: {lines[:2]}"
        return
    n_x, L, r_lo, r_hi, ndim = c.faces_core
    centre = (np.arange(n_x) + 0.5) * (L / n_x) - 0.5 * L
    r_cells = np.sqrt(sum(g * g for g in np.meshgrid(*([centre] * ndim), indexing="ij")))
    n_in = int(np.sum((r_cells >= r_lo) & (r_cells < r_hi)))
    n = sum(int(k) for k, _ in lines)
    print(f"FACES {n} first-order faces in 2 steps; {n_in} cells in {r_lo} <= r < {r_hi}")
    assert 1 <= n <= 2 * 2 * ndim * n_in, f"{n} first-order faces, outside [1, {2 * 2 * ndim * n_in}]"
    cell = r"\(" + r", ".join([r"(\d+)"] * ndim) + r"\)"
    for k, where in lines:
        # the log lists the first FACE_REPORT_CELLS cells per shard (time_integration/integrators.py)
        assert 0 < len(re.findall(r"[xyz][+-]", where)) <= int(k), where
        for idx in re.findall(cell, where):
            r = float(np.sqrt(sum(centre[int(i)] ** 2 for i in idx)))
            assert r_lo <= r < r_hi, f"a first-order face at cell {idx}, r={r:.3f} outside [{r_lo}, {r_hi}): {where}"


def check(name, tmp_path):
    """Run `name` and assert: IC built, finite, two steps, time advanced, mass as its case says."""
    c = CASES[name]
    code, text, rec = run(name, tmp_path)
    if code != 0:
        if c.crash and c.crash in text:
            raise ScriptCrashed(c.reason)
        pytest.fail(f"{name} crashed (exit {code}):\n{text[-3000:]}")
    assert rec["done"]
    if name.endswith("poisson_accuracy_run"):
        return poisson_ratio(text)
    assert rec.get("ic"), "the IC was never built"
    if "m0" in rec:
        assert rec["ic_finite"], "the IC is not finite"
    if c.group == "autodiff":
        assert rec["task_finite"], "the task returned a non-finite number"
        for k in c.nonempty:
            assert rec["task_sizes"].get(k, 0) > 0, f"the task returned no {k}"
        return rec
    assert rec["n_steps"] == 2, f"took {rec['n_steps']} steps, not 2"
    check_core_faces(c, text)
    assert rec["t"] > rec["t0"], f"time did not advance: t={rec['t']} from {rec['t0']}"
    if "m1" in rec:
        assert rec["finite"], "the state after two steps is not finite"
    if c.mass == "none":
        return rec
    if c.mass == "particles" or "n_dm0" in rec:
        assert rec["n_dm1"] == rec["n_dm0"] and rec["dm_finite"], "dark matter particles lost or not finite"
        assert rec["m_dm1"] == pytest.approx(rec["m_dm0"], rel=1e-12), "dark matter mass changed"
    if c.mass == "particles":
        return rec
    m0, m1 = rec["m0"], rec["m1"]
    tol = 8 * np.finfo(rec["dtype"]).eps * max(m0, m1)
    net = 0.0
    if c.mass in ("budget", "cc85"):
        b = np.asarray(rec["budget"])
        net = float(b[0].sum() - b[1].sum())
    if c.mass == "cc85":
        assert rec["injected"] > 0.0, "no cell injects: the grid is too coarse for the CC85 sphere"
        net += rec["injected"]
    print(f"MEASURED {name}: t {rec['t0']} -> {rec['t']:.6e}, mass {m0:.17g} -> {m1:.17g}, "
          f"(change - expected)/m = {(m1 - m0 - net) / max(m0, m1):.2e} ({c.mass}, {rec['dtype']})")
    if abs(m1 - m0 - net) > tol:
        raise MassMismatch(f"mass {m0:.17g} -> {m1:.17g}: change {m1 - m0:.12e}, "
                           f"expected {net:.12e} ({c.mass}), tolerance {tol:.1e}")
    return rec


def poisson_ratio(text):
    """The script's best potential amplitude against the discrete Laplacian's own prediction."""
    nx = int(re.search(r"Poisson accuracy: N=(\d+)", text).group(1))
    ratio = float(re.search(r"ratio\(n_extra=80\)=([0-9.]+)", text).group(1))
    half = np.pi / nx                               # k dx / 2 for the fundamental mode
    predicted = (half / np.sin(half)) ** 2
    assert abs(ratio - predicted) < 2e-4, f"phi ratio {ratio} vs discrete prediction {predicted:.5f}"
    return ratio
