"""Sedov-Taylor 3D with the sedov3d_run settings: shock radius R(t) and the interior profile vs the exact solution."""
from dataclasses import replace

import numpy as np
import pytest

from luna.hydro import exact_sedov as se
from tests._helpers import check, check_conserved, evolve, gpu_only

FRACS = (0.5, 0.75, 1.0)            # of the run script's t_end (10^4 yr)
R_BOUND = {32: 1.0}                 # max |R - R_exact| in cells over FRACS; measured 0.811
PROF_BOUND = {32: (0.11, 0.12)}     # L1 behind the shock of rho / rho2, p / p2; measured 0.0812, 0.0907



def sedov_cfg(n):
    """The scripts/validation/runs/sedov3d_run.py Config at n^3, FP64, no output."""
    from scripts.validation.runs.sedov3d_run import config
    return replace(config(nx=n, n_snaps=1), enable_x64=True, write_snapshots=False,
                   output_dir="", clear_output_dir=False)


def shock_radius(r, rho, rho0, dx):
    """Outermost radius where the shell-averaged density falls through (peak + rho0) / 2."""
    edges = np.arange(0.0, r.max() + dx, 0.5 * dx)
    k = np.digitize(r.ravel(), edges) - 1
    cnt = np.bincount(k, minlength=edges.size)
    prof = np.bincount(k, rho.ravel(), minlength=edges.size) / np.maximum(cnt, 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    prof, rc = prof[:rc.size][cnt[:rc.size] > 0], rc[cnt[:rc.size] > 0]
    i0 = int(np.argmax(prof))
    m = 0.5 * (prof[i0] + rho0)
    j = i0 + int(np.argmax(prof[i0:] < m))
    return rc[j - 1] + (m - prof[j - 1]) / (prof[j] - prof[j - 1]) * (rc[j] - rc[j - 1])


def run_sedov(n, mesh_shape=None):
    """(max |R - R_exact| in cells over FRACS, L1 rho/rho2, L1 p/p2 behind the shock at t_end)."""
    from luna.ic_generation.ic import initial_conditions
    cfg = sedov_cfg(n)
    g, L = cfg.gamma, cfg.Lx
    dx = L / n
    U0 = np.asarray(initial_conditions(cfg))
    rho0, p_bg = float(U0[0].min()), float((g - 1.0) * U0[4].min())
    E = float(np.sum(U0[4] - p_bg / (g - 1.0)) * dx ** 3)
    times = [f * cfg.t_end for f in FRACS]
    states, steps = evolve(cfg, U0, times, mesh_shape)
    for t, U in zip(times, states):   # open box, but the faces see the ambient at rest: zero net flux, round-off
        check_conserved(f"sedov N={n} t={t:.4g}", U0, U)
    x = (np.arange(n) + 0.5) * dx - 0.5 * L
    r = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    dR = []
    for t, U in zip(times, states):
        R_ex = se.xi0(g) * (E * t * t / rho0) ** 0.2
        dR.append((shock_radius(r, U[0], rho0, dx) - R_ex) / dx)
    rho_ex, _, p_ex, R = se.solution(r, times[-1], E, rho0, g)
    U = states[-1]
    p = (g - 1.0) * (U[4] - 0.5 * (U[1:4] ** 2).sum(0) / U[0])
    inside = r < R
    Rdot = 0.4 * R / times[-1]
    rho2, p2 = rho0 * (g + 1) / (g - 1), rho0 * Rdot ** 2 * 2 / (g + 1)
    e_rho = float(np.mean(np.abs(U[0] - rho_ex)[inside])) / rho2
    e_p = float(np.mean(np.abs(p - p_ex - p_bg)[inside])) / p2
    print(f"MEASURED sedov N={n} mesh={mesh_shape} steps={steps} R/dx={R / dx:.2f} "
          f"dR/dx={['%.3f' % v for v in dR]}")
    return float(np.max(np.abs(dR))), e_rho, e_p


def test_sedov_shock_radius_and_profile(n=32, R_BOUND=R_BOUND, PROF_BOUND=PROF_BOUND):
    """R(t) to a fraction of a cell, interior L1 bounded, the same on 1 and 2 devices."""
    one = run_sedov(n)
    two = run_sedov(n, (2, 1, 1))
    assert np.allclose(one, two, rtol=1e-10, atol=1e-12), f"1 vs 2 devices {one} {two}"
    check(f"sedov N={n} max |R - R_exact| (cells)", one[0], R_BOUND[n])
    check(f"sedov N={n} L1 rho/rho2 behind shock", one[1], PROF_BOUND[n][0])
    check(f"sedov N={n} L1 p/p2 behind shock", one[2], PROF_BOUND[n][1])


# measured on one MI300A: R 0.785, 0.825, 0.884 cells; L1 (0.0530, 0.0561), (0.0325, 0.0298), (0.0209, 0.0182)
R_BOUND_GPU = {64: 1.0, 128: 1.1, 256: 1.2}
PROF_BOUND_GPU = {64: (0.07, 0.073), 128: (0.043, 0.039), 256: (0.028, 0.024)}


@gpu_only
@pytest.mark.parametrize("n", [64, 128, 256])
def test_sedov_gpu(n):
    """The same at 64^3, 128^3, 256^3 on 1 and 2 GPUs."""
    test_sedov_shock_radius_and_profile(n, R_BOUND_GPU, PROF_BOUND_GPU)


@pytest.mark.parametrize("gamma, alpha", [(1.4, 0.8511), (5.0 / 3.0, 0.4936)])
def test_exact_sedov_energy_constant(gamma, alpha):
    """The closed form's alpha against the classic tabulated values (Kamm 2000)."""
    check(f"Sedov alpha gamma={gamma:.4f} rel err", abs(se.sedov_alpha(gamma) - alpha) / alpha, 2e-4)
