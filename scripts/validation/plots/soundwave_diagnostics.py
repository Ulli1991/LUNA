"""3-way diagnostic for the residual 1st-order phase error in LUNA soundwave.

Runs three additional configurations in sequence and prints the L1 and slope
of each. Used to localize whether the 1st-order phase error comes from:
  (A) The single-eigenmode IC interacting with the scheme (u=0 IC test).
  (B) The PLM reconstruction (ORDER=0 first-order Godunov test).
  (C) The time integration / CFL scaling (CFL=0.2 test).

Compare to the baseline (current eigenmode, PLM, CFL=0.4): slope ≈ -1.00.

Submitted with `submit_validation_1d.sh` setting VAL_TARGET=diagnostic.
"""
import numpy as np

from scripts.validation.runner import sweep
from scripts.validation.metrics import l1_error, convergence_slope
from luna.ic_generation import soundwave
from luna.file_io.read_snapshot import read_snapshot, latest_snapshot_dir


NX_SCAN = [128, 256, 512, 1024, 2048, 4096]
LX = 1.0
GAMMA = 5.0 / 3.0
RHO0 = 1.0
P0 = 0.6
AMP = 1e-4
M = 1
CS = np.sqrt(GAMMA * P0 / RHO0)
T_END = LX / CS


def analytic(x):
    return RHO0 + AMP * np.sin(2.0 * np.pi * M / LX * x)


COMMON = dict(
    test_problem="Soundwave",
    SPATIAL_DIMENSION=1,
    BOUNDARY=0,
    Lx=LX,
    gamma=GAMMA,
    t_end=T_END,
    time_bet_snaps=T_END,
    LIMITER=1,        # MonCen
    PLOT_AFTER_SIM=False,
    enable_x64=True,
    clear_output_dir=True,
)


def run_scan(name, overrides, output_template):
    print(f"\n========== {name} ==========")
    base = dict(COMMON)
    base.update(overrides)
    results = sweep(base, "Nx", NX_SCAN, output_dir_template=output_template)
    l1s = []
    for Nx, outdir in results:
        snap = read_snapshot(latest_snapshot_dir(outdir))
        l1 = l1_error(snap["rho"], analytic(snap["x"]))
        l1s.append(l1)
        print(f"  Nx={Nx:5d}  L1={l1:.3e}")
    slope = convergence_slope(NX_SCAN, l1s)
    print(f"  slope L1 = {slope:.3f}")
    return l1s, slope


def main():
    # (A) OLD u=0 superposition IC with current (BC-fixed) scheme
    A_l1, A_slope = run_scan(
        "A: u=0 superposition IC (OLD), PLM/MUSCL-Hancock, CFL=0.4",
        dict(problem=soundwave.Params(single_mode=False), ORDER=1, CFL=0.4),
        "snapshots/diag_A_u0_Nx{}",
    )

    # (B) ORDER=0 first-order Godunov, eigenmode IC
    B_l1, B_slope = run_scan(
        "B: ORDER=0 (Godunov), single eigenmode, CFL=0.4",
        dict(problem=soundwave.Params(single_mode=True), ORDER=0, CFL=0.4),
        "snapshots/diag_B_order0_Nx{}",
    )

    # (C) Half CFL = 0.2, eigenmode IC, PLM
    C_l1, C_slope = run_scan(
        "C: CFL=0.2 half timestep, single eigenmode, PLM/MUSCL-Hancock",
        dict(problem=soundwave.Params(single_mode=True), ORDER=1, CFL=0.2),
        "snapshots/diag_C_cfl02_Nx{}",
    )

    print("\n\n===== SUMMARY =====")
    print(f"A (u=0 IC)         slope = {A_slope:.3f}   L1@128 = {A_l1[0]:.3e}")
    print(f"B (ORDER=0)        slope = {B_slope:.3f}   L1@128 = {B_l1[0]:.3e}")
    print(f"C (CFL=0.2)        slope = {C_slope:.3f}   L1@128 = {C_l1[0]:.3e}")
    print("Reference (eigenmode, ORDER=1, CFL=0.4): slope=-1.00, L1@128=1.27e-6")
    print("  - If A is -2 and B/C are -1: u=0 IC works; bug only with eigenmode.")
    print("  - If C halves the error vs reference: bug scales with dt → time integrator.")
    print("  - If C error matches reference: bug scales with dx → spatial scheme.")
    print("  - If B has the same magnitude as reference at any Nx: PLM doesn't help.")


if __name__ == "__main__":
    main()
