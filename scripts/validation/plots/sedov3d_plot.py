"""Sedov 3D plot for the Arepo-units run (snapshots/sedov3d_arepo_Nx128/).

Two panels:
  Left:  spherically-averaged radial profiles (rho, v_r, P) at t = t_end,
         overlaid with the Sedov-Taylor shock radius (vertical line).
  Right: shock radius vs time R_s(t) from each snapshot (peak-density
         location), compared to the analytic R_s(t) = xi_0 (E t^2 / rho_0)^(1/5)
         with xi_0 ~ 1.15 for gamma = 5/3.

Saves figures/sedov3d.{pdf,png}.
"""
from pathlib import Path
import glob, os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scripts.validation.metrics import radial_profile


SEC_PER_YEAR = 3.15576e7
KILOPARSEC_CGS = 3.08568e21
PROTON_MASS_CGS = 1.67262e-24
BOLTZMANN_CGS = 1.38065e-16
SOLAR_MASS_CGS = 1.98892e33
PARSEC_CGS = 3.08568e18

# Match sedov3d_run.py
UNIT_LENGTH = KILOPARSEC_CGS
UNIT_MASS = 1e10 * SOLAR_MASS_CGS
UNIT_VELOCITY = 1e5
UNIT_TIME_S = UNIT_LENGTH / UNIT_VELOCITY
UNIT_DENSITY = UNIT_MASS / UNIT_LENGTH**3
UNIT_PRESSURE = UNIT_MASS / (UNIT_LENGTH * UNIT_TIME_S**2)
E0_ERG = 1.0e51
N0_CGS = 1.0
RHO0_CGS = N0_CGS * PROTON_MASS_CGS

OUTDIR = "snapshots/sedov3d_arepo_Nx128"
XI_0_3D = 1.15            # Sedov-Taylor constant for gamma=5/3 in 3D


def load_snapshot(snapdir):
    """Stitch shards in a snapdir into global arrays (rho, vx, vy, vz, p, t)."""
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"]); Nz = int(a["Nz_global"])
        Lx = float(a["Lx"]); dx = float(a["dx"]); dy = float(a["dy"]); dz = float(a["dz"])
        t = float(a["time"])
    rho = np.empty((Nx, Ny, Nz)); vx = np.empty_like(rho)
    vy = np.empty_like(rho); vz = np.empty_like(rho); p = np.empty_like(rho)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"]); zs = int(c.attrs["z_start"])
            shp = c["density"].shape
            rho[xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["density"][:]
            vx [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_x"][:]
            vy [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_y"][:]
            vz [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_z"][:]
            p  [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["pressure"][:]
    return dict(rho=rho, vx=vx, vy=vy, vz=vz, p=p, t=t,
                Lx=Lx, dx=dx, dy=dy, dz=dz, Nx=Nx)


def shock_radius_from_snap(snap):
    """Peak-density location in spherically-averaged profile, in pc."""
    cell_center = (snap["Nx"]/2 - 0.5, snap["Nx"]/2 - 0.5, snap["Nx"]/2 - 0.5)
    r_bins, rho_r = radial_profile(snap["rho"], snap["dx"], snap["dy"], snap["dz"],
                                   center=cell_center)
    idx = int(np.argmax(rho_r))
    # Convert from code (kpc) to pc
    return r_bins[idx] * UNIT_LENGTH / PARSEC_CGS


def analytic_R_s(t_yr, E0=E0_ERG, rho0=RHO0_CGS, xi=XI_0_3D):
    """Sedov-Taylor shock radius in pc, given t in years."""
    t_s = t_yr * SEC_PER_YEAR
    R_cm = xi * (E0 * t_s**2 / rho0) ** 0.2
    return R_cm / PARSEC_CGS


def main():
    snaps_dirs = sorted(glob.glob(os.path.join(OUTDIR, "snapdir_*")))
    if not snaps_dirs:
        raise SystemExit(f"No snapshots in {OUTDIR}")
    snaps = [load_snapshot(d) for d in snaps_dirs]

    # === Left panel: radial profiles at final snapshot ===
    final = snaps[-1]
    ic = final["Nx"]/2 - 0.5   # cell-index of domain centre for even Nx
    cell_center = (ic, ic, ic)
    # Physical centre coords (for v_r calculation only)
    cx, cy, cz = final["Nx"]/2*final["dx"], final["Nx"]/2*final["dy"], final["Nx"]/2*final["dz"]
    r_bins, rho_r = radial_profile(final["rho"], final["dx"], final["dy"], final["dz"], center=cell_center)
    _, p_r = radial_profile(final["p"], final["dx"], final["dy"], final["dz"], center=cell_center)
    # Radial velocity field
    X, Y, Z = np.meshgrid(
        (np.arange(final["Nx"]) + 0.5) * final["dx"],
        (np.arange(final["Nx"]) + 0.5) * final["dy"],
        (np.arange(final["Nx"]) + 0.5) * final["dz"],
        indexing="ij",
    )
    rvec = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)
    vr_field = ((X-cx)*final["vx"] + (Y-cy)*final["vy"] + (Z-cz)*final["vz"]) / np.maximum(rvec, 1e-30)
    _, vr_r = radial_profile(vr_field, final["dx"], final["dy"], final["dz"], center=cell_center)

    # Convert to physical units
    r_pc = r_bins * UNIT_LENGTH / PARSEC_CGS
    rho_cgs = rho_r * UNIT_DENSITY
    p_cgs = p_r * UNIT_PRESSURE
    vr_kms = vr_r * UNIT_VELOCITY / 1e5   # back to km/s (UnitVelocity already km/s)

    t_final_yr = final["t"] * UNIT_TIME_S / SEC_PER_YEAR
    R_s_analytic_pc = analytic_R_s(t_final_yr)

    fig, (axL, axR, axS) = plt.subplots(1, 3, figsize=(15, 4.4))
    color_rho, color_p, color_v = "C0", "C1", "C2"
    axL.plot(r_pc, rho_cgs / RHO0_CGS, label=r"$\rho/\rho_0$", color=color_rho)
    axL.plot(r_pc, p_cgs / (RHO0_CGS * (1e8)**2), label=r"$P / (\rho_0 c_s^2)$ scaled", color=color_p)
    # Plot v_r/cs0 with cs0 in km/s
    cs0_kms = np.sqrt(5.0/3.0 * BOLTZMANN_CGS * 1e4 / PROTON_MASS_CGS) / 1e5
    axL.plot(r_pc, vr_kms / max(cs0_kms, 1e-3), label=fr"$v_r / c_{{s,0}}$ (×{1/cs0_kms:.1e})", color=color_v)
    axL.axvline(R_s_analytic_pc, color="k", linestyle="--", lw=1.2,
                label=fr"$R_s$ (analytic) = {R_s_analytic_pc:.1f} pc")
    axL.set_xlabel("r [pc]"); axL.set_ylabel("scaled")
    axL.set_xlim(0, final["Lx"] * UNIT_LENGTH / PARSEC_CGS / 2)
    axL.legend(frameon=False, fontsize=9)
    axL.tick_params(top=True, right=True, direction="in")
    axL.grid(True, linestyle=":", alpha=0.4)
    axL.set_title(f"Radial profile at t = {t_final_yr:.0f} yr")

    # === Right panel: R_shock(t) over snapshots vs analytic ===
    t_yrs = np.array([s["t"] * UNIT_TIME_S / SEC_PER_YEAR for s in snaps])
    R_sim = np.array([shock_radius_from_snap(s) for s in snaps])
    t_dense = np.linspace(max(t_yrs[0], 1.0), t_yrs[-1], 200)
    R_th = np.array([analytic_R_s(t) for t in t_dense])
    axR.loglog(t_yrs[1:], R_sim[1:], "o-", color="C0", label="LUNA peak-ρ location")
    axR.loglog(t_dense, R_th, "k--", label=r"$R_s(t) = \xi (E t^2/\rho_0)^{1/5}$, $\xi$=1.15")
    axR.set_xlabel("t [yr]"); axR.set_ylabel(r"$R_{shock}$ [pc]")
    axR.legend(frameon=False, fontsize=9)
    axR.tick_params(top=True, right=True, direction="in", which="both")
    axR.grid(True, which="both", linestyle=":", alpha=0.4)
    axR.set_title("Shock radius evolution")

    # === Right-most panel: density projection (column integrate along z) ===
    # Use the final snapshot. Convert column density to physical (g/cm^2).
    rho_proj = final["rho"].sum(axis=2) * final["dz"] * UNIT_DENSITY * UNIT_LENGTH
    L_pc = final["Lx"] * UNIT_LENGTH / PARSEC_CGS
    im = axS.imshow(rho_proj.T, origin="lower",
                    extent=(0, L_pc, 0, L_pc),
                    cmap="magma", aspect="equal")
    axS.set_xlabel("x [pc]"); axS.set_ylabel("y [pc]")
    axS.set_title(f"Column density at t = {t_final_yr:.0f} yr")
    axS.tick_params(top=True, right=True, direction="in")
    cb = fig.colorbar(im, ax=axS, fraction=0.046, pad=0.04)
    cb.set_label(r"$\int\rho\,dz$  [g cm$^{-2}$]")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/sedov3d.pdf"); fig.savefig("figures/sedov3d.png", dpi=200)
    print("Saved figures/sedov3d.{pdf,png}")
    print(f"R_s (sim, final) = {R_sim[-1]:.2f} pc")
    print(f"R_s (analytic, final) = {R_s_analytic_pc:.2f} pc")


if __name__ == "__main__":
    main()
