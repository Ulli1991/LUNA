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

from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

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

XI_0_3D = 1.15            # Sedov-Taylor constant for gamma=5/3 in 3D


def _list_res_outdirs():
    """All snapshots/sedov3d_arepo_Nx*/ directories, sorted by Nx ascending.

    Returns list of (Nx, dir) pairs.
    """
    import re
    candidates = []
    for d in glob.glob("snapshots/sedov3d_arepo_Nx*"):
        m = re.match(r"^snapshots/sedov3d_arepo_Nx(\d+)$", d)
        if m and os.path.isdir(d):
            candidates.append((int(m.group(1)), d))
    if not candidates:
        raise SystemExit("No sedov3d_arepo_Nx* snapshot dirs found.")
    candidates.sort(key=lambda x: x[0])
    return candidates


# 2-panel figure (R_s(t) + projection) uses the highest available Nx.
OUTDIR = _list_res_outdirs()[-1][1]
print(f"2-panel sedov snapshot dir (highest Nx): {OUTDIR}")


def load_snapshot(snapdir):
    """Stitch shards in a snapdir into global arrays (rho, vx, vy, vz, p, t)."""
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"]); Nz = int(a["Nz_global"])
        Lx = float(a["Lx"]); dx = float(a["dx"]); dy = float(a["dy"]); dz = float(a["dz"])
        t = float(a["time"])
    rho = np.empty((Nx, Ny, Nz), dtype=np.float32); vx = np.empty_like(rho)
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

    # Stream pass 1: compute R_s + t per snap without holding all in memory.
    # Each 1024^3 fp32 snap is ~21 GB, so holding 11 simultaneously OOMs even
    # on a full-node 240 GB job. Last snap is kept for the projection + radial
    # profile panels below.
    t_yrs_list = []
    R_sim_list = []
    final = None
    for i, d in enumerate(snaps_dirs):
        s = load_snapshot(d)
        t_yrs_list.append(s["t"] * UNIT_TIME_S / SEC_PER_YEAR)
        R_sim_list.append(shock_radius_from_snap(s))
        if i == len(snaps_dirs) - 1:
            final = s
        else:
            del s
    t_yrs = np.array(t_yrs_list)
    R_sim = np.array(R_sim_list)
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

    # === Figure 1: 4-panel radial-profile RESOLUTION STUDY ===
    # Overlay n, v_r, P, T profiles from every available Nx run at t = t_end.
    R_xlim = final["Lx"] * UNIT_LENGTH / PARSEC_CGS / 2

    figR, axes = plt.subplots(2, 2, figsize=(FIGSIZE[0]*2*0.6, FIGSIZE[1]*2*0.6))
    cmap = plt.get_cmap("magma")
    all_res = _list_res_outdirs()
    colors = [cmap(0.15 + 0.7 * i / max(1, len(all_res) - 1))
              for i in range(len(all_res))]

    for (Nx_i, dir_i), color in zip(all_res, colors):
        snap_files = sorted(glob.glob(os.path.join(dir_i, "snapdir_*")))
        if not snap_files:
            continue
        s_final = load_snapshot(snap_files[-1])
        ic_i = s_final["Nx"]/2 - 0.5
        cc_i = (ic_i, ic_i, ic_i)
        cx_i = s_final["Nx"]/2 * s_final["dx"]
        cy_i = s_final["Nx"]/2 * s_final["dy"]
        cz_i = s_final["Nx"]/2 * s_final["dz"]
        r_bins_i, rho_i = radial_profile(s_final["rho"], s_final["dx"], s_final["dy"], s_final["dz"], center=cc_i)
        _, p_i = radial_profile(s_final["p"], s_final["dx"], s_final["dy"], s_final["dz"], center=cc_i)
        X, Y, Z = np.meshgrid(
            (np.arange(s_final["Nx"]) + 0.5) * s_final["dx"],
            (np.arange(s_final["Nx"]) + 0.5) * s_final["dy"],
            (np.arange(s_final["Nx"]) + 0.5) * s_final["dz"],
            indexing="ij",
        )
        rvec = np.sqrt((X-cx_i)**2 + (Y-cy_i)**2 + (Z-cz_i)**2)
        vr_field = ((X-cx_i)*s_final["vx"] + (Y-cy_i)*s_final["vy"] + (Z-cz_i)*s_final["vz"]) / np.maximum(rvec, 1e-30)
        _, vr_i = radial_profile(vr_field, s_final["dx"], s_final["dy"], s_final["dz"], center=cc_i)

        r_pc_i = r_bins_i * UNIT_LENGTH / PARSEC_CGS
        n_cgs_i = rho_i * UNIT_DENSITY / PROTON_MASS_CGS
        p_cgs_i = p_i * UNIT_PRESSURE
        vr_kms_i = vr_i * UNIT_VELOCITY / 1e5
        T_K_i = p_cgs_i / np.maximum(n_cgs_i, 1e-30) / BOLTZMANN_CGS

        label = f"Nx={Nx_i}"
        axes[0, 0].plot(r_pc_i, n_cgs_i, color=color, label=label)
        axes[0, 1].plot(r_pc_i, vr_kms_i, color=color, label=label)
        axes[1, 0].plot(r_pc_i, p_cgs_i, color=color, label=label)
        axes[1, 1].plot(r_pc_i, T_K_i, color=color, label=label)

    panel_meta = [
        (axes[0, 0], "n [cm$^{-3}$]", "log"),
        (axes[0, 1], r"$v_r$ [km s$^{-1}$]", "linear"),
        (axes[1, 0], "P [erg cm$^{-3}$]", "log"),
        (axes[1, 1], "T [K]", "log"),
    ]
    for ax, ylabel, yscale in panel_meta:
        ax.axvline(R_s_analytic_pc, color="k", linestyle="--", lw=1.2,
                   label=fr"$R_s$ analytic = {R_s_analytic_pc:.1f} pc")
        ax.set_xlabel("r [pc]"); ax.set_ylabel(ylabel)
        ax.set_xlim(0, R_xlim)
        ax.set_yscale(yscale)
        ax.legend()
        decorate(ax)
    figR.suptitle(f"Sedov 3D radial-profile resolution study at t = {t_final_yr:.0f} yr",
                  fontsize=11)
    figR.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    figR.savefig("figures/sedov3d_radial.pdf")
    figR.savefig("figures/sedov3d_radial.png", dpi=200)

    # === Figure 2: R_s(t) + column density projection ===
    fig, (axR, axS) = plt.subplots(1, 2, figsize=(FIGSIZE[0]*2, FIGSIZE[1]))
    t_dense = np.linspace(max(t_yrs[0], 1.0), t_yrs[-1], 200)
    R_th = np.array([analytic_R_s(t) for t in t_dense])
    axR.loglog(t_yrs[1:], R_sim[1:], "o-", color="C0", label="LUNA peak-ρ location")
    axR.loglog(t_dense, R_th, "k--", label=r"$R_s(t) = \xi (E t^2/\rho_0)^{1/5}$, $\xi$=1.15")
    axR.set_xlabel("t [yr]"); axR.set_ylabel(r"$R_{shock}$ [pc]")
    axR.legend()
    decorate(axR)

    # === Right panel: density projection (column integrate along z) ===
    rho_proj = final["rho"].sum(axis=2) * final["dz"] * UNIT_DENSITY * UNIT_LENGTH
    L_pc = final["Lx"] * UNIT_LENGTH / PARSEC_CGS
    axS.imshow(rho_proj.T, origin="lower",
               extent=(0, L_pc, 0, L_pc),
               cmap="magma", aspect="equal")
    axS.set_xlabel("x [pc]"); axS.set_ylabel("y [pc]")
    axS.grid(False)

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    Nx_fig = final["Nx"]
    fig.savefig(f"figures/sedov3d_Nx{Nx_fig}.pdf"); fig.savefig(f"figures/sedov3d_Nx{Nx_fig}.png", dpi=200)
    print(f"Saved figures/sedov3d_Nx{Nx_fig}.{{pdf,png}}")
    print(f"R_s (sim, final) = {R_sim[-1]:.2f} pc")
    print(f"R_s (analytic, final) = {R_s_analytic_pc:.2f} pc")


if __name__ == "__main__":
    main()
