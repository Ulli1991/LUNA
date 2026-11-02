"""Turbulent self-gravitating cloud sphere (LUNA analogue of Torch's `turbsph`).

Parameters are cfg.problem (Params below); the Torch setup is in docs/TORCH.md.
"""
from dataclasses import dataclass

import numpy as np

from luna.ic_generation._grid import local_range


@dataclass(frozen=True)
class Params:
    """Turbulent-Sphere-3D's parameters, cfg.problem (mu is sink_star_mu, the seed is seed)."""
    mass_msun: float = 2000.0
    radius_pc: float = 2.0
    T_K: float = 15.0
    mach: float = 8.0
    turb_slope: float = -2.0
    amb_frac: float = 0.01
    profile: str = "tophat"               # tophat | gaussian
    edge_ratio: float = 1.0 / 3.0
    amb_T_K: float = 0.0
    amb_mu: float = 0.0
    amb_n_cm3: float = 0.0
    turb_spectrum: str = "power"          # power | kolmogorov
    turb_v0_kms: float = 0.0
    turb_kmin: int = 2
    turb_kmax: int = 32
    zero_drift: bool = True


PC_CM = 3.08568e18
MSUN_G = 1.98892e33
K_B = 1.380649e-16
M_H = 1.6726219e-24


def _shell_amp(k2, slope):
    """Per-mode amplitude |v_k| for a shell spectrum E(k) ~ k^slope; k2 = |k|^2, nonzero."""
    return np.sqrt(k2) ** ((slope - 2.0) / 2.0)


def _turbulent_field(N, slope, seed):
    """Unit-rms 3-component velocity field with shell spectrum E(k) ~ k^slope."""
    rng = np.random.default_rng(seed)
    kf = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(kf, kf, kf, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    k2[0, 0, 0] = 1.0
    amp = _shell_amp(k2, slope)
    amp[0, 0, 0] = 0.0                         # no DC / bulk motion
    v = np.empty((3, N, N, N))
    for c in range(3):
        fk = (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))) * amp
        v[c] = np.fft.ifftn(fk).real
    sigma = np.sqrt(np.mean(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
    return v / sigma                           # unit rms speed


def _kolmogorov_field(N, kmin, kmax, seed, slope=-5.0 / 3.0):
    """Torch's turb-sphere velocity on modes kmin <= |k| <= kmax (k = 2 pi / box), unnormalized."""
    rng = np.random.default_rng(seed)
    a = N / 2 - np.abs(np.arange(N, dtype=np.float64) - N / 2)
    k2 = a[:, None, None] ** 2 + a[None, :, None] ** 2 + a[None, None, :] ** 2
    amp = np.where((k2 >= kmin * kmin) & (k2 <= kmax * kmax),
                   _shell_amp(np.maximum(k2, 1.0), slope), 0.0)
    v = np.empty((3, N, N, N))
    for c in range(3):
        re = rng.random((N, N, N)) - 0.5
        im = rng.random((N, N, N)) - 0.5
        v[c] = np.fft.ifftn(np.sqrt(2.0) * (re + 1j * im) * amp).real
    return v


def _gaussian_rho_c(M, R, q):
    """Central density of rho_c exp(-(r/R)^2 ln(1/q)) holding M inside R."""
    from math import erf, exp, log, pi, sqrt
    s = R / sqrt(log(1.0 / q))
    U = R / s
    return M / (4.0 * pi * s ** 3 * (sqrt(pi) / 4.0 * erf(U) - 0.5 * U * exp(-U * U)))


def cloud_params(cfg):
    """The cloud's numbers in code units, from the config."""
    if not (cfg.Nx == cfg.Ny == cfg.Nz):
        raise ValueError("turbulent_sphere_3d assumes a cubic grid")
    u = cfg.units
    if not u.is_enabled:
        raise ValueError("turbulent_sphere_3d requires cfg.units enabled (pc/Msun/km·s)")
    q = cfg.problem
    M_cloud = float(q.mass_msun)
    R_pc = float(q.radius_pc)
    T = float(q.T_K)
    mach = float(q.mach)
    mu = float(cfg.sink_star_mu)
    amb_frac = float(q.amb_frac)

    # code density unit is M_sun / pc^3
    rho_cloud = M_cloud / (4.0 / 3.0 * np.pi * R_pc ** 3)
    # isothermal sound speed in code velocity
    cs = np.sqrt(K_B * T / (mu * M_H)) / u.UnitVelocity_in_cm_per_s
    dx = cfg.Lx / cfg.Nx
    p = dict(N=cfg.Nx, gamma=cfg.gamma, R_pc=R_pc, rho_cloud=rho_cloud,
             rho_amb=amb_frac * rho_cloud, cs=cs,
             sigma_v=mach * cs,
             dx=dx, xc=(np.arange(cfg.Nx) + 0.5) * dx, cxyz=0.5 * cfg.Lx,
             slope=float(q.turb_slope),
             seed=int(cfg.seed),
             zero_drift=bool(q.zero_drift),
             profile=str(q.profile),
             spectrum=str(q.turb_spectrum))
    if p["profile"] not in ("tophat", "gaussian"):
        raise ValueError(f"problem.profile={p['profile']!r}: 'tophat' or 'gaussian'")
    if p["spectrum"] not in ("power", "kolmogorov"):
        raise ValueError(f"problem.turb_spectrum={p['spectrum']!r}: 'power' or 'kolmogorov'")
    if p["profile"] == "gaussian":
        edge = float(q.edge_ratio)
        if not 0.0 < edge < 1.0:
            raise ValueError(f"problem.edge_ratio={edge}: rho(R)/rho(0) must be in (0, 1)")
        rho_c = _gaussian_rho_c(M_cloud, R_pc, edge)
        # ambient in pressure equilibrium with the cloud edge unless its density is given
        T_amb = float(q.amb_T_K) or T
        mu_amb = float(q.amb_mu) or mu
        n_amb = float(q.amb_n_cm3)
        cs_amb = np.sqrt(K_B * T_amb / (mu_amb * M_H)) / u.UnitVelocity_in_cm_per_s
        unit_rho = u.UnitMass_in_g / u.UnitLength_in_cm ** 3
        rho_amb = (n_amb * mu_amb * M_H / unit_rho if n_amb > 0.0
                   else edge * rho_c * cs * cs / (cs_amb * cs_amb))
        p.update(rho_c=rho_c, edge_ratio=edge, ln_q=np.log(1.0 / edge), rho_amb=rho_amb,
                 cs_amb=cs_amb)
    if p["spectrum"] == "kolmogorov":
        p.update(kmin=int(q.turb_kmin),
                 kmax=int(q.turb_kmax),
                 v0=float(q.turb_v0_kms) * 1.0e5
                 / u.UnitVelocity_in_cm_per_s)
        if p["v0"] <= 0.0:
            raise ValueError("problem.turb_spectrum='kolmogorov' needs problem.turb_v0_kms > 0")
    return p


def _rho_slab(p, X, Y, Z):
    """Density and cloud mask (r <= R for Gaussian, tanh edge for top hat) at cell centres."""
    c = p["cxyz"]
    r = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2)
    if p["profile"] == "gaussian":
        inside = (r <= p["R_pc"]).astype(np.float64)
        rho = np.where(inside > 0, p["rho_c"] * np.exp(-(r / p["R_pc"]) ** 2 * p["ln_q"]),
                       p["rho_amb"])
        return rho, inside
    # smooth top-hat sphere (tanh edge ~1 cell) so there's no zero-density cell
    inside = 0.5 * (1.0 - np.tanh((r - p["R_pc"]) / p["dx"]))
    return p["rho_amb"] + (p["rho_cloud"] - p["rho_amb"]) * inside, inside


def velocity_field(p):
    """Whole-box velocity before the cloud mask: unit rms (power law) or rms v0 (Kolmogorov)."""
    if p["spectrum"] == "power":
        v = _turbulent_field(p["N"], p["slope"], p["seed"])
        target = 1.0
        if not p["zero_drift"]:
            return v
    else:
        v = _kolmogorov_field(p["N"], p["kmin"], p["kmax"], p["seed"])
        target = p["v0"]
    xc = p["xc"]
    rho, inside = _rho_slab(p, xc[:, None, None], xc[None, :, None], xc[None, None, :])
    w = rho * inside
    W = np.sum(w)
    if p["zero_drift"]:
        for c in range(3):
            v[c] -= np.sum(w * v[c]) / W
    ms = np.sum(w * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)) / W
    v *= target / np.sqrt(ms)
    return v


def cloud_slab(p, v_unit, ranges):
    """U on the cell ranges ((x0, x1), (y0, y1), (z0, z1)) from the whole-box field `v_unit`."""
    (x0, x1), (y0, y1), (z0, z1) = ranges
    xc, cxyz = p["xc"], p["cxyz"]
    X, Y, Z = np.meshgrid(xc[x0:x1], xc[y0:y1], xc[z0:z1], indexing="ij")
    if p.get("profile") == "gaussian" or p.get("spectrum") == "kolmogorov":
        return _torch_slab(p, v_unit, ranges, X, Y, Z)
    r = np.sqrt((X - cxyz) ** 2 + (Y - cxyz) ** 2 + (Z - cxyz) ** 2)
    # smooth top-hat sphere (tanh edge ~1 cell) so there's no zero-density cell
    inside = 0.5 * (1.0 - np.tanh((r - p["R_pc"]) / p["dx"]))
    rho = p["rho_amb"] + (p["rho_cloud"] - p["rho_amb"]) * inside

    v = v_unit[:, x0:x1, y0:y1, z0:z1] * p["sigma_v"]
    v *= inside                                          # turbulence only in the cloud
    P = rho * p["cs"] * p["cs"]
    E = P / (p["gamma"] - 1.0) + 0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return np.stack([rho, rho * v[0], rho * v[1], rho * v[2], E])


def _torch_slab(p, v, ranges, X, Y, Z):
    """U for Torch's cloud: ambient at rest, each phase at its own temperature."""
    (x0, x1), (y0, y1), (z0, z1) = ranges
    rho, inside = _rho_slab(p, X, Y, Z)
    scale = 1.0 if p["spectrum"] == "kolmogorov" else p["sigma_v"]
    v = v[:, x0:x1, y0:y1, z0:z1] * (scale * inside)
    cs2 = np.where(inside > 0.5, p["cs"] ** 2, p.get("cs_amb", p["cs"]) ** 2)
    E = rho * cs2 / (p["gamma"] - 1.0) + 0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return np.stack([rho, rho * v[0], rho * v[1], rho * v[2], E])


def cloud_summary(U, p, dx, G):
    """Mass, densities, energies and virial numbers of the cloud (r <= R) in host U, code units."""
    rho = U[0]
    N = rho.shape[0]
    xc = (np.arange(N) + 0.5) * dx - p["cxyz"]
    r = np.sqrt(xc[:, None, None] ** 2 + xc[None, :, None] ** 2 + xc[None, None, :] ** 2)
    cl = r <= p["R_pc"]
    dV = dx ** 3
    m = rho[cl] * dV
    order = np.argsort(r[cl])
    rs, ms = r[cl][order], m[order]
    Menc = np.cumsum(ms)
    Epot = -G * float(np.sum(Menc * ms / np.maximum(rs, 0.5 * dx)))
    v2 = (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho ** 2
    Ekin = 0.5 * float(np.sum(rho[cl] * v2[cl]) * dV)
    M = float(np.sum(m))
    P = rho * p["cs"] ** 2
    Eth = 1.5 * float(np.sum(P[cl]) * dV)
    pbulk = [float(np.sum(U[k][cl]) * dV) for k in (1, 2, 3)]
    return dict(M_cloud=M, M_box=float(np.sum(rho) * dV), rho_max=float(rho.max()),
                rho_amb=float(rho[0, 0, 0]), Ekin=Ekin, Epot=Epot, Eth=Eth,
                Q=Ekin / abs(Epot), alpha_vir=2.0 * Ekin / abs(Epot),
                v_rms_mw=float(np.sqrt(2.0 * Ekin / M)),
                v_bulk=[q / M for q in pbulk])


def turbulent_sphere_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """The cloud U, whole or on this rank's cell ranges."""
    # Every rank builds the whole-box FFT field; at 1024^3 write the IC once (cfg.ic_dir).
    p = cloud_params(cfg)
    N = p["N"]
    ranges = tuple(local_range(N, sl) for sl in (x_slice, y_slice, z_slice))
    return cloud_slab(p, velocity_field(p), ranges)
