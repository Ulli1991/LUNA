"""Linear-wave eigenmode suite (as Stone et al. 2008; the Athena++ regression gates): sound both ways,
entropy and shear, grid-aligned in 1D and oblique in 2D, one period against the initial state."""
import functools

import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve

G, EPS = 5.0 / 3.0, 1e-6
RHO0, P0 = 1.0, 1.0 / G                  # c = 1
V0 = 0.5            # bulk flow carrying entropy and shear: subsonic (at v0 = c, u - c = 0 and HLL, HLLC
                    # both reduce to the upwind flux, so the contact resolution would go untested)
NS = {1: (32, 64, 128, 256, 512), 2: (32, 64, 128, 256)}
MODES = {1: ("sound_r", "sound_l", "entropy"), 2: ("sound_r", "sound_l", "entropy", "shear")}
SCHEMES = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1)}
NORMS = ("L1", "L2", "Linf")


def geometry(sdim, n):
    """(shape, box lengths, unit wave vector n, wavelength) of the 1D or oblique 2D setup."""
    if sdim == 1:
        return (n,), (1.0,), np.array([1.0]), 1.0
    k = 2.0 * np.pi * np.array([1.0, 2.0])                # box 1 x 1/2: one wavelength along each axis
    return (n, n // 2), (1.0, 0.5), k / np.linalg.norm(k), 2.0 * np.pi / np.linalg.norm(k)


def initial(sdim, n, mode, eps=EPS):
    """(U0, period): the background plus eps times the conserved right eigenvector, as exact cell averages."""
    shape, box, nhat, lam = geometry(sdim, n)
    h = [b / s for b, s in zip(box, shape)]
    kvec = 2.0 * np.pi * nhat / lam
    xs = np.meshgrid(*[(np.arange(s) + 0.5) * hh for s, hh in zip(shape, h)], indexing="ij")
    phase = sum(kk * x for kk, x in zip(kvec, xs))
    avg = np.prod([np.sinc(kk * hh / (2 * np.pi)) for kk, hh in zip(kvec, h)])   # cell average of sin
    s = avg * np.sin(phase)
    v0 = np.zeros(sdim) if mode.startswith("sound") else V0 * nhat
    drho, dv, dp = 0.0, np.zeros(sdim), 0.0
    if mode.startswith("sound"):
        drho, dv, dp = 1.0, (1.0 if mode == "sound_r" else -1.0) * nhat, 1.0  # c = rho0 = 1: du = +-c drho/rho0, dp = c^2 drho
    elif mode == "entropy":
        drho = 1.0
    else:                                                                   # shear: velocity along the wave front
        dv = np.array([-nhat[1], nhat[0]])
    rho = RHO0 + eps * drho * s
    mom = np.stack([RHO0 * v0[i] + eps * (RHO0 * dv[i] + v0[i] * drho) * s for i in range(sdim)])
    E = P0 / (G - 1.0) + 0.5 * RHO0 * v0 @ v0 + eps * (dp / (G - 1.0) + 0.5 * (v0 @ v0) * drho + RHO0 * (v0 @ dv)) * s
    return np.stack([rho, *mom, E]), lam / (1.0 if mode.startswith("sound") else V0)   # period: c = 1, |v0|


@functools.lru_cache(maxsize=None)
def run(sdim, n, mode, scheme, eps=EPS, frac=1.0):
    """(U0, U at frac periods) of the wave seeded with eps."""
    solver, limiter = SCHEMES[scheme]
    shape, box, _, _ = geometry(sdim, n)
    kw = dict(zip(("Nx", "Ny"), shape)) | dict(zip(("Lx", "Ly"), box))
    cfg = accuracy_cfg(test_problem="Soundwave", SPATIAL_DIMENSION=sdim, BOUNDARY=0, gamma=G,
                       riemann_solver=solver, LIMITER=limiter, **kw)
    U0, period = initial(sdim, n, mode, eps)
    return U0, np.asarray(evolve(cfg, U0, frac * period)[0])


@functools.lru_cache(maxsize=None)
def errors(sdim, n, mode, scheme, frac=1.0):
    """(L1, L2, Linf) over EPS of the part of U(period) - U0 odd in eps, (U+ - U0+ - U- + U0-) / 2, each the max
    over the conserved variables: no even orders in EPS (the even part, ~4e-6 EPS, was 17 % of 1D MC at 512)."""
    (U0p, Up), (U0m, Um) = run(sdim, n, mode, scheme, frac=frac), run(sdim, n, mode, scheme, eps=-EPS, frac=frac)
    e = ((Up - U0p) - (Um - U0m)).reshape(Up.shape[0], -1) / (2.0 * EPS)
    return (float(np.abs(e).mean(1).max()), float(np.sqrt((e * e).mean(1)).max()), float(np.abs(e).max()))


# {(dim, scheme, mode): bounds}: L1/L2/Linf of (U - U0)/EPS (max over conserved variables) ~1.3x measured,
# orders measured -0.1 .. +0.3 (CPU, FP64, default dt). Minmod clips extrema: L1 order ~1.85 in 1D, ~1.6
# oblique in 2D. The Athena++ gate (hydro_linwave.py) is err64 / err32 <= 0.325 (HLLE 0.33, 0.34), i.e.
# order >= ~1.6, plus absolute bounds; its comment says 0.25, the check does not.
BOUNDS = {
    (1, 'hllc-minmod', 'sound_r'): dict(L1={32: 0.03, 64: 0.0084, 128: 0.0023, 256: 0.00062, 512: 0.00017},
        L2={32: 0.038, 64: 0.013, 128: 0.004, 256: 0.0013, 512: 0.00041},
        Linf={32: 0.086, 64: 0.037, 128: 0.016, 256: 0.0063, 512: 0.0026},
        L1_order=(1.77, 2.17),
        L2_order=(1.54, 1.94),
        Linf_order=(1.17, 1.57)),
    # measured L1 0.0225, 0.00644, 0.00176, 0.000472, 0.000125; 1.875 | L2 0.029, 0.00944, 0.00304, 0.000973, 0.000311; 1.636 | Linf 0.0659, 0.0281, 0.0117, 0.00483, 0.00197; 1.267
    (1, 'hllc-minmod', 'sound_l'): dict(L1={32: 0.03, 64: 0.0084, 128: 0.0023, 256: 0.00062, 512: 0.00017},
        L2={32: 0.038, 64: 0.013, 128: 0.004, 256: 0.0013, 512: 0.00041},
        Linf={32: 0.086, 64: 0.037, 128: 0.016, 256: 0.0063, 512: 0.0026},
        L1_order=(1.77, 2.17),
        L2_order=(1.54, 1.94),
        Linf_order=(1.17, 1.57)),
    # measured L1 0.0225, 0.00644, 0.00176, 0.000472, 0.000125; 1.875 | L2 0.029, 0.00944, 0.00304, 0.000973, 0.000311; 1.636 | Linf 0.0659, 0.0281, 0.0117, 0.00483, 0.00197; 1.267
    (1, 'hllc-minmod', 'entropy'): dict(L1={32: 0.063, 64: 0.02, 128: 0.0055, 256: 0.0015, 512: 0.00041},
        L2={32: 0.075, 64: 0.026, 128: 0.0082, 256: 0.0027, 512: 0.00083},
        Linf={32: 0.15, 64: 0.062, 128: 0.026, 256: 0.011, 512: 0.0042},
        L1_order=(1.73, 2.13),
        L2_order=(1.53, 1.93),
        Linf_order=(1.17, 1.57)),
    # measured L1 0.0482, 0.0148, 0.00419, 0.00115, 0.000309; 1.826 | L2 0.0574, 0.0196, 0.0063, 0.00201, 0.000637; 1.627 | Linf 0.109, 0.047, 0.0195, 0.00798, 0.00323; 1.271
    (1, 'hll-minmod', 'sound_r'): dict(L1={32: 0.03, 64: 0.0084, 128: 0.0023, 256: 0.00062, 512: 0.00017},
        L2={32: 0.038, 64: 0.013, 128: 0.004, 256: 0.0013, 512: 0.00041},
        Linf={32: 0.086, 64: 0.037, 128: 0.016, 256: 0.0063, 512: 0.0026},
        L1_order=(1.77, 2.17),
        L2_order=(1.54, 1.94),
        Linf_order=(1.17, 1.57)),
    # measured L1 0.0225, 0.00644, 0.00176, 0.000472, 0.000125; 1.875 | L2 0.029, 0.00944, 0.00304, 0.000973, 0.000311; 1.636 | Linf 0.0659, 0.0281, 0.0117, 0.00483, 0.00197; 1.267
    (1, 'hll-minmod', 'sound_l'): dict(L1={32: 0.03, 64: 0.0084, 128: 0.0023, 256: 0.00062, 512: 0.00017},
        L2={32: 0.038, 64: 0.013, 128: 0.004, 256: 0.0013, 512: 0.00041},
        Linf={32: 0.086, 64: 0.037, 128: 0.016, 256: 0.0063, 512: 0.0026},
        L1_order=(1.77, 2.17),
        L2_order=(1.54, 1.94),
        Linf_order=(1.17, 1.57)),
    # measured L1 0.0225, 0.00644, 0.00176, 0.000472, 0.000125; 1.875 | L2 0.029, 0.00944, 0.00304, 0.000973, 0.000311; 1.636 | Linf 0.0659, 0.0281, 0.0117, 0.00483, 0.00197; 1.267
    (1, 'hll-minmod', 'entropy'): dict(L1={32: 0.1, 64: 0.044, 128: 0.013, 256: 0.0034, 512: 0.00093},
        L2={32: 0.13, 64: 0.053, 128: 0.017, 256: 0.0054, 512: 0.0018},
        Linf={32: 0.25, 64: 0.11, 128: 0.045, 256: 0.019, 512: 0.0075},
        L1_order=(1.62, 2.02),
        L2_order=(1.48, 1.88),
        Linf_order=(1.16, 1.56)),
    # measured L1 0.0764, 0.0336, 0.00941, 0.00258, 0.000709; 1.721 | L2 0.0991, 0.0402, 0.0129, 0.00413, 0.00131; 1.576 | Linf 0.187, 0.0818, 0.0345, 0.0142, 0.00574; 1.257
    (1, 'hllc-mc', 'sound_r'): dict(L1={32: 0.011, 64: 0.0026, 128: 0.00057, 256: 0.00014, 512: 3.2e-05},
        L2={32: 0.015, 64: 0.0043, 128: 0.0012, 256: 0.00034, 512: 9.5e-05},
        Linf={32: 0.037, 64: 0.014, 128: 0.005, 256: 0.0018, 512: 0.00066},
        L1_order=(2.00, 2.40),
        L2_order=(1.72, 2.12),
        Linf_order=(1.35, 1.75)),
    # measured L1 0.00805, 0.00194, 0.000435, 0.000104, 2.41e-05; 2.099 | L2 0.0114, 0.00324, 0.000918, 0.000257, 7.26e-05; 1.824 | Linf 0.0279, 0.0104, 0.0038, 0.00136, 0.000506; 1.450
    (1, 'hllc-mc', 'sound_l'): dict(L1={32: 0.011, 64: 0.0026, 128: 0.00057, 256: 0.00014, 512: 3.2e-05},
        L2={32: 0.015, 64: 0.0043, 128: 0.0012, 256: 0.00034, 512: 9.5e-05},
        Linf={32: 0.037, 64: 0.014, 128: 0.005, 256: 0.0018, 512: 0.00066},
        L1_order=(2.00, 2.40),
        L2_order=(1.72, 2.12),
        Linf_order=(1.35, 1.75)),
    # measured L1 0.00805, 0.00194, 0.000435, 0.000104, 2.41e-05; 2.099 | L2 0.0114, 0.00324, 0.000918, 0.000257, 7.26e-05; 1.824 | Linf 0.0279, 0.0104, 0.0038, 0.00136, 0.000506; 1.450
    (1, 'hllc-mc', 'entropy'): dict(L1={32: 0.017, 64: 0.0045, 128: 0.0011, 256: 0.00024, 512: 5.7e-05},
        L2={32: 0.023, 64: 0.0064, 128: 0.0018, 256: 0.00051, 512: 0.00015},
        Linf={32: 0.052, 64: 0.019, 128: 0.0065, 256: 0.0023, 512: 0.00085},
        L1_order=(1.96, 2.36),
        L2_order=(1.72, 2.12),
        Linf_order=(1.39, 1.79)),
    # measured L1 0.0129, 0.00339, 0.000797, 0.000184, 4.37e-05; 2.063 | L2 0.0171, 0.0049, 0.00137, 0.000385, 0.00011; 1.824 | Linf 0.0398, 0.0141, 0.00496, 0.00173, 0.000651; 1.489
    (2, 'hllc-minmod', 'sound_r'): dict(L1={32: 0.15, 64: 0.066, 128: 0.019, 256: 0.0053},
        L2={32: 0.2, 64: 0.079, 128: 0.026, 256: 0.0084},
        Linf={32: 0.37, 64: 0.17, 128: 0.069, 256: 0.029},
        L1_order=(1.52, 1.92),
        L2_order=(1.42, 1.82),
        Linf_order=(1.13, 1.53)),
    # measured L1 0.113, 0.0506, 0.0146, 0.00404; 1.623 | L2 0.148, 0.0605, 0.0198, 0.00639; 1.521 | Linf 0.277, 0.123, 0.0523, 0.0217; 1.227
    (2, 'hllc-minmod', 'sound_l'): dict(L1={32: 0.15, 64: 0.066, 128: 0.019, 256: 0.0053},
        L2={32: 0.2, 64: 0.079, 128: 0.026, 256: 0.0084},
        Linf={32: 0.37, 64: 0.17, 128: 0.069, 256: 0.029},
        L1_order=(1.52, 1.92),
        L2_order=(1.42, 1.82),
        Linf_order=(1.13, 1.53)),
    # measured L1 0.113, 0.0506, 0.0146, 0.00404; 1.623 | L2 0.148, 0.0605, 0.0198, 0.00639; 1.521 | Linf 0.277, 0.123, 0.0523, 0.0217; 1.227
    (2, 'hllc-minmod', 'entropy'): dict(L1={32: 0.14, 64: 0.062, 128: 0.019, 256: 0.0053},
        L2={32: 0.18, 64: 0.073, 128: 0.025, 256: 0.0079},
        Linf={32: 0.32, 64: 0.15, 128: 0.06, 256: 0.025},
        L1_order=(1.48, 1.88),
        L2_order=(1.41, 1.81),
        Linf_order=(1.13, 1.53)),
    # measured L1 0.105, 0.047, 0.0142, 0.00405; 1.581 | L2 0.137, 0.0556, 0.0188, 0.00606; 1.506 | Linf 0.246, 0.108, 0.046, 0.019; 1.230
    (2, 'hllc-minmod', 'shear'): dict(L1={32: 0.18, 64: 0.068, 128: 0.024, 256: 0.0068},
        L2={32: 0.22, 64: 0.083, 128: 0.031, 256: 0.0098},
        Linf={32: 0.37, 64: 0.17, 128: 0.07, 256: 0.029},
        L1_order=(1.46, 1.86),
        L2_order=(1.39, 1.79),
        Linf_order=(1.12, 1.52)),
    # measured L1 0.134, 0.0523, 0.0185, 0.0052; 1.555 | L2 0.168, 0.0635, 0.0234, 0.00752; 1.487 | Linf 0.282, 0.123, 0.0532, 0.0222; 1.222
    (2, 'hll-minmod', 'sound_r'): dict(L1={32: 0.18, 64: 0.079, 128: 0.024, 256: 0.0067},
        L2={32: 0.23, 64: 0.094, 128: 0.032, 256: 0.011},
        Linf={32: 0.42, 64: 0.19, 128: 0.08, 256: 0.034},
        L1_order=(1.48, 1.88),
        L2_order=(1.41, 1.81),
        Linf_order=(1.12, 1.52)),
    # measured L1 0.133, 0.0603, 0.0182, 0.00511; 1.585 | L2 0.177, 0.072, 0.0242, 0.00783; 1.505 | Linf 0.321, 0.143, 0.0611, 0.0254; 1.220
    (2, 'hll-minmod', 'sound_l'): dict(L1={32: 0.18, 64: 0.079, 128: 0.024, 256: 0.0067},
        L2={32: 0.23, 64: 0.094, 128: 0.032, 256: 0.011},
        Linf={32: 0.42, 64: 0.19, 128: 0.08, 256: 0.034},
        L1_order=(1.48, 1.88),
        L2_order=(1.41, 1.81),
        Linf_order=(1.12, 1.52)),
    # measured L1 0.133, 0.0603, 0.0182, 0.00511; 1.585 | L2 0.177, 0.072, 0.0242, 0.00783; 1.505 | Linf 0.321, 0.143, 0.0611, 0.0254; 1.220
    (2, 'hll-minmod', 'entropy'): dict(L1={32: 0.34, 64: 0.11, 128: 0.047, 256: 0.014},
        L2={32: 0.39, 64: 0.14, 128: 0.056, 256: 0.018},
        Linf={32: 0.6, 64: 0.26, 128: 0.12, 256: 0.048},
        L1_order=(1.42, 1.82),
        L2_order=(1.36, 1.76),
        Linf_order=(1.12, 1.52)),
    # measured L1 0.255, 0.0811, 0.0361, 0.0101; 1.516 | L2 0.298, 0.105, 0.0429, 0.0138; 1.459 | Linf 0.46, 0.2, 0.087, 0.0366; 1.216
    (2, 'hll-minmod', 'shear'): dict(L1={32: 0.3, 64: 0.095, 128: 0.043, 256: 0.012},
        L2={32: 0.35, 64: 0.13, 128: 0.05, 256: 0.017},
        Linf={32: 0.54, 64: 0.24, 128: 0.11, 256: 0.043},
        L1_order=(1.42, 1.82),
        L2_order=(1.36, 1.76),
        Linf_order=(1.12, 1.52)),
    # measured L1 0.228, 0.0726, 0.0323, 0.00901; 1.516 | L2 0.266, 0.0941, 0.0384, 0.0123; 1.459 | Linf 0.412, 0.179, 0.0779, 0.0327; 1.216
    (2, 'hllc-mc', 'sound_r'): dict(L1={32: 0.055, 64: 0.014, 128: 0.003, 256: 0.00064},
        L2={32: 0.074, 64: 0.021, 128: 0.0057, 256: 0.0016},
        Linf={32: 0.17, 64: 0.058, 128: 0.021, 256: 0.0071},
        L1_order=(2.05, 2.45),
        L2_order=(1.77, 2.17),
        Linf_order=(1.41, 1.81)),
    # measured L1 0.0423, 0.0103, 0.00231, 0.000486; 2.149 | L2 0.0566, 0.0158, 0.00437, 0.00117; 1.866 | Linf 0.126, 0.0446, 0.0156, 0.00544; 1.511
    (2, 'hllc-mc', 'sound_l'): dict(L1={32: 0.055, 64: 0.014, 128: 0.003, 256: 0.00064},
        L2={32: 0.074, 64: 0.021, 128: 0.0057, 256: 0.0016},
        Linf={32: 0.17, 64: 0.058, 128: 0.021, 256: 0.0071},
        L1_order=(2.05, 2.45),
        L2_order=(1.77, 2.17),
        Linf_order=(1.41, 1.81)),
    # measured L1 0.0423, 0.0103, 0.00231, 0.000486; 2.149 | L2 0.0566, 0.0158, 0.00437, 0.00117; 1.866 | Linf 0.126, 0.0446, 0.0156, 0.00544; 1.511
    (2, 'hllc-mc', 'entropy'): dict(L1={32: 0.058, 64: 0.018, 128: 0.0048, 256: 0.0012},
        L2={32: 0.073, 64: 0.023, 128: 0.0066, 256: 0.0019},
        Linf={32: 0.15, 64: 0.052, 128: 0.019, 256: 0.0066},
        L1_order=(1.78, 2.18),
        L2_order=(1.65, 2.05),
        Linf_order=(1.40, 1.80)),
    # measured L1 0.0444, 0.0138, 0.00362, 0.000899; 1.881 | L2 0.0555, 0.017, 0.00501, 0.00145; 1.753 | Linf 0.114, 0.04, 0.014, 0.00507; 1.498
    (2, 'hllc-mc', 'shear'): dict(L1={32: 0.04, 64: 0.011, 128: 0.0027, 256: 0.00063},
        L2={32: 0.059, 64: 0.015, 128: 0.0041, 256: 0.0012},
        Linf={32: 0.13, 64: 0.041, 128: 0.014, 256: 0.0045},
        L1_order=(1.89, 2.29),
        L2_order=(1.80, 2.20),
        Linf_order=(1.50, 1.90)),
    # measured L1 0.0302, 0.00804, 0.00204, 0.000483; 1.988 | L2 0.0447, 0.0113, 0.00312, 0.000853; 1.899 | Linf 0.0964, 0.0308, 0.0102, 0.00342; 1.605
}
LR_TOL = 1e-5   # sound_l is sound_r mirrored; with the even part removed the norms agree to round-off and
                # the scheme's mirror asymmetry (measured <= 6.6e-7)
CASES = [(sd, sc, mode) for sd in (1, 2) for sc in SCHEMES for mode in MODES[sd]]


@pytest.mark.parametrize("sdim,scheme,mode", CASES, ids=[f"{sd}d-{sc}-{mode}" for sd, sc, mode in CASES])
def test_linear_wave(sdim, scheme, mode):
    """L1, L2, Linf at every N and their orders; sound: left- and right-going errors equal."""
    b, ns = BOUNDS[sdim, scheme, mode], NS[sdim]
    errs = [errors(sdim, n, mode, scheme) for n in ns]
    for n, e in zip(ns, errs):
        for j, norm in enumerate(NORMS):
            check(f"{sdim}D {scheme} {mode} N={n} {norm}", e[j], b[norm][n])
    for j, norm in enumerate(NORMS):
        order = -np.polyfit(np.log(ns), np.log([e[j] for e in errs]), 1)[0]
        check(f"{sdim}D {scheme} {mode} {norm} order", order, b[f"{norm}_order"][0], lower=True)
        check(f"{sdim}D {scheme} {mode} {norm} order", order, b[f"{norm}_order"][1])
    if mode == "sound_l":
        for n in ns:
            r, l_ = errors(sdim, n, "sound_r", scheme), errors(sdim, n, "sound_l", scheme)
            check(f"{sdim}D {scheme} N={n} |sound_r - sound_l| / sound_r", max(abs(a - c) / a for a, c in zip(r, l_)), LR_TOL)


def test_bound_catches_a_short_run():
    """Teeth: stopped at 0.98 of a period the 1D sound wave misses its L1 bound by far."""
    e = errors(1, 256, "sound_r", "hllc-minmod", frac=0.98)[0]
    assert e > BOUNDS[1, "hllc-minmod", "sound_r"]["L1"][256], f"L1 {e:.3e} of a short run passes"
