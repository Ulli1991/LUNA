"""Rate and cooling tables of chemcool networks 5 and 17 (S. Glover's SGChem), built once in numpy.
Single-precision Fortran literals are f32(...); notes in docs/CHEMISTRY.md.
"""
from functools import lru_cache
import os

import numpy as np


def f32(x):
    """A Fortran default-REAL (4-byte) literal, widened to double."""
    return float(np.float32(x))


def _f32_sum(*terms):
    """Left-to-right sum evaluated in single precision (a folded REAL*4 constant)."""
    s = np.float32(terms[0])
    for t in terms[1:]:
        s = np.float32(s + np.float32(t))
    return float(s)


# --- constants exactly as the Fortran holds them (chemcool_consts.h, fs_data.h)

KBOLTZ = f32(1.38066e-16)       # BOLTZMANN macro, single-precision literal
EV = f32(1.60219e-12)           # ELECTRON_VOLT
MH = f32(1.6726e-24)            # PROTONMASS

GRAVITY = f32(6.672e-8)         # GRAVITY
PI = f32(3.14159265358979323846)  # PI macro, also a single literal
KB_FS = 1.38066e-16             # fs_data.h kb (double: 1.38066d-16)

ABHE = f32(0.1)                 # He:H by number (ABHE macro)
ABHE_NET17 = f32(0.079)         # ABHE for networks 1 and 17 (chemcool_consts.h)
TEMP_1EV = EV / KBOLTZ          # cheminmo.F: kb * temp_1ev = 1 eV

NMD = 10000                     # temperature table length
TMIN, TMAX = 1.0, 1.0e8
DTLOG = np.log10(TMAX) / (NMD - 1)

# Fine-structure atomic data (fs_data.h): Einstein A [s^-1], energies [erg]
OX_A10, OX_A20, OX_A21 = 8.865e-5, 1.275e-10, 1.772e-5
OX_E10, OX_E21 = 2.2771e2 * KB_FS, 9.886e1 * KB_FS
OX_E20 = OX_E10 + OX_E21
CII_A10, CII_E10 = 2.291e-6, 9.125e1 * KB_FS
SIII_A10, SIII_E10 = 2.17e-4, 4.1224e2 * KB_FS

H2_OPR = 2.4                    # coolinmo.F: ortho:para for the OI/CII H2 rates
F_ORTHO, F_PARA = H2_OPR / (1.0 + H2_OPR), 1.0 / (1.0 + H2_OPR)


_KERNEL_VIEWS = None    # name -> view of a table, while a fused kernel is traced (fused.py)
_SEEN = None            # name -> arrays, while fused.py records the tables a solve reads


def device_table(name, build):
    """build() (a table's JAX arrays), or the view of it that a fused kernel being traced reads."""
    if _KERNEL_VIEWS is not None:
        return _KERNEL_VIEWS[name]
    out = build()
    if _SEEN is not None:
        _SEEN[name] = out
    return out


def data_path(name):
    """Path of a chemistry data file in $LUNA_DATA/chemistry (not in the repository)."""
    root = os.environ.get("LUNA_DATA")
    if not root:
        raise RuntimeError("LUNA_DATA is not set; the chemistry data live in "
                           "$LUNA_DATA/chemistry (see docs/CHEMISTRY.md, Data files)")
    path = os.path.join(root, "chemistry", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing (see docs/CHEMISTRY.md, Data files)")
    return path


@lru_cache(maxsize=None)
def raw():
    """The DATA statements and data files of chemcool (tools/chem_net5_tables.py)."""
    with np.load(data_path("net5_raw.npz")) as z:
        return {k: z[k] for k in z.files}


def temptab():
    """coolinmo.F:212 -- T_i = 10**((i-1) dtlog), 1 K .. 1e8 K."""
    return 10.0 ** (np.arange(NMD) * DTLOG)


def _slopes(tab, x):
    """dt*tab of cheminmo.F:1733 / coolinmo.F:1620: forward difference per unit x, last entry repeated."""
    d = (tab[..., 1:] - tab[..., :-1]) * (1.0 / (x[1:] - x[:-1]))
    return np.concatenate([d, d[..., -1:]], axis=-1)


# --------------------------------------------------------------------------
# spline.F: Glover's cubic spline (uniform-parameter Hermite form)
# --------------------------------------------------------------------------
def spline_eval(positions, values, new_positions):
    """spline.F spline_eval: cubic spline through values at new_positions; 0 outside the data (spline.F:37)."""
    x = np.asarray(positions, float)
    y = np.asarray(values, float)
    n = y.size
    # spline_derivatives: tridiagonal system, forward elimination + back subst.
    diag = np.full(n, 4.0)
    diag[0] = diag[-1] = 2.0
    rhs = np.empty(n)
    rhs[0] = 3.0 * (y[1] - y[0])
    rhs[-1] = 3.0 * (y[-1] - y[-2])
    rhs[1:-1] = 3.0 * (y[2:] - y[:-2])
    for i in range(1, n):
        f = 1.0 / diag[i - 1]
        diag[i] = diag[i] - f * 1.0
        rhs[i] = rhs[i] - f * rhs[i - 1]
    d = np.empty(n)
    d[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        d[i] = (rhs[i] - (0.0 + 1.0 * d[i + 1])) / diag[i]
    a, b = y[:-1], d[:-1]
    c = 3.0 * (y[1:] - y[:-1]) - 2.0 * d[:-1] - d[1:]
    e = 2.0 * (y[:-1] - y[1:]) + d[:-1] + d[1:]
    pt = np.asarray(new_positions, float)
    j = np.clip(np.searchsorted(x, pt, side="right") - 1, 0, n - 2)
    dpt = (pt - x[j]) / (x[j + 1] - x[j])
    out = a[j] + b[j] * dpt + c[j] * dpt**2 + e[j] * dpt**3
    out = np.where(pt == x[-1], y[-1], out)
    return np.where((pt < x[0]) | (pt > x[-1]), 0.0, out)


# --------------------------------------------------------------------------
# cheminmo.F: chemical rate coefficients used by network 5
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def chem_table():
    """Rows of chtab network 5 reads, {row: (values at temptab, slope per K)}; ch7 per unit dust-to-gas."""
    T = temptab()
    temp2 = T * 1e-2
    tinv = 1.0 / T
    t4log = np.log10(T) - 4.0
    lnte = np.log(T / TEMP_1EV)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        ch = {}
        # ch1: H2 + H collisional dissociation, high-n limit (inverse of the
        # ABN02 three-body rate) -- cheminmo.F:196, iflag_3bh2a = 1
        ch[1] = np.where(T <= 3e2, 1.1e-9 * T**0.135 * np.exp(-5.2e4 / T),
                         (3.7e-8 / T**0.485) * np.exp(-5.2e4 / T))
        # ch3: low-n / high-n ratio for H2 + H (MS86 low-n rate) -- :242
        alpha = 1.0 + 5.48 * TEMP_1EV / T
        f1 = np.sqrt(T) * np.exp(-alpha)
        alpha2 = 1.0 + 5.48 * TEMP_1EV / 4.5e3
        f2 = np.sqrt(4.5e3) * np.exp(-alpha2)
        low = 1.2e-16 * f1 / f2
        ch[3] = np.where(ch[1] < 1e-100, 1e-100,
                         np.maximum(low / np.where(ch[1] < 1e-100, 1.0, ch[1]), 1e-100))
        # ch2: H2 + H2 dissociation, high-n (PSS83) -- :272, iflag_3bh2b = 1
        ch[2] = 6.5e-8 * np.exp(-5.2e4 / T) / T**f32(0.485)
        # ch4: low-n / high-n ratio for H2 + H2 (MKM98 low-n) -- :286
        low = (5.996e-30 * T**f32(4.1881) * np.exp(-54657.4 / T)
               / (1.0 + 6.761e-06 * T)**f32(5.6881))
        ch[4] = np.where(ch[2] < 1e-100, 1e-100,
                         np.maximum(low / np.where(ch[2] < 1e-100, 1.0, ch[2]), 1e-100))
        # ch5, ch6: inverse critical densities, H2-H (LS83 / 10) and H2-H2 -- :303
        ch[5] = 1.0 / 10.0 ** (3.0 - 0.416 * t4log - 0.327 * t4log * t4log)
        ch[6] = 1.0 / 10.0 ** (4.845 - 1.3 * t4log + 1.62 * t4log * t4log)
        # ch7: H2 formation on dust, HM79, per unit dust-to-gas ratio -- :319
        ch[7] = 1.0 * 3e-17 * temp2**0.5
        # ch11: collisional ionisation H + e (A97 fit to J87) -- :349
        ch[11] = np.exp(-32.71396786 + 13.5365560 * lnte - 5.73932875 * lnte**2
                        + 1.56315498 * lnte**3 - 0.28770560 * lnte**4
                        + 3.48255977e-2 * lnte**5 - 2.63197617e-3 * lnte**6
                        + 1.11954395e-4 * lnte**7 - 2.03914985e-6 * lnte**8)
        # ch13: case-B recombination of H+ (F92) -- :370
        ch[13] = (2.753e-14 * (315614.0 * tinv)**1.5
                  / (1.0 + (115188.0 * tinv)**0.407)**2.242)
        # ch25: H2 + e collisional dissociation (TT02a) -- :548
        ch[25] = 4.4886e-9 * T**0.109127 * np.exp(-1.01858e5 / T)
        # ch34, ch35: WD01 grain recombination fit for H+ (C3 T^C4, -C5-C6 lnT) -- :675
        ch[34] = 5.087e2 * T**1.586e-2
        ch[35] = -0.4723 - 1.102e-5 * np.log(T)
    return {k: (v, _slopes(v, T)) for k, v in ch.items()}


@lru_cache(maxsize=None)
def chem_table17():
    """The chtab rows network 17 reads beyond chem_table()'s, with GIZMO's flags (docs/CHEMISTRY.md)."""
    T = temptab()
    temp3 = T / 3e2
    tinv = 1.0 / T
    lgT = np.log10(T)
    lnT = np.log(T)
    te = T / TEMP_1EV
    lnte = np.log(te)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        ch = {}
        # ch8: He+ + H- -> He + H (SLD98) -- cheminmo.F:329
        ch[8] = 2.32e-7 * temp3**-0.52 * np.exp(np.minimum(T, 2e4) / 2.24e4)
        # ch9: He+ + H2 -> He + H + H+ (B84), frozen below 3 K -- :343
        ch[9] = 3.7e-14 * np.exp(-3.5e1 / np.maximum(T, 3.0))
        # ch10, ch17, ch18: He+ recombination (AP73 dielectronic, HS98 radiative,
        # optically thick), and H ionisation by its emission -- :383-424
        di = 1.9e-3 * np.exp(-473421.0 / T) * (1.0 + 0.3 * np.exp(-94684.0 / T)) / T**1.5
        rec_a = 1e-11 * (12.72 - 1.615 * lgT - 0.3162 * lgT**2 + 4.930e-2 * lgT**3) / np.sqrt(T)
        rec_b = 1e-11 * (11.19 - 1.676 * lgT - 0.2852 * lgT**2 + 4.433e-2 * lgT**3) / np.sqrt(T)
        gs = rec_a - rec_b
        ch[10] = di
        ch[17] = 0.677 * gs + rec_b + di
        ch[18] = 0.96 * ch[17] + 0.677 * gs
        # ch14, ch15: collisional ionisation of He and He+ (A97 fits to J87)
        ch[14] = np.exp(_poly([-44.09864886, 23.91596563, -10.7532302, 3.05803875,
                               -0.56851189, 6.79539123e-2, -5.00905610e-3, 2.06723616e-4,
                               -3.64916141e-6], lnte))
        ch[15] = np.exp(_poly([-68.71040990, 43.93347633, -18.4806699, 4.70162649,
                               -0.76924663, 8.113042e-2, -5.32402063e-3, 1.97570531e-4,
                               -3.16558106e-6], lnte))
        # ch20: case-B recombination of He++ (F92 scaled)
        ch[20] = (2.0 * 2.753e-14 * (4.0 * 315614.0 * tinv)**1.5
                  / (1.0 + (4.0 * 115188.0 * tinv)**0.407)**2.242)
        # ch21: H + e -> H- + photon (A97 fit to W79)
        ch[21] = np.where(T <= 6000, 10.0 ** (-17.845 + 0.762 * lgT + 0.1523 * lgT**2
                                              - 3.274e-2 * lgT**3),
                          10.0 ** (-16.4199 + 0.1998 * lgT**2 - 5.447e-3 * lgT**4
                                   + 4.0415e-5 * lgT**6))
        # ch22: H- + H -> H2 + e (GP98 fit to L91, iflag_ad = 1)
        ch[22] = np.where(T <= 3e2, 1.5e-9, 4.0e-9 / T**0.17)
        # ch23: H + H+ -> H2+ + photon (GP98 fit to RP76)
        ch[23] = 10.0 ** (-19.38 - 1.523 * lgT + 1.118 * lgT**2 - 0.1269 * lgT**3)
        # ch24: H2 + H+ -> H2+ + H, low-density limit (SKHS04 fit; exp tail above 3e4 K)
        ch[24] = np.where(T <= 3e4, _poly([-3.3232183e-7, 3.3735382e-7, -1.4491368e-7,
                                           3.4172805e-8, -4.7813720e-9, 3.9731542e-10,
                                           -1.8171411e-11, 3.5311932e-13], lnT)
                          * np.exp(-2.1237150e4 / T), 3.7289e-10 * np.exp(-2.1237150e4 / T))
        # ch26: H- + e -> H + 2e (A97 fit to J87)
        ch[26] = np.exp(_poly([-18.01849334, 2.3608522, -0.28274430, 1.62331664e-2,
                               -3.36501203e-2, 1.17832978e-2, -1.65619470e-3, 1.06827520e-4,
                               -2.63128581e-6], lnte))
        # ch27: H- + H -> 2H + e (A97 fit to J87)
        ch[27] = np.where(te <= 1e-1, 2.5634e-9 * te**1.78186, np.exp(_poly(
            [-20.37260896, 1.13944933, -0.14210135, 8.4644554e-3, -1.4327641e-3, 2.0122503e-4,
             8.6639632e-5, -2.5850097e-5, 2.4555012e-6, -8.0683825e-8], lnte)))
        # ch28: mutual neutralisation H- + H+ -> 2H (GP98 / MAP70, iflag_mn = 1)
        ch[28] = 5.7e-6 / np.sqrt(T) + 6.3e-8 - 9.2e-11 * np.sqrt(T) + 4.4e-13 * T
        # ch29: H- + H+ -> H2+ + e (GP98 fit to P78)
        ch[29] = np.where(T <= 8e3, 6.9e-9 / T**0.35, 9.6e-7 / T**0.9)
        # ch30: H2+ dissociative recombination, low-density limit (A97 fit to S94)
        ch[30] = np.where(T <= 6.17e2, 1e-8, 1.32e-6 / T**0.76)
        # ch31: H- + He -> H + He + e (H82)
        ch[31] = 4.1e-17 * T**2 * np.exp(-19870.0 / T)
        # ch32, ch33: charge transfer He+ + H -> He + H+ (SLD98), He + H+ -> He+ + H
        ch[32] = 1.2e-15 * (T / 3e2)**0.25
        ch[33] = np.where(T <= 1e4, 1.26e-9 * np.exp(-1.275e5 / T) / T**0.75,
                          4e-37 * T**4.74)
        # ch92: CO + He+ -> C+ + O + He (PDAN89), capped at 1.6e-9
        ch[92] = np.minimum(1.4e-9 / np.sqrt(temp3), 1.6e-9)
        # ch98, ch99: D + H+ <-> D+ + H charge transfer (S02, kept positive)
        ch[98] = np.where(T <= 2.7, 2e-10 * T**0.402 * np.exp(-37.1 / T),
                          np.where(T <= 2e5, 2e-10 * T**0.402 * np.exp(-37.1 / T)
                                   - 3.31e-17 * T**1.48, 3.44e-10 * T**0.35))
        ch[99] = 2.06e-10 * T**0.396 * np.exp(-33.0 / T) + 2.03e-9 * T**-0.332
        # ch100, ch101: D+ + H2 <-> HD + H+ (G82 via GP02)
        ch[100] = (0.417 + 0.846 * lgT - 0.137 * lgT**2) * 1e-9
        ch[101] = 1.1e-9 * np.exp(-4.88e2 / T)
        # ch102: D + H2 -> H + HD (Glover's fit to MPS03; 5.88886 is a REAL*4)
        ch[102] = np.where(T <= 2e3, 10.0 ** (-5.64737e1 + f32(5.88886) * lgT + 7.19692 * lgT**2
                                              + 2.25069 * lgT**3 - 2.16903 * lgT**4
                                              + 3.17887e-1 * lgT**5),
                           3.17e-10 * np.exp(-5.207e3 / T))
        # ch103: HD + H -> H2 + D (S59 via GP02, flattened below 200 K)
        ch[103] = np.where(T <= 2e2, 5.25e-11 * np.exp(-4.43e3 / T),
                           5.25e-11 * np.exp((-4.43e3 / T) + (1.739e5 / T**2)))
        # ch108-110: three-body H2 formation, third body H (ABN02), H2 (PSS83), He (WK75)
        ch[108] = np.where(T <= 3e2, 1.3e-32 / temp3**0.38, 1.3e-32 / temp3)
        ch[109] = 6.875e-30 / T
        ch[110] = 6.9e-32 * T**-0.4
        # ch126: H2 + e -> H + H- (GP98); ch127: H2+ + H- -> H2 + H (DL87)
        ch[126] = 2.7e-8 * T**-1.27 * np.exp(-4.3e4 / T)
        ch[127] = 1.4e-7 / np.sqrt(temp3)
        # ch137-139: H2 + He collisional dissociation, low-n, LTE, 1/n_crit (DRCM87),
        # frozen above 1e4 K; 3.801, 27.029, 0.115 are REAL*4
        tfix = np.minimum(T, 1e4)
        ch[138] = 6.6e-10 * T**f32(0.115) * np.exp(-5.2e4 / T)
        ch[137] = 10.0 ** (f32(3.801) * np.log10(tfix) - f32(27.029) - 2.9487e4 / tfix)
        ch[139] = 1.0 / 10.0 ** (5.0792 * (1.0 - 1.23e-5 * (tfix - 2e3)))
        # ch144: H + H -> H+ + e + H (GVZ87); ch158: H + He -> H+ + e + He (L91)
        ch[144] = 1.2e-17 * T**f32(1.2) * np.exp(-1.578e5 / T)
        ch[158] = 1.75e-17 * T**f32(1.3) * np.exp(-1.578e5 / T)
        # ch153-156: HeH+ chemistry (G94, SLD98)
        ch[153] = 3.0e-8 * (T / 3e2)**-0.47
        ch[154] = 3.0e-10 * np.exp(-6.72e3 / T)
        ch[155] = 8.0e-20 * (T / 3e2)**-0.24 * np.exp(-T / 4000)
        ch[156] = 1.04e-9 * (T / 3e2)**0.13 * np.exp(-T / 3.31e4)
        # ch157: H2+ dissociative recombination, LTE limit (Coppola+ 2011), frozen > 1e4 K
        tf = np.where(T > 1e4, 1e4, T)
        ch[157] = 1e6 * (4.2278e-14 - 2.3088e-17 * tf + 7.3428e-21 * tf**2 - 7.5474e-25 * tf**3
                         + 3.3468e-29 * tf**4 - 5.5280e-34 * tf**5)
        # ch159, ch160: H2+ + H collisional dissociation, low-n and LTE limits
        ch[159] = 1.54e-12 * T**f32(0.45) * np.exp(-3.1e4 / T)
        ch[160] = np.exp(f32(-32.912) + 6.9498e-5 * T - 3.3248e4 / T - 4.08e-9 * T**2)
        # ch161-172: H2 + H collision-induced (CID) and dissociative-tunnelling (DT)
        # rates of MSM96: log10 high-n and low-n rates and n_crit, 450 K..45000 K
        tm = np.where(T > 4.5e4, 4.5e4, np.where(T < 4.5e2, 4.5e2, T))
        lm = np.log10(tm)
        ch[161] = (-178.4239 - 68.42243 * lm + 43.20243 * lm**2 - 4.633167 * lm**3
                   + 69.70086 * np.log10(1.0 + 4.087038e4 / tm))
        ch[162] = -2.370570e4 / tm
        ch[163] = (1.288953e2 - 5.391334e1 * lm + 5.315517 * lm**2
                   - 1.973427e1 * np.log10(1.0 + 1.678095e4 / tm))
        ch[164] = -2.578611e4 / tm
        ch[165] = 10.0 ** (1.482123e1 - 4.890915 * lm + 4.749030e-1 * lm**2 - 1.338283e2 / tm)
        ch[166] = (8.227443e-1 + 5.864073e-1 * np.exp(-tm / 1.85e3)
                   - 2.056313 * np.exp(-tm / 4.4e2))
        ch[167] = (-142.7664 + 42.70741 * lm - 2.027365 * lm**2 - 2.582097e-1 * lm**3
                   + 2.136094e1 * np.log10(1.0 + 2.753531e4 / tm))
        ch[168] = -2.146779e4 / tm
        ch[169] = (60.34928 - 27.43096 * lm + 2.676150 * lm**2
                   - 11.28215 * np.log10(1.0 + 1.425455e4 / tm))
        ch[170] = -23125.2 / tm
        ch[171] = 10.0 ** (9.305564 - 2.464009 * lm + 0.1985955 * lm**2 + 743.0600 / tm)
        ch[172] = (7.502286e-1 + 2.358848e-1 * np.exp(-tm / 1.85e3)
                   + 2.937507 * np.exp(-tm / 4e2))
        # ch173: H2 + H+ -> H2+ + H, LTE limit (Coppola+ 2011)
        ch[173] = 1e6 * np.exp(f32(-33.081) + 6.3173e-5 * T - 2.3478e4 / T - 1.8691e-9 * T**2)
    return {k: (v, _slopes(v, T)) for k, v in sorted(ch.items())}


@lru_cache(maxsize=None)
def chem_table17_fixed():
    """The chtab rows the fixed network 17 changes: ch100 clamped at 0 (negative above ~4.3e6 K)."""
    v = np.maximum(0.0, chem_table17()[100][0])
    return {100: (v, _slopes(v, temptab()))}


# --------------------------------------------------------------------------
# coolinmo.F: cooling coefficients used by network 5
# --------------------------------------------------------------------------

# GCC's powi_table: gfortran turns x**n (integer constant n) into the
# multiplication chain x**n = x**a * x**(n - a), a = _POWI[n]
_POWI = [0, 1, 1, 2, 2, 3, 3, 4, 4, 6, 5, 6, 6, 10, 7, 9, 8]


def ipow(x, n):
    """x**n for a small integer constant n, multiplied as gfortran -O2 does."""
    if n == 1:
        return x
    a = _POWI[n]
    return ipow(x, a) * ipow(x, n - a)


def _poly(coeffs, x):
    """c0 + c1 x + c2 x**2 + ..., evaluated left to right like the Fortran."""
    out = coeffs[0] + coeffs[1] * x
    for k, c in enumerate(coeffs[2:], start=2):
        out = out + c * ipow(x, k)
    return out


@lru_cache(maxsize=None)
def cool_table():
    """Rows of cltab network 5 reads, keyed by Fortran row: (values, slope per K)."""
    T = temptab()
    temp2, temp3 = T * 1e-2, T * 1e-3
    tau = T * 1e-3 + 1.0
    tinv = 1.0 / T
    tinth, tintq = tinv ** (1.0 / 3.0), tinv**0.75
    tsqrt = np.sqrt(T)
    tisqt = 1.0 / tsqrt
    L3 = np.log10(temp3)
    h2e20, h2e31 = 508.95, 852.5
    cl = {}
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        # cl1-3: H2 rovibrational cooling (LPF99); T < 100 K from the J=2-0, 3-1
        # lines, 100 K..1e4 K from the spline below, > 1e4 K frozen -- :246
        e20, e31 = np.exp(-h2e20 / T), np.exp(-h2e31 / T)
        h2n2 = 0.25 * (5.0 * e20 / (1.0 + 5.0 * e20))
        h2n3 = 0.75 * ((7.0 / 3.0) * e31 / (1.0 + (7.0 / 3.0) * e31))
        lte_low = np.maximum(2.94e-11 * h2e20 * h2n2 * KBOLTZ
                             + 4.76e-10 * h2e31 * h2n3 * KBOLTZ, 1e-60)
        q02 = 10.0 ** (-9.121 - (3.983 / tau) - (1.844e-1 / tau**2)) * 5.0 * e20
        q13 = 10.0 ** (-9.493 - (4.435 / tau) + (2.707e-1 / tau**2)) * (7.0 / 3.0) * e31
        h_low = (F_PARA * q02 * h2e20 + F_ORTHO * q13 * h2e31) * KBOLTZ
        q02 = 10.0 ** (-9.946 - (2.688 / tau) + (2.020e-1 / tau**2)) * 5.0 * e20
        q13 = 10.0 ** (-9.785 - (3.024 / tau) + (2.930e-1 / tau**2)) * (7.0 / 3.0) * e31
        h2_low = (F_PARA * q02 * h2e20 + F_ORTHO * q13 * h2e31) * KBOLTZ
        r = raw()
        h2_temp = 10.0 ** (2.0 + 5e-2 * np.arange(41))
        mid = (T >= 1e2) & (T <= 1e4)
        for row, low_T, high, data in ((1, lte_low, -18.253, "h2_lte"),
                                       (2, h_low, -21.943, "h2_h_rate"),
                                       (3, h2_low, -22.758, "h2_h2_rate")):
            v = np.where(T < 5.0, 1e-60, np.where(T < 1e2, low_T, 10.0**high))
            cl[row] = np.where(mid, 10.0 ** spline_eval(h2_temp, r[data], T), v)
        # cl4: H2-H low-density rate (WF07) -- :307
        c_a = [-16.818342, 37.383713, 58.145166, 48.656103, 20.159831, 3.8479610]
        c_b = [-24.311209, 3.5692468, -11.332860, -27.850082, -21.328264, -4.2519023]
        c_c = [-24.311209, 4.6450521, -3.7209846, 5.9369081, -5.5108047, 1.5538288]
        hi = c_c[0] + c_c[1] + c_c[2] + c_c[3] + c_c[4] + c_c[5]
        cl[4] = np.where(T < 1e1, 0.0, np.where(
            T < 1e2, 10.0 ** _poly(c_a, L3), np.where(
                T < 1e3, 10.0 ** _poly(c_b, L3), np.where(
                    T < 1e4, 10.0 ** _poly(c_c, L3), 10.0**hi))))
        # cl5: H2-H2 (LPF99), cl64: H2-He (FRZ98, BFD99), cl65: H2-H+ (G90, K02)
        for row, c in ((5, [-23.962112, 2.09433740, -0.77151436, 0.43693353,
                            -0.14913216, -0.033638326]),
                       (64, [-23.689237, 2.1892372, -0.81520438, 0.29036281,
                             -0.16596184, 0.19191375]),
                       (65, [-21.716699, 1.3865783, -0.37915285, 0.11453688,
                             -0.23214154, 0.058538864])):
            hi = c[0] + c[1] + c[2] + c[3] + c[4] + c[5]
            cl[row] = np.where(T < 1e1, 0.0,
                               np.where(T < 1e4, 10.0 ** _poly(c, L3), 10.0**hi))
        # cl66: H2-e (DRD83); the 200 K..1e4 K fit is written in single literals
        c_lo = [-34.286155, -48.537163, -77.121176, -51.352459, -15.169160, -0.98120322]
        c_hi = [-22.190316, 1.5728955, -0.21335100, 0.96149759, -0.91023195, 0.13749749]
        cl[66] = np.where(T < 1e1, 0.0, np.where(
            T < 2e2, 10.0 ** _poly(c_lo, L3), np.where(
                T < 1e4, 10.0 ** _poly([f32(c) for c in c_hi], L3),
                10.0 ** _f32_sum(*c_hi))))
        # cl12, cl13: CO vibrational cooling, low-density (NK93 for H2; BYD02 for H)
        ctinv = np.exp(-3.080e3 * tinv)
        cl[12] = 1.83e-26 * T * ctinv * np.exp(-68.0 * tinth)
        cl[13] = np.where(T < 7e2, 1.049e-35 * T**f32(3.98) * ctinv,
                          1.062e-29 * T**f32(1.87) * ctinv)
        # cl14: gas-grain energy transfer (HM89 eq 2.15) -- :545
        gg = 1.0 - f32(0.8) * np.exp(-75.0 * tinv)
        cl[14] = 3.8e-33 * tsqrt * gg * np.exp(-T / 3e4)
        # cl16, cl17: photoelectric heating (BT94 as in Wolfire+03) -- :555
        cl[16] = np.sqrt(T) / 1.0          # / phi_pah, phi_pah = 1
        cl[17] = 3.7e-2 * (T / 1e4) ** 0.7
        # cl18-29: OI fine-structure de-excitation, H (AKD07), H2, e-, H+
        cl.update(_oi_rates(T, tintq, tisqt))
        # cl45-48: CII fine structure (HM89/K86 H; WBV96 H2; WB02 e-; f_LTE)
        cl[45] = np.where(T <= 2e3, 8e-10 * temp2**0.07, 3.113619e-10 * temp2**0.385)
        cl[46] = np.where(T < 250.0,
                          (4.7e-10 + 4.6e-13 * T) * F_ORTHO + (2.5e-10 * T**f32(0.12)) * F_PARA,
                          (5.85e-10 * T**f32(0.07)) * F_ORTHO
                          + (4.85e-10 * T**f32(0.07)) * F_PARA)
        cl[47] = np.where(T <= 2e3, 3.86e-7 / np.sqrt(temp2), 2.426206e-7 / temp2**0.345)
        cl[48] = 2.0 * np.exp(-91.25 / T)
        # cl49-51: SiII fine structure (R90 H, DK91 e-, f_LTE)
        cl[49] = 4.95e-10 * temp2**f32(0.24)
        cl[50] = 1.2e-6 / np.sqrt(temp2)
        cl[51] = 2.0 * np.exp(-412.24 / T)
        # cl52: HI excitation (Lyman alpha) cooling (C92) -- :1003
        cl[52] = np.where(T < 1e3, 0.0,
                          7.5e-19 * np.exp(-1.18348e5 / T) / (1.0 + np.sqrt(T / 1e5)))
        # cl58: H2 collision-induced emission (RA04) -- :1152
        cl[58] = 2.289e-49 * T**4
    return {k: (v, _slopes(v, T)) for k, v in cl.items()}


@lru_cache(maxsize=None)
def cool_table_hd():
    """cltab rows 53-57: the HD cooling fit (LNA05) in powers of log10 n_HI."""
    T = temptab()
    thd = np.where(T > 2e4, np.log10(2e4), np.log10(T))
    s = f32
    cl = {53: _poly([s(-42.57688), s(21.93385), s(-10.19097), s(2.19906), s(-0.17334)], thd),
          54: _poly([s(0.92433), s(0.77952), s(-0.54263), s(0.11711), -8.35e-3], thd),
          55: _poly([s(0.54962), s(-1.06447), s(0.62343), s(-0.13768), 1.06e-2], thd),
          56: _poly([-7.676e-2, 0.11864, -7.366e-2, 1.759e-2, -1.482e-3], thd),
          57: _poly([2.75e-3, -3.66e-3, 2.514e-3, -6.6631e-4, 6.1926e-5], thd)}
    return {k: (v, _slopes(v, T)) for k, v in cl.items()}


def _oi_rates(T, tintq, tisqt):
    """coolinmo.F:590-702: OI 1-0, 2-0, 2-1 collisional de-excitation rates."""
    e10 = OX_E10 / KBOLTZ
    e20 = OX_E20 / KBOLTZ
    e21 = OX_E21 / KBOLTZ
    c18 = [f32(c) for c in (4.581, -156.118, 2679.979, -78996.962, 1308323.468,
                            -13011761.861, 71010784.971, -162826621.855)]
    c19 = [f32(c) for c in (3.297, -168.382, 1844.099, -68362.889, 1376864.737,
                            -17964610.169, 134374927.808, -430107587.886)]
    c20 = [f32(c) for c in (3.437, 17.443, -618.761, 3757.156, -12736.468,
                            22785.266, -22759.228, 12668.261)]
    x5 = 1.0 / 5.0**0.75
    out = {}
    # H: AKD07 below 1000 K, power law above; T < 5 K frozen at 5 K
    out[18] = np.where(T < 5.0, (5e-11 / 3.0) * np.exp(_poly(c18, x5)) * np.exp(e10 / 5.0),
                       np.where(T < 1e3, (5e-11 / 3.0) * np.exp(_poly(c18, tintq))
                                * np.exp(e10 / T), 6.81e-11 * T**0.376))
    out[19] = np.where(T < 5.0, 5e-11 * np.exp(_poly(c19, x5)) * np.exp(e20 / 5.0),
                       np.where(T < 1e3, 5e-11 * np.exp(_poly(c19, tintq)) * np.exp(e20 / T),
                                6.34e-11 * T**0.36))
    out[20] = np.where(T < 5e1, 2.62e-12 * T**0.74,
                       np.where(T < 1e3, 3e-11 * np.exp(_poly(c20, tisqt)) * np.exp(e21 / T),
                                3.61e-10 * T**0.158))
    # H2 (Flower, priv. comm.), ortho and para
    s = f32
    out[21] = F_ORTHO * 2.70e-11 * T**s(0.362) + F_PARA * 3.46e-11 * T**s(0.316)
    out[22] = F_ORTHO * 5.49e-11 * T**s(0.317) + F_PARA * 7.07e-11 * T**s(0.268)
    out[23] = F_ORTHO * 2.74e-14 * T**s(1.060) + F_PARA * 3.33e-15 * T**s(1.360)
    # electrons (BBT98 fits)
    out[24] = 5.12e-10 * T**s(-0.075)
    out[25] = 4.863e-10 * T**s(-0.026)
    out[26] = 1.082e-14 * T**s(0.926)
    # protons (P90)
    out[27] = np.where(T < 194, 6.38e-11 * T**s(0.40),
                       np.where(T < 3686, 7.75e-12 * T**s(0.80), 2.65e-10 * T**s(0.37)))
    out[28] = np.where(T < 511, 6.10e-13 * T**s(1.10),
                       np.where(T < 7510, 2.12e-12 * T**s(0.90), 4.49e-10 * T**s(0.30)))
    out[29] = np.where(T < 2090, 2.029e-11 * T**s(0.56), 3.434e-10 * T**s(0.19))
    return out


# --------------------------------------------------------------------------
# WSS_CIE_COOL: hot-gas metal cooling (wss_z_collis.F), He and free-free rows
# --------------------------------------------------------------------------
WSS_ELEMENTS = ("C", "N", "O", "Ne", "Mg", "Si", "S", "Ca", "Fe")
WSS_NODES = 352
WSS_LOGT_RANGE = f32(6.98185)    # log10 T from 2 to 8.98185, as get_coolrates_wssCIE assumes


@lru_cache(maxsize=None)
def cie_raw():
    """wss_z_collis.txt (T, nine metal columns) and the HeI/HeII CIE fractions."""
    with np.load(data_path("cie_tables.npz")) as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def wss_cie_table():
    """read_wss_collis_table: log10 |rate x max(exp(-(1e4/T)^2), 1e-20)|, shape (352, 9)."""
    # per element at solar abundance [erg cm^3 s^-1 per n_H^2] (Wiersma, Schaye & Smith
    # 2009); the factor suppresses T < 1e4 K, abs() turns the negative (net heating)
    # entries into cooling: both as in the Fortran (docs/CHEMISTRY.md)
    r = cie_raw()
    temp = r["wss_temp"]
    fac = np.exp(-((1e4 / temp) ** 2))
    fac = np.where(fac < f32(1e-20), f32(1e-20), fac)
    with np.errstate(divide="ignore"):
        return np.log10(np.abs(r["wss_rates"] * fac[:, None]))


@lru_cache(maxsize=None)
def cool_table_cie():
    """cltab rows 59-63 (HeI, HeII excitation; free-free), which WSS_CIE_COOL leaves on."""
    T = temptab()
    cl = {}
    with np.errstate(over="ignore", under="ignore"):
        # cl59: HeI excitation from n = 1 (Glover's fit to BBFT00) -- coolinmo.F:1041
        cl[59] = np.where(T < 2e3, 0.0, 1.1e-19 * T**0.082 * np.exp(-2.3e5 / T))
        # cl60: HeI excitation from the metastable 2^3S state (C92 / B81)
        cl[60] = np.where(T < 1e2, 0.0, 9.1e-27 * np.exp(-1.3179e4 / T)
                          / (T**0.1687 * (1.0 + np.sqrt(T / 1e5))))
        # cl61: HeII excitation (C92 / B81)
        cl[61] = np.where(T < 4e3, 0.0, 5.54e-17 * np.exp(-4.73638e5 / T)
                          / (T**0.397 * (1.0 + np.sqrt(T / 1e5))))
        # cl62, cl63: bremsstrahlung of singly / doubly charged ions (SK87, S78);
        # the Gaunt-factor fits are single-precision literals
        lt, lt4 = np.log10(T), np.log10(T / 4.0)
        gff = np.where(T < 3.2e5, f32(0.79464) + f32(0.1243) * lt,
                       f32(2.13164) - f32(0.1240) * lt)
        cl[62] = 1.426e-27 * np.sqrt(T) * gff
        gff = np.where(T < 1.28e6, f32(0.79464) + f32(0.1243) * lt4,
                       f32(2.13164) - f32(0.1240) * lt4)
        cl[63] = 4.0 * 1.426e-27 * np.sqrt(T) * gff
    return {k: (v, _slopes(v, T)) for k, v in cl.items()}


# --------------------------------------------------------------------------
# coolinmo.F:1175-1297: CO rotational + vibrational cooling (NK93, NLM95)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def co_tables():
    """CO cooling tables on 1 K x 0.1 dex grids as coolinmo fills them, log form, with forward differences."""
    r = raw()
    t_rot = 4.0 + 1.0 * np.arange(1, 1997)
    n_rot = 14.4 + 0.1 * np.arange(1, 47)
    t_vib = 99.0 + 1.0 * np.arange(1, 3902)
    n_vib = 12.9 + 0.1 * np.arange(1, 62)
    out = {"t_rot": t_rot, "n_rot": n_rot, "t_vib": t_vib, "n_vib": n_vib}
    out["L0"] = spline_eval(r["co_temp"], r["co_data_L0"], t_rot)
    for name, key in (("lte", "co_data_LTE"), ("n05", "co_data_n05"), ("alp", "co_data_alp")):
        data = r[key].reshape(10, 12)          # (column, temperature)
        small = np.array([spline_eval(r["co_temp"], row, t_rot) for row in data])
        out[name] = np.array([spline_eval(r["co_column"], small[:, j], n_rot)
                              for j in range(t_rot.size)]).T      # (46, 1996)
    data = r["co_vib_LTE"].reshape(8, 6)
    small = np.array([spline_eval(r["co_vib_temp"], row, t_vib) for row in data])
    out["vib"] = np.array([spline_eval(r["co_vib_column"], small[:, j], n_vib)
                           for j in range(t_vib.size)]).T         # (61, 3901)
    for k in ("L0", "lte", "n05", "alp", "vib"):
        d = out[k][..., 1:] - out[k][..., :-1]
        out["d" + k] = np.concatenate([d, d[..., -1:]], axis=-1)
    return out


# --------------------------------------------------------------------------
# calc_temp.F: e/(k n) -> T table (energy-table.txt)
# --------------------------------------------------------------------------
def spec_energy_h2(T):
    """calc_temp.F calc_spec: energy per H2 molecule / k [K] from eh2-table.txt; 1.5 T outside 1..5e4 K."""
    eh2 = raw()["eh2_table"]
    T = np.asarray(T, float)
    idx = np.clip(T.astype(int), 1, eh2.size - 1)
    inside = (T >= 1.0) & (T <= 5e4)
    en = eh2[idx - 1] + (T - idx) * (eh2[np.minimum(idx, eh2.size - 1)] - eh2[idx - 1])
    return np.where(inside, en, 1.5 * T)


@lru_cache(maxsize=None)
def energy_table():
    """energy-table.txt: T and gamma at e/(k n_tot) = 1..50000 K for x_H2 = 0.1..0.5, (5, 50000) each."""
    with np.load(data_path("energy_table.npz")) as z:
        temp, gamma = z["temp"].reshape(5, 50000), z["gamma"].reshape(5, 50000)
    ekn = np.arange(1, 50001, dtype=float)
    return np.broadcast_to(ekn, temp.shape), temp, gamma


@lru_cache(maxsize=None)
def energy_table_regenerated(abhe=ABHE):
    """The stand-in used before the original energy table was found (docs/CHEMISTRY.md)."""
    eh2 = raw()["eh2_table"]
    ekn = np.arange(1, 50001, dtype=float)
    Tn = np.arange(1, eh2.size + 1, dtype=float)          # integer T nodes 1..5e4
    temp = np.empty((5, ekn.size))
    for k in range(5):
        x = 0.1 * (k + 1)
        a = x / (1.0 + abhe - x)
        b = (1.0 + abhe - 2.0 * x) / (1.0 + abhe - x)
        en = a * eh2 + b * 1.5 * Tn                        # e/kn at the nodes
        envelope = np.maximum.accumulate(en)
        i = np.searchsorted(envelope, ekn, side="left") - 1   # first node reaching ekn
        ii = np.maximum(i, 0)
        assert np.all(en[ii + 1] > en[ii]) and np.all(en[ii + 1] >= ekn)
        lin = Tn[ii] + (ekn - en[ii]) / (en[ii + 1] - en[ii])
        temp[k] = np.where(i < 0, ekn / 1.5, lin)         # below 1 K: e/kn = 1.5 T
    gamma = 1.0 + temp / ekn
    return np.broadcast_to(ekn, temp.shape), temp, gamma
