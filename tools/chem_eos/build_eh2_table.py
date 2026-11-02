"""Recover the H2 internal energy e_H2(T) and its slope from Glover's energy-table.txt.

    python tools/chem_eos/build_eh2_table.py [energy-table.txt] [--out $LUNA_DATA/chemistry/eh2_table.npz]

The table holds e/(k n_tot) -> T, gamma for x_H2 = 0.1..0.5 at He/H = 0.1, i.e.
ekn = [x e_H2 + (1+a-2x) 1.5 T] / (1+a-x) and gamma = 1 + 1/(d ekn/dT). Each block
gives e_H2 and e_H2' at its T samples; the shipped table resamples the x = 0.1 block
(the widest T range) by cubic Hermite onto a log-T grid. Also prints how well the
five blocks agree (docs/CHEMISTRY.md, "Temperature by a Newton solve").
"""
import argparse
import os

import numpy as np
from scipy.interpolate import CubicHermiteSpline

DATA = os.path.join(os.environ.get("LUNA_DATA", "."), "chemistry")
A0 = 0.1                      # He/H the table was built for (double 0.1 fits best)
XS = (0.1, 0.2, 0.3, 0.4, 0.5)
N = 50000
DEX = 1e-3                    # node spacing in log10 T
T_TOP = 5e4                   # last node at or above this; e_H2 = 3 T beyond the data
E_HOT = 3.0                   # e_H2 / T above ~2e4 K in the table (exact there)
T_COLD = 4.0                  # below, e_H2 = 1.5 T (the table: to 1e-13, i.e. its rounding)


def blocks(tab, a0=A0):
    """Per x block: (T, e_H2, e_H2') recovered from (ekn, T, gamma)."""
    out = []
    for k, x in enumerate(XS):
        ekn, t, g = tab[N * k:N * (k + 1)].T
        e = (ekn * (1 + a0 - x) - (1 + a0 - 2 * x) * 1.5 * t) / x
        de = ((1 + a0 - x) / (g - 1) - 1.5 * (1 + a0 - 2 * x)) / x
        out.append((t, e, de))
    return out


def agreement(bl):
    """Each block's e_H2 (and slope) against the x = 0.1 block's Hermite interpolant."""
    t0, e0, d0 = bl[0]
    h = CubicHermiteSpline(t0, e0, d0)
    rows = []
    for k in range(1, len(bl)):
        t, e, d = bl[k]
        m = (t > 1.0) & (t <= t0[-1])
        re = np.abs(h(t[m]) / e[m] - 1)
        rd = np.abs(h.derivative()(t[m]) / d[m] - 1)
        rows.append((XS[k], re.max(), t[m][np.argmax(re)], np.median(re), rd.max(),
                     t[m][np.argmax(rd)], np.median(rd)))
    return rows


def resample(bl):
    """Nodes T_i = 10**(i DEX) from 1 K, with e_H2 and e_H2' of the x = 0.1 block."""
    t0, e0, d0 = bl[0]
    h = CubicHermiteSpline(t0, e0, d0)
    n = int(np.ceil(np.log10(T_TOP) / DEX)) + 1
    t = 10.0 ** (DEX * np.arange(n))
    inside = t <= t0[-1]
    tc = np.minimum(t, t0[-1])
    e = np.where(inside, h(tc), E_HOT * t)
    de = np.where(inside, h.derivative()(tc), E_HOT)
    cold = t < T_COLD
    return t, np.where(cold, 1.5 * t, e), np.where(cold, 1.5, de)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=os.path.join(DATA, "eh2_table.npz"))
    a = ap.parse_args()
    bl = blocks(np.loadtxt(a.src))
    for x, emax, te, emed, dmax, td, dmed in agreement(bl):
        print(f"x={x:.1f} vs x=0.1: e_H2 max {emax:.1e} (T={te:.4g}) median {emed:.1e}; "
              f"e_H2' max {dmax:.1e} (T={td:.4g}) median {dmed:.1e}")
    t, e, de = resample(bl)
    np.savez_compressed(a.out, ln_t0=np.log(t[0]), dln_t=DEX * np.log(10.0), e_h2=e, de_h2=de)
    print(f"{a.out}: {t.size} nodes, T = {t[0]:g} .. {t[-1]:.6g} K")


if __name__ == "__main__":
    main()
