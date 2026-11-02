"""Build $LUNA_DATA/chemistry/net5_raw.npz from Glover's chemcool Fortran sources.

Undecorated Fortran literals (``1.5``, no ``d`` exponent) are single precision
under gfortran's default REAL kind, so they are stored here rounded to float32
and widened back, exactly the value the Fortran holds in memory.

    python tools/chem_net5_tables.py GIZMO/cooling/chemcool
    python tools/chem_net5_tables.py --orig-energy-table energy-table.txt
    python tools/chem_net5_tables.py --cie GIZMO/cooling/chemcool

The second form packs Glover's energy-table.txt (e/kn -> T, gamma; not in the
chemcool source tree) into $LUNA_DATA/chemistry/energy_table.npz as exact
doubles (a list-directed Fortran read and numpy parse the same decimals).
The third packs the WSS_CIE_COOL data into $LUNA_DATA/chemistry/cie_tables.npz:
the temperatures and nine metal columns of wss_z_collis.txt (exact doubles) and
the HeI/HeII CIE fractions of coolinmo.F (single-precision literals).
numpy only: safe to run on a login node.
"""
import argparse
import os
import re

import numpy as np

OUT = os.path.join(os.environ.get("LUNA_DATA", "."), "chemistry", "net5_raw.npz")


def fortran_literal(tok):
    """Value a Fortran DATA literal holds in memory (d-exponent => double)."""
    tok = tok.strip()
    if re.search(r"[dD]", tok):
        return float(tok.replace("d", "e").replace("D", "e"))
    return float(np.float32(float(tok)))


def data_block(text, name):
    """The values of `DATA name /.../` in a fixed-form Fortran header."""
    lines = []
    for line in text.splitlines():
        if line[:1] in "cC*!":           # fixed-form comment line
            continue
        lines.append(line[6:] if len(line) > 5 and line[5] not in " 0" else line)
    flat = " ".join(lines)
    m = re.search(r"DATA\s+" + re.escape(name) + r"\s*/([^/]*)/", flat)
    if m is None:
        raise KeyError(name)
    return np.array([fortran_literal(t) for t in m.group(1).split(",") if t.strip()])


def extract(src):
    read = lambda f: open(os.path.join(src, f)).read()  # noqa: E731
    mol, isrf, shield = read("mol_data.h"), read("isrf.h"), read("shield_data.h")
    out = {}
    for name in ["h2_lte", "h2_h_rate", "h2_h2_rate",
                 "co_temp", "co_column", "co_data_L0", "co_data_LTE",
                 "co_data_n05", "co_data_alp",
                 "co_vib_temp", "co_vib_column", "co_vib_LTE"]:
        out[name] = data_block(mol, name)
    out["chi_tab_mathis"] = data_block(isrf, "chi_tab_mathis")
    out["chi_tab_black"] = data_block(isrf, "chi_tab_black")
    for name in ["CO_shield_columns", "CO_shield_factors",
                 "H2_shield_columns", "H2_shield_factors"]:
        out[name] = data_block(shield, name)
    # eh2-table.txt: T, e_H2/k per molecule (list-directed read into REAL*8)
    eh2 = np.loadtxt(os.path.join(src, "eh2-table.txt"))
    out["eh2_table"] = eh2[:, 1]
    # H2-cooling-ratios.dat: 101 x 101, temperature outer, column inner
    op = np.loadtxt(os.path.join(src, "H2-cooling-ratios.dat")).reshape(101, 101, 3)
    out["h2_opac_temp"] = op[:, 0, 0]
    out["h2_opac_column"] = op[0, :, 1]
    out["h2_opac"] = op[:, :, 2]
    sizes = dict(h2_lte=41, co_temp=12, co_column=10, co_data_LTE=120,
                 co_vib_temp=6, co_vib_column=8, co_vib_LTE=48,
                 chi_tab_mathis=200, CO_shield_columns=51, H2_shield_columns=43,
                 eh2_table=50000)
    for k, n in sizes.items():
        assert out[k].size == n, (k, out[k].size)
    return out


def extract_cie(src):
    """wss_z_collis.txt (T + the C..Fe columns 17-25) and coolinmo.F's HeI_tbl / HeII_tbl."""
    wss = np.loadtxt(os.path.join(src, "wss_z_collis.txt"))
    assert wss.shape == (352, 25)
    text = open(os.path.join(src, "coolinmo.F")).read()
    out = {"wss_temp": wss[:, 0], "wss_rates": wss[:, 16:25]}
    for name in ("HeI_tbl", "HeII_tbl"):
        m = re.search(r"\b" + name + r"\s*=\s*\(/(.*?)/\)", text, re.S)
        body = " ".join(line[6:] for line in m.group(1).splitlines())
        vals = np.array([fortran_literal(t) for t in body.replace("$", " ").split(",")
                         if t.strip()])
        assert vals.size == 51, (name, vals.size)
        out[name.lower()] = vals
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", nargs="?", help="chemcool source directory")
    ap.add_argument("--orig-energy-table", help="pack this energy-table.txt")
    ap.add_argument("--cie", help="chemcool source directory: pack the WSS_CIE_COOL tables")
    a = ap.parse_args()
    if a.cie:
        out = os.path.join(os.path.dirname(OUT), "cie_tables.npz")
        np.savez_compressed(out, **extract_cie(a.cie))
        print(f"wrote {out} ({os.path.getsize(out)/1e3:.0f} kB)")
    if a.src:
        data = extract(a.src)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        np.savez_compressed(OUT, **data)
        print(f"wrote {OUT} ({os.path.getsize(OUT)/1e3:.0f} kB)")
    if a.orig_energy_table:
        t = np.loadtxt(a.orig_energy_table)
        assert t.shape == (250000, 3)
        assert np.array_equal(t[:, 0], np.tile(np.arange(1, 50001), 5))
        out = os.path.join(os.path.dirname(OUT), "energy_table.npz")
        np.savez_compressed(out, temp=t[:, 1], gamma=t[:, 2])
        print(f"wrote {out} ({os.path.getsize(out)/1e3:.0f} kB)")

if __name__ == "__main__":
    main()
