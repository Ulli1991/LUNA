"""McNally, Lyra & Passy 2012 (arXiv:1111.1764v2) reference curves read from the vector figures of the arXiv source.

    python tools/mcnally_digitize.py SRC_DIR [out=<luna-reference-solutions>/pencil/mcnally_kh]      # SRC_DIR holds fig4.pdf, fig8.pdf (arXiv source)

The figures are matplotlib PDFs whose content streams store path coordinates with 10 decimals; this parses those
numbers as float64 (MuPDF's drawing API reads them as float32, ~1e-6 relative). The axes are calibrated from the
tick marks the same streams draw. Writes mcnally2012_fig4_Mref.txt (Pencil 4096^2 M(t), fig. 4 top, and its GCI,
fig. 4 bottom) and mcnally2012_fig8_KEref.txt (Pencil 4096^2 max(rho V_y^2 / 2), fig. 8). Host numpy only.
"""
import hashlib
import os
import re
import sys
import zlib

import numpy as np

from luna.reference_data import reference_root

NUM = r"-?\d+(?:\.\d*)?|-?\.\d+"
TOKEN = re.compile(rf"\[[^\]]*\]|\([^)]*\)|/[A-Za-z0-9_.+-]+|{NUM}|[A-Za-z'\"*]+")
BLUE = (0.0, 0.0, 1.0)


def content(path):
    """The page content stream of a one-page PDF, inflated."""
    raw = open(path, "rb").read()
    out = []
    for m in re.finditer(rb"<<(.*?)>>\s*stream\r?\n", raw, re.S):
        if b"/Length" not in m.group(1):
            continue
        start = m.end()
        data = raw[start:raw.index(b"endstream", start)]
        try:
            data = zlib.decompress(data) if b"/FlateDecode" in m.group(1) else data
        except zlib.error:
            continue
        text = data.decode("latin-1")
        if " re W n" in text or " l\n" in text or " l " in text:
            out.append(text)
    return max(out, key=len)


def paths(stream):
    """[(clip rect, stroke rgb, dash, [[(x, y), ...] per subpath], painted op)] of every painted path."""
    st, stack, out, cur, args = dict(clip=None, rgb=(0.0, 0.0, 0.0), dash=""), [], [], [], []
    for tok in TOKEN.findall(stream):
        if re.fullmatch(NUM, tok) or tok[0] in "[(/":
            args.append(tok)
            continue
        if tok == "q":
            stack.append(dict(st))
        elif tok == "Q":
            st = stack.pop()
        elif tok == "re":
            cur.append(("re", [float(a) for a in args[-4:]]))
        elif tok == "W":
            st["clip"] = next((tuple(v) for k, v in reversed(cur) if k == "re"), None)
        elif tok == "RG":
            st["rgb"] = tuple(float(a) for a in args[-3:])
        elif tok == "G":
            st["rgb"] = (float(args[-1]),) * 3
        elif tok == "d":
            st["dash"] = args[-2] if len(args) >= 2 else ""
        elif tok == "m":
            cur.append(("m", [(float(args[-2]), float(args[-1]))]))
        elif tok == "l":
            cur[-1][1].append((float(args[-2]), float(args[-1])))
        elif tok in ("S", "B", "f", "n", "h", "b"):
            if tok in ("S", "B"):
                subs = [pts for k, pts in cur if k == "m"]
                if subs:
                    out.append((st["clip"], st["rgb"], st["dash"], subs, tok))
            if tok != "h":
                cur = []
        args = []
    return out


def curves(stream, clip):
    """{(rgb, dash): points (n, 2)} of the stroked multi-point curves inside the clip rectangle `clip`."""
    out = {}
    for c, rgb, dash, subs, op in paths(stream):
        if op == "S" and c is not None and np.allclose(c, clip):
            pts = np.array([p for s in subs for p in s])
            if len(pts) > 2:
                out.setdefault((rgb, dash.replace(" ", "")), []).append(pts)
    return {k: np.concatenate(v) for k, v in out.items()}


def solid_blue(stream, clip):
    """The solid blue curve in `clip` (matplotlib's first line: the paper's Pencil reference)."""
    found = {k: v for k, v in curves(stream, clip).items() if k[0] == BLUE and k[1] in ("", "[]")}
    (pts,) = found.values()
    return pts


def ticks(stream, x_frame, length):
    """Sorted y of the horizontal tick marks of `length` points drawn from the frame edge x = x_frame."""
    out = []
    for _, _, _, subs, op in paths(stream):
        if op == "B" and len(subs) == 1 and len(subs[0]) == 2:
            (xa, ya), (xb, yb) = subs[0]
            if abs(xa - x_frame) < 1e-9 and abs(ya - yb) < 1e-9 and abs(abs(xb - xa) - length) < 1e-9:
                out.append(ya)
    return np.array(sorted(set(out)))


def axis(majors, lo, hi, values):
    """y -> value on an axis whose major ticks inside [lo, hi] carry `values` (evenly spaced, checked)."""
    y = majors[(majors >= lo - 1e-9) & (majors <= hi + 1e-9)]
    assert len(y) == len(values) and np.ptp(np.diff(y)) < 1e-8, f"ticks {y} for {values}"
    return lambda yy: values[0] + (np.asarray(yy) - y[0]) / (y[1] - y[0]) * (values[1] - values[0])


def times(pts, x0, x_per_unit):
    """t of the points, checked to be the paper's outputs t = 0.02 k, k = 0, 1, ..."""
    t = (pts[:, 0] - x0) / x_per_unit
    k = np.round(t / 0.02)
    dev = np.abs(t - 0.02 * k).max()
    assert dev < 1e-6 and (np.diff(k) == 1).all() and k[0] == 0, f"not t = 0.02 k: {dev:.1e}"   # fig. 8: float32 t
    return 0.02 * k


def fig4(src):
    """(t, M_ref, GCI_abs): fig. 4 top (log10 M, majors 1e-3 ... 1) and bottom (GCI, majors 0 ... 1.2e-7)."""
    s = content(os.path.join(src, "fig4.pdf"))
    top, bottom = (72.0, 231.7090909091, 446.4, 157.0909090909), (72.0, 43.2, 446.4, 157.0909090909)
    majors = ticks(s, 72.0, 4.0)
    pm, pg = solid_blue(s, top), solid_blue(s, bottom)
    t = times(pm, 72.0, 446.4 / 1.6)                     # frame 72 ... 518.4 is t = 0 ... 1.6 (ticks every 0.2)
    assert np.array_equal(t, times(pg, 72.0, 446.4 / 1.6))
    log_m = axis(majors, top[1], top[1] + top[3], [-3.0, -2.0, -1.0, 0.0])
    gci = axis(majors, bottom[1], bottom[1] + bottom[3], [0.0, 0.2e-7, 0.4e-7, 0.6e-7, 0.8e-7, 1.0e-7, 1.2e-7])
    return t, 10.0 ** log_m(pm[:, 1]), gci(pg[:, 1])


def fig8(src):
    """(t, K_ref, spread): fig. 8's top-panel solid 'Reference' (log10 majors 1e-4 ... 1e-1) and its largest
    relative difference to the copies the six other panels draw."""
    s = content(os.path.join(src, "fig8.pdf"))
    majors = ticks(s, 51.84, 4.0)
    panels = sorted({c for c, *_ in paths(s) if c is not None}, key=lambda c: -c[1])
    assert len(panels) == 7, panels                        # Pe, Ep, At, Ph, Ne, Nc, No, top to bottom
    refs = []
    for c in panels:
        pts = solid_blue(s, c)
        t = times(pts, 51.84, 230.4 / 1.5)                 # frame 51.84 ... 282.24 is t = 0 ... 1.5
        refs.append(10.0 ** axis(majors, c[1], c[1] + c[3], [-4.0, -3.0, -2.0, -1.0])(pts[:, 1]))
    spread = max(float(np.abs(r / refs[0] - 1).max()) for r in refs[1:])
    return t, refs[0], spread


def main(argv):
    src = argv[0]
    out = next((a[4:] for a in argv if a.startswith("out=")), os.path.join(reference_root(), "pencil", "mcnally_kh"))
    head = ("McNally, Lyra & Passy 2012, ApJS 201, 18, arXiv:1111.1764v2 (arXiv source, {fig}); Pencil Code 4096^2 "
            "reference.\nRead by tools/mcnally_digitize.py from the PDF content stream (10-decimal path coordinates "
            "in float64),\naxes calibrated on the drawn tick marks; times are t = 0.02 k (to 1e-6; fig. 4 to 1e-8).\n"
            "source sha256 {sha}\n")
    sha = {f: hashlib.sha256(open(os.path.join(src, f), "rb").read()).hexdigest() for f in ("fig4.pdf", "fig8.pdf")}
    t, m, gci = fig4(src)
    np.savetxt(os.path.join(out, "mcnally2012_fig4_Mref.txt"), np.column_stack([t, m, gci]), fmt="%.2f %.10e %.6e",
               header=head.format(sha=sha["fig4.pdf"], fig="fig4.pdf, top: mode amplitude M, eqs. 6-9; bottom: GCI, eq. 19, F_s = 1.25")
               + "columns: t  M_ref  GCI_abs")
    t8, k, spread = fig8(src)
    np.savetxt(os.path.join(out, "mcnally2012_fig8_KEref.txt"), np.column_stack([t8, k]), fmt="%.2f %.10e",
               header=head.format(sha=sha["fig8.pdf"], fig="fig8.pdf, the solid 'Reference' curve, top panel: max(rho V_y^2 / 2)")
               + f"the six other panels redraw it: largest relative difference {spread:.1e}\ncolumns: t  K_ref")
    print(f"M_ref: {len(t)} points, M(0) = {m[0]!r}, M(1.5) = {m[-1]!r}, max GCI {gci.max():.3e}")
    print(f"K_ref: {len(t8)} points, K(0) = {k[0]!r}, K(1.5) = {k[-1]!r}, panel spread {spread:.2e}")


if __name__ == "__main__":
    main(sys.argv[1:])
