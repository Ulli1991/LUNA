"""Pack the Taylor-Green spectral reference's velocity coefficients at a few times into a small npz for luna-reference-solutions.

    python tools/tg_reference_pack.py $LUNA_DATA/hydro/tg_spectral_N128.npz luna-reference-solutions/spectral/taylor_green/tg_spectral_N128.npz \
        [--times 1 2] [--eps 1e-9] [--source "<how the input was made>"]

Keeps every coefficient with |c| >= eps max|c| (sparse: component, kx, ky, kz index, value); the rest is
zero. tests/_tg_reference.reference rebuilds the dense field_c from it."""
import argparse
import datetime
import hashlib
import subprocess

import numpy as np


def pack(src, times, eps):
    """(dict for np.savez, max relative dropped |c|, kept count) from a tools/tg_spectral_reference.py file."""
    with np.load(src) as z:
        ft, kc = z["field_t"], int(z["kc"])
        idx = [int(np.argmin(np.abs(ft - t))) for t in times]
        assert all(abs(ft[i] - t) < 1e-6 for i, t in zip(idx, times)), (ft, times)
        c = z["field_c"][idx]
        out = dict(N=z["N"], kc=kc, field_t=ft[idx], shape=np.array(c.shape[1:]))
    cut = eps * np.abs(c).max()
    keep = np.abs(c) >= cut
    out["coord"] = np.argwhere(keep).astype(np.int16)          # (time, component, i, j, l)
    out["value"] = c[keep].astype(np.complex64)
    return out, float(np.abs(np.where(keep, 0, c)).max() / np.abs(c).max()), int(keep.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--times", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--source", default="", help="how the input file was made (stored as provenance)")
    a = ap.parse_args()
    out, dropped, n = pack(a.src, a.times, a.eps)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    md5 = hashlib.md5(open(a.src, "rb").read()).hexdigest()
    out["provenance"] = np.array(
        f"packed {datetime.datetime.now().isoformat(timespec='seconds')} by tools/tg_reference_pack.py "
        f"(luna {git}) from {a.src} (md5 {md5}), times {a.times}, eps {a.eps}: {n} coefficients kept, "
        f"largest dropped {dropped:.2e} of max |c|. Input: {a.source}")
    np.savez_compressed(a.out, **out)
    print(out["provenance"])


if __name__ == "__main__":
    main()
