"""AthenaK snapshot loader.

AthenaK writes a custom `.bin` format during simulation runs. Their
companion Python module (`vis/python/bin_convert.py`) supplies a
`read_binary` function that returns a dict of per-meshblock arrays.

This module is a thin LUNA-side wrapper around `read_binary` that
re-shapes the per-meshblock data into flat 1D numpy arrays in the same
format LUNA's plot scripts expect: {t, gamma, x, rho, vx, P, e_int}.

Set ATHENAK_VIS_PATH if your AthenaK install is somewhere other than
/raven/u/uli/athenak/vis/python.
"""
import os
import sys
import numpy as np


ATHENAK_VIS_PATH = os.environ.get(
    "ATHENAK_VIS_PATH",
    "/raven/u/uli/athenak/vis/python",
)


def _import_bin_convert():
    if ATHENAK_VIS_PATH not in sys.path:
        sys.path.insert(0, ATHENAK_VIS_PATH)
    import bin_convert  # noqa: E402
    return bin_convert


def inspect(path):
    """Dump the structure of an AthenaK .bin file for debugging."""
    bc = _import_bin_convert()
    fd = bc.read_binary(path)
    print(f"File: {path}")
    print(f"  time  = {fd['time']}")
    print(f"  cycle = {fd['cycle']}")
    print(f"  Nx1,Nx2,Nx3 = {fd['Nx1']}, {fd['Nx2']}, {fd['Nx3']}")
    print(f"  n_mbs = {fd['n_mbs']}")
    print(f"  var_names = {fd['var_names']}")
    print(f"  x1 range = [{fd['x1min']}, {fd['x1max']}]")


def load_1d(path, gamma=5.0 / 3.0):
    """Load a 1D AthenaK snapshot (.bin) into a flat dict.

    Variables stored by AthenaK for hydro_w output are PRIMITIVES:
        dens, velx, vely, velz, eint  (eint = specific internal energy)

    LUNA uses pressure rather than eint, so we convert: P = (γ-1) ρ eint.

    Parameters
    ----------
    path : str
        Path to .bin file.
    gamma : float
        Adiabatic index (AthenaK doesn't store it in the binary header).

    Returns
    -------
    dict with keys: t, gamma, x, rho, vx, P, e_int
    """
    bc = _import_bin_convert()
    fd = bc.read_binary(path)

    if fd["Nx2"] != 1 or fd["Nx3"] != 1:
        raise ValueError(
            f"load_1d expects a 1D snapshot, got shape ({fd['Nx1']}, {fd['Nx2']}, {fd['Nx3']})"
        )

    Nx = int(fd["Nx1"])
    x1min, x1max = float(fd["x1min"]), float(fd["x1max"])
    dx = (x1max - x1min) / Nx
    x = x1min + (np.arange(Nx) + 0.5) * dx

    # mb_data stores per-meshblock arrays; for a single-MB 1D run, each is
    # shape (n_mbs=1, nz=1, ny=1, nx=Nx). We squeeze + concatenate across MBs
    # ordered by meshblock x1 origin.
    def _stitch(varname):
        arrs = fd["mb_data"][varname]  # numpy array, leading axis is meshblock index
        # For a uniform-grid single-MB case, this is straightforward.
        # If multi-MB, sort by mb_geometry[:,0] (x1min of each block).
        if fd["n_mbs"] == 1:
            return np.squeeze(arrs[0]).ravel()
        # Multi-MB: order by x1min of each block, then concatenate along x.
        geom = np.asarray(fd["mb_geometry"])  # (n_mbs, 6) usually
        order = np.argsort(geom[:, 0])
        pieces = [np.squeeze(arrs[i]).ravel() for i in order]
        return np.concatenate(pieces)

    rho = _stitch("dens")
    vx = _stitch("velx")
    eint = _stitch("eint")
    # AthenaK stores `eint` as the INTERNAL ENERGY DENSITY (p/(γ-1)), not the
    # specific internal energy per mass. So pressure is (γ-1)·eint, NOT
    # (γ-1)·ρ·eint as in the standard "specific eint" convention.
    P = (gamma - 1.0) * eint

    return {
        "t": float(fd["time"]),
        "gamma": gamma,
        "x": x,
        "rho": rho,
        "vx": vx,
        "P": P,
        "e_int": eint,
    }
