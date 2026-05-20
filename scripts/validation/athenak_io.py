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


def load_2d(path, gamma=5.0 / 3.0):
    """Load a 2D AthenaK snapshot (.bin) and stitch meshblocks into one array.

    Returns dict with keys: t, gamma, x, y, Lx, Ly, rho, vx, vy, P, e_int.
    rho/vx/vy/P are (Nx, Ny) numpy arrays in code units.
    """
    bc = _import_bin_convert()
    fd = bc.read_binary(path)

    if fd["Nx3"] != 1:
        raise ValueError(
            f"load_2d expects a 2D snapshot, got shape "
            f"({fd['Nx1']}, {fd['Nx2']}, {fd['Nx3']})"
        )

    Nx = int(fd["Nx1"]); Ny = int(fd["Nx2"])
    x1min, x1max = float(fd["x1min"]), float(fd["x1max"])
    x2min, x2max = float(fd["x2min"]), float(fd["x2max"])
    dx = (x1max - x1min) / Nx
    dy = (x2max - x2min) / Ny
    x = x1min + (np.arange(Nx) + 0.5) * dx
    y = x2min + (np.arange(Ny) + 0.5) * dy

    # Per-MB arrays shape (n_mbs, nz=1, ny_block, nx_block). Stitch into
    # global (Nx, Ny) by mb_geometry[:, 0:2] for x1min and mb_geometry[:, 2:4]
    # for x2min — locate each MB in the global grid.
    geom = np.asarray(fd["mb_geometry"])
    n_mbs = int(fd["n_mbs"])

    def _stitch(varname):
        # AthenaK's bin_convert returns mb_data[var] as a *list* of per-MB
        # arrays (could be a numpy array of arrays). Force to numpy.
        arrs = fd["mb_data"][varname]
        first = np.asarray(arrs[0])
        out = np.empty((Nx, Ny), dtype=first.dtype)
        for m in range(n_mbs):
            block = np.squeeze(np.asarray(arrs[m]))  # (ny_block, nx_block)
            block_ny, block_nx = block.shape
            x_lo = geom[m, 0]
            y_lo = geom[m, 2]
            ix = int(round((x_lo - x1min) / dx))
            iy = int(round((y_lo - x2min) / dy))
            out[ix:ix+block_nx, iy:iy+block_ny] = block.T
        return out

    rho  = _stitch("dens")
    vx   = _stitch("velx")
    vy   = _stitch("vely")
    eint = _stitch("eint")
    P    = (gamma - 1.0) * eint

    return {
        "t":     float(fd["time"]),
        "gamma": gamma,
        "x":     x,
        "y":     y,
        "Lx":    x1max - x1min,
        "Ly":    x2max - x2min,
        "rho":   rho,
        "vx":    vx,
        "vy":    vy,
        "P":     P,
        "e_int": eint,
    }
