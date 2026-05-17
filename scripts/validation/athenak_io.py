"""AthenaK .athdf snapshot loader.

AthenaK writes HDF5 files whose exact dataset/group layout depends on the
output block configuration and whether mesh-refinement is on. The cleanest
path: write a layout-introspection helper now, then have a real loader once
we have a sample .athdf from the user's reference runs.

Until then `load_1d` returns a best-effort dict for the common single-block
unrefined output, and `inspect` prints the full HDF5 tree so the user can
confirm the layout when the file arrives.
"""
import h5py
import numpy as np


def inspect(path):
    """Print the HDF5 tree of an AthenaK .athdf file.

    Useful first step when a new reference file arrives — confirms which
    dataset names and dimensions to expect.
    """
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        print("Attributes:")
        for k, v in f.attrs.items():
            print(f"  {k} = {v}")
        print("Structure:")

        def _visit(name, obj):
            kind = "GROUP" if isinstance(obj, h5py.Group) else "DSET"
            shape = getattr(obj, "shape", "")
            dtype = getattr(obj, "dtype", "")
            print(f"  [{kind}] {name}  {shape} {dtype}")
            for ak, av in obj.attrs.items():
                print(f"      @{ak} = {av}")

        f.visititems(_visit)


def load_1d(path, prefer="prim"):
    """Load a 1D AthenaK snapshot into a flat dict.

    Tries the conventional layout used by AthenaK for single-meshblock,
    uniform-grid outputs:
        - /x1v       cell-center x coordinates (shape (Nmb, Nx_block))
        - /prim or /cons  variables, shape (Nvar, Nmb, ..., Nx_block)
                          Variable index order from `VariableNames` attr.
        - Time attribute on root group.

    For uniform single-block 1D the returned dict has flat 1D arrays:
        {"x": (Nx,), "rho": (Nx,), "vx": (Nx,), "P": (Nx,),
         "t": float, "gamma": float}

    NOTE: not yet exercised against a real .athdf file. Will likely need
    tweaks once the user provides a reference snapshot — use `inspect()`
    first to confirm the dataset names.
    """
    with h5py.File(path, "r") as f:
        t = float(f.attrs.get("Time", np.nan))
        gamma = float(f.attrs.get("Gamma", 5.0 / 3.0))

        # Variable name ordering, if recorded.
        varnames_raw = f.attrs.get("VariableNames", None)
        if varnames_raw is None:
            # Fall back to a common AthenaK ordering for hydro primitives.
            varnames = ["dens", "velx", "vely", "velz", "eint"] if prefer == "prim" \
                       else ["dens", "mom1", "mom2", "mom3", "etot"]
        else:
            varnames = [
                (v.decode() if isinstance(v, bytes) else str(v))
                for v in varnames_raw
            ]

        if prefer not in f:
            # Some AthenaK builds write top-level datasets per variable instead.
            data = {name: np.asarray(f[name]).ravel() for name in varnames if name in f}
        else:
            arr = np.asarray(f[prefer])
            # arr shape is typically (Nvar, Nmb, Nx) for 1D single-block.
            # Squeeze meshblock axes and ravel any leftover singleton spatial dims.
            arr = np.squeeze(arr)
            data = {name: arr[i].ravel() for i, name in enumerate(varnames[:arr.shape[0]])}

        x = np.squeeze(np.asarray(f["x1v"])).ravel() if "x1v" in f else None

    # Normalize to LUNA-side names (rho, vx, P).
    out = {"t": t, "gamma": gamma}
    if x is not None:
        out["x"] = x
    if "dens" in data:
        out["rho"] = data["dens"]
    if "velx" in data:
        out["vx"] = data["velx"]
    if "press" in data:
        out["P"] = data["press"]
    elif "eint" in data and "rho" in out:
        # If only specific internal energy is stored: P = (gamma-1) * rho * eint
        out["P"] = (gamma - 1.0) * out["rho"] * data["eint"]
    return out
