"""Snapshot writer for the coupled gas+DM cosmological runs."""
import os

import h5py
import jax
import numpy as np


def _local(arr):
    """This process's shard of a globally sharded grid, as float32."""
    return np.asarray(arr.addressable_shards[0].data, dtype=np.float32)


def write_coupled_snapshot(cfg, out_dir, a_now, rho_gas, rho_dm,
                           lo=None, hi=None, is_main_process=True):
    """Write gas and DM density at scale factor ``a_now``; return the path.

    On several processes each writes its shard to ``coupled_a<a>_rank<n>.hdf5``
    with its ``lo``/``hi`` index box (required then); no rank gathers the grid.
    """
    os.makedirs(out_dir, exist_ok=True)
    nproc = jax.process_count()

    def _header(f):
        f.attrs["a"] = a_now
        f.attrs["z"] = 1.0 / a_now - 1.0
        f.attrs["BoxSize"] = float(cfg.Lx)
        f.attrs["Nx"] = int(cfg.Nx)
        f.attrs["Ny"] = int(cfg.Ny)
        f.attrs["Nz"] = int(cfg.Nz)

    if nproc > 1:
        if lo is None or hi is None:
            raise ValueError("write_coupled_snapshot needs lo/hi on "
                             f"{nproc} processes: without the index box the "
                             "per-rank files cannot be reassembled")
        pid = jax.process_index()
        fn = os.path.join(out_dir, f"coupled_a{a_now:.4f}_rank{pid:03d}.hdf5")
        with h5py.File(fn, "w") as f:
            _header(f)
            f.attrs["rank"] = pid
            f.attrs["nproc"] = nproc
            f.attrs["lo"] = np.asarray(lo, dtype=np.int64)
            f.attrs["hi"] = np.asarray(hi, dtype=np.int64)
            f.create_dataset("rho_gas", data=_local(rho_gas))
            f.create_dataset("rho_dm", data=_local(rho_dm))
        if pid == 0:
            print(f"[cosmo-hydro] wrote per-rank snaps "
                  f"coupled_a{a_now:.4f}_rank*.hdf5 ({nproc} files)", flush=True)
        return fn

    fn = os.path.join(out_dir, f"coupled_a{a_now:.4f}.hdf5")
    with h5py.File(fn, "w") as f:
        _header(f)
        f.create_dataset("rho_gas", data=np.asarray(rho_gas, dtype=np.float32))
        f.create_dataset("rho_dm", data=np.asarray(rho_dm, dtype=np.float32))
    if is_main_process:
        print(f"[cosmo-hydro] wrote {fn}", flush=True)
    return fn
