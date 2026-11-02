"""The in-situ analysis of a run: what is due at time t, computed on the GPUs, written by one or all ranks."""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import jax
import numpy as np

from luna.analysis.config import InsituConfig
from luna.file_io.checkpoint_sets import write_atomically
from luna.file_io.checkpoint_utils import code_revision, config_json

KINDS = ("stats", "spectra", "sf", "snip", "frame")


def check(cfg):
    """Errors that make in-situ analysis meaningless for this run (empty list: fine)."""
    errs = []
    ic = cfg.insitu
    if ic is None:
        return errs
    if not isinstance(ic, InsituConfig):
        errs.append("insitu must be an InsituConfig")
        return errs
    if cfg.SPATIAL_DIMENSION != 3 or cfg.BOUNDARY != 0 or cfg.AMR:
        errs.append("insitu: 3-D periodic uniform grids only (SPATIAL_DIMENSION=3, BOUNDARY=0, no AMR)")
    if not (cfg.Nx == cfg.Ny == cfg.Nz and cfg.Lx == cfg.Ly == cfg.Lz):
        errs.append("insitu: cubic grid and box only (Nx = Ny = Nz, Lx = Ly = Lz)")
    if ic.snip_mode not in ("mean", "stride"):
        errs.append(f"insitu.snip_mode={ic.snip_mode!r}: 'mean' or 'stride'")
    if not 0 <= ic.frame_axis <= 2:
        errs.append(f"insitu.frame_axis={ic.frame_axis}: 0, 1 or 2")
    for k in KINDS:
        if getattr(ic, f"{k}_dt") < 0:
            errs.append(f"insitu.{k}_dt < 0")
    return errs


class InSitu:
    """Runs each analysis when its cadence is due; all ranks call `due`, every write is collective-safe."""

    def __init__(self, cfg, mesh, is_main=True):
        self.cfg, self.ic, self.mesh, self.is_main = cfg, cfg.insitu, mesh, is_main
        self.n = int(cfg.Nx)
        self.box = float(cfg.Lx)
        self.iso = float(cfg.isothermal_cs2) > 0.0
        self.root = os.path.join(cfg.output_dir, self.ic.subdir)
        self.last = {k: None for k in KINDS}
        self.frame_count = 0
        self._fns = {}
        self._writer = ThreadPoolExecutor(max_workers=1)    # frames leave the step loop at once
        self._pending = []
        self.header = {"code": "LUNA", "code_revision": code_revision(), "N": self.n, "box": self.box,
                       "dx": self.box / self.n, "boundary": "periodic", "gamma": float(cfg.gamma),
                       "isothermal_cs2": float(cfg.isothermal_cs2),
                       "mesh": [int(mesh.shape[a]) for a in ("x", "y", "z")],
                       "config_json": config_json(cfg)}
        if self.ic.snip_dt > 0:
            nloc = [self.n // s for s in self.header["mesh"]]
            if any(m % self.ic.snip_factor for m in nloc):
                raise ValueError(f"insitu.snip_factor={self.ic.snip_factor} must divide the "
                                 f"per-GPU block {nloc}")
        if self.ic.spectra_dt > 0:
            from luna.analysis.spectra import check_pencil
            check_pencil(self.n, tuple(self.header["mesh"]))
        if is_main:
            os.makedirs(self.root, exist_ok=True)
            with open(os.path.join(self.root, "header.json"), "w") as fh:
                json.dump({k: v for k, v in self.header.items() if k != "config_json"}, fh, indent=2)

    def _fn(self, kind):
        if kind not in self._fns:
            c, ic, m = self.cfg, self.ic, self.mesh
            if kind == "stats":
                from luna.analysis.pdfs import build_stats
                self._fns[kind] = build_stats(m, c.gamma, self.box / self.n, self.n, ic.pdf_bins,
                                              ic.pdf_ranges, self.iso)
            elif kind == "spectra":
                from luna.analysis.spectra import build_spectra
                self._fns[kind] = build_spectra(m, c.gamma, self.box, self.n, ic.spectra_single)
            elif kind == "sf":
                from luna.analysis.structure import build_structure_functions
                self._fns[kind] = build_structure_functions(m, c.gamma, self.n, ic.sf_orders)
            elif kind == "snip":
                from luna.analysis.reduced import build_coarsen
                self._fns[kind] = build_coarsen(m, ic.snip_factor, ic.snip_mode, self.iso)
            elif kind == "frame":
                from luna.analysis.reduced import build_frames
                self._fns[kind] = build_frames(m, c.gamma, self.box, self.n, ic.frame_axis,
                                               tuple(ic.frame_fields), ic.frame_ranges)
        return self._fns[kind]

    def due(self, kind, t):
        """True once per cadence interval of `kind` (the first call records the start)."""
        dt = getattr(self.ic, f"{kind}_dt")
        if dt <= 0:
            return False
        k = math.floor(t / dt + 1e-9)
        if self.last[kind] is not None and k <= self.last[kind]:
            return False
        self.last[kind] = k
        return True

    def after_step(self, state):
        """Run and write whatever is due at state.t."""
        U, t, step = state.U, float(state.t), int(state.step)
        for kind in ("stats", "spectra", "sf"):
            if self.due(kind, t):
                self._write_record(kind, self._fn(kind)(U), t, step)
        if self.due("snip", t):
            from luna.analysis.reduced import snip_header, write_snipshot
            hdr = snip_header({**self.header, "time": t, "step": step}, self.ic.snip_factor,
                              self.ic.snip_mode, self.iso, self.n, self.box)
            write_snipshot(self._fn("snip")(U), hdr, os.path.join(self.root, "snip", f"{step:09d}"))
        if self.due("frame", t):
            self._frames(U, t, step)

    def _write_record(self, kind, out, t, step):
        """Rank 0 writes one HDF5 file per record (every output is replicated)."""
        if not self.is_main:
            return
        d = os.path.join(self.root, kind)
        os.makedirs(d, exist_ok=True)

        def write(tmp):
            with h5py.File(tmp, "w") as f:
                for k, v in {**self.header, "time": t, "step": step}.items():
                    f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
                for k, v in out.items():
                    f.create_dataset(k, data=np.asarray(v.addressable_data(0)))
        write_atomically(os.path.join(d, f"{step:09d}.h5"), write)

    def _frames(self, U, t, step):
        """Every rank holds every image; frame i of a field is written by rank i mod process_count."""
        from luna.analysis.reduced import FRAMES, write_frame
        fn, ranges = self._fn("frame")
        out = fn(U)
        writer = self.frame_count % jax.process_count()
        self.frame_count += 1
        if jax.process_index() != writer:
            return
        for name, (img, clipped) in out.items():
            d = os.path.join(self.root, "frames", name)
            os.makedirs(d, exist_ok=True)
            hdr = {**{k: v for k, v in self.header.items() if k != "config_json"},
                   "time": t, "step": step, "field": name, "definition": FRAMES[name][1],
                   "axis": int(self.ic.frame_axis), "range": list(ranges[name]),
                   "encoding": "uint16, value = lo + (hi - lo) * pixel / 65535; image[i, j] over the two other axes in order",
                   "clipped_below_above": [float(c) for c in np.asarray(clipped.addressable_data(0))]}
            self._pending.append(self._writer.submit(
                write_frame, os.path.join(d, f"frame_{step:09d}.h5"),
                np.asarray(img.addressable_data(0)), hdr, self.ic.frame_png_max))
        self._pending = [f for f in self._pending if not f.done() or f.result()]

    def close(self):
        """Wait for the frames still being written; a failed write raises here."""
        for f in self._pending:
            f.result()
        self._pending = []
        self._writer.shutdown(wait=True)
