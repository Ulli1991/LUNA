"""The face report of the dense hydro step: the Hancock face states that went unphysical (audit C8), when and where.

allow_first_order_faces=True: they took the first-order (cell-centre) state; each step with one is logged, the
run end prints the totals. False (default): the step made them NaN and the run stops here, naming step, t, cells
and faces. AMR (amr/hydro.report_amr_faces) sends its reports through jax.debug.callback into AMR_EVENTS.
"""
import numpy as np

AMR_EVENTS = []


def faces_of(mask):
    """'x+' etc. of a face bitmask (bit 2 d: the +d face, 2 d + 1: the -d face)."""
    return ",".join(f"{'xyz'[b // 2]}{'+-'[b % 2]}" for b in range(6) if int(mask) >> b & 1)


def _cells(rep):
    """(cells, face bitmasks) reported, lexicographically sorted, the -1 padding dropped."""
    c, f = np.asarray(rep["cells"]), np.asarray(rep["faces"])
    ok = (c >= 0).all(axis=1)
    c, f = c[ok], f[ok]
    o = np.lexsort(c.T[::-1]) if len(c) else np.arange(0)
    return c[o], f[o]


class FaceLog:
    """Reads one step's face report a step late (after the next dt, which syncs anyway), logs or refuses it."""

    def __init__(self, cfg, grid, is_main=True):
        self.allow = bool(getattr(cfg, "allow_first_order_faces", False))
        self.floors = bool(getattr(cfg, "preserve_face_states", False))
        self.h = (grid.dx, grid.dy, grid.dz)[:grid.spatial_dimension]
        self.is_main = is_main
        self.pending = None
        self.total = self.total_floor = self.steps = 0

    def push(self, rep, step, t):
        """Keep this step's report (device arrays, not read yet)."""
        self.pending = (rep, step, t)

    def where(self, listed):
        """'first at the x+ face of cell (i, j, k) x=(..)', then the other listed cells with their faces."""
        cells, faces = listed
        if not len(cells):
            return "no cell listed"
        i = cells[0]
        x = ", ".join(f"{(n + 0.5) * h:.6g}" for n, h in zip(i, self.h))
        s = f"first at the {faces_of(faces[0])} face of cell {tuple(int(v) for v in i)} x=({x})"
        if len(cells) > 1:
            s += "; also " + " ".join(f"{tuple(int(v) for v in c)}{faces_of(f)}" for c, f in zip(cells[1:], faces[1:]))
        return s

    def flush(self):
        """Read the pending report: log it, or raise when first-order faces are not allowed."""
        if self.pending is None:
            return
        rep, step, t = self.pending
        self.pending = None
        n = int(np.asarray(rep["n"]).astype(np.int64).sum())
        nf = int(np.asarray(rep["n_floor"]).astype(np.int64).sum())
        if n == 0 and nf == 0:
            return
        self.total += n
        self.total_floor += nf
        self.steps += n > 0
        where = self.where(_cells(rep))
        if n and not self.allow:
            raise RuntimeError(
                f"[hydro] step {step} t={t:.6e}: {n} unphysical Hancock face states (rho <= 0 or p <= 0, "
                f"audit C8), {where}. Refused: the step is second order only; the faces are NaN. Lower CFL, "
                "use a more diffusive limiter, or set allow_first_order_faces=True to take the first-order "
                "(cell-centre) state there.")
        if self.is_main:
            line = f"[hydro] step {step} t={t:.6e}: {n} first-order faces (C8)"
            if self.floors:
                line += f", {nf} RAMSES-floored faces (preserve_face_states)"
            print(f"{line}, {where}", flush=True)

    def finish(self):
        """Flush, then the run's totals (only when any face fell back or was floored)."""
        self.flush()
        if self.is_main and (self.total or self.total_floor):
            line = f"[hydro] run total: {self.total} first-order faces (C8) on {self.steps} steps"
            if self.floors:
                line += f", {self.total_floor} RAMSES-floored faces"
            print(line, flush=True)


def amr_face_event(level, n, slots, cells, masks, coords):
    """jax.debug.callback target of amr/hydro.report_amr_faces: keep one level's report for the runner."""
    AMR_EVENTS.append(dict(level=int(level), n=int(n), slots=np.asarray(slots), cells=np.asarray(cells),
                           masks=np.asarray(masks), coords=np.asarray(coords)))


def flush_amr(acfg, allow, steps, t0, t1, is_main=True, totals=None):
    """Read AMR_EVENTS after jax.effects_barrier(): raise (refused) or log them; steps, t0, t1: the chunk they came
    from (the AMR runner advances several steps per dispatch); totals: dict accumulating the run's count."""
    import jax
    ev, AMR_EVENTS[:] = list(AMR_EVENTS), []
    n = sum(e["n"] for e in ev)
    if jax.process_count() > 1:
        # each process hears its own devices' callbacks: decide together, or one rank raises while the
        # others wait in the next collective
        from jax.experimental import multihost_utils
        n_all = int(np.sum(multihost_utils.process_allgather(np.asarray(n, np.int64))))
        if n_all and not ev and not allow:
            raise RuntimeError(f"[amr] steps {steps[0]}..{steps[1]}: {n_all} unphysical Hancock face states "
                               "(audit C8) on another process (its log names them). Refused.")
    if not ev:
        return 0
    e = ev[0]
    ok = e["slots"] >= 0
    lines = []
    for slot, cell, mask, bc in list(zip(e["slots"][ok], e["cells"][ok], e["masks"][ok], e["coords"][ok]))[:8]:
        bs, L = acfg.block_shape, e["level"]
        dx = acfg.level_dx(L)
        where = f"the {faces_of(mask)} face of level {L} block {int(slot)} cell {tuple(int(v) for v in cell)}"
        if (bc >= 0).all():
            g = [int(b) * s + int(c) for b, s, c in zip(bc, bs, cell)]
            where += f" (block coords {tuple(int(v) for v in bc)}, x=(" + ", ".join(
                f"{(gi + 0.5) * d:.6g}" for gi, d in zip(g, dx)) + "))"
        lines.append(where)
    head = (f"[amr] steps {steps[0]}..{steps[1]} t={t0:.6e}..{t1:.6e}: {n} unphysical Hancock face states "
            f"(audit C8) on levels {sorted({x['level'] for x in ev})}, first at {lines[0] if lines else '?'}")
    if len(lines) > 1:
        head += "; also " + "; ".join(lines[1:])
    if not allow:
        raise RuntimeError(head + ". Refused: the step is second order only; the faces are NaN. Set "
                                  "allow_first_order_faces=True to take the first-order (cell-centre) state there.")
    if totals is not None:
        totals["n"] = totals.get("n", 0) + n
        totals["chunks"] = totals.get("chunks", 0) + 1
    if is_main:
        print(head.replace("unphysical Hancock face states", "first-order faces"), flush=True)
    return n
