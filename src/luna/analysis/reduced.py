"""Snipshots (reduced-sampling rho, rho v) and movie frames, written in parallel without a global gather of the state."""
import json
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from luna.analysis.fields import AXES, primitives, vorticity_divergence
from luna.file_io.checkpoint_sets import write_atomically
from luna.utils.shmap import shard_map

# frame field -> (default range of the plotted quantity, its definition)
FRAMES = {"column_density": ((-1.5, 1.5), "log10(Sigma / <Sigma>), Sigma = sum of rho dx along the axis"),
          "slice_s": ((-6.0, 4.0), "ln(rho / <rho>) in the mid-plane normal to the axis"),
          "slice_log10_vorticity": ((-1.0, 3.0), "log10 |curl v| in the mid-plane, code units")}


def build_coarsen(mesh, factor, mode, isothermal):
    """Return coarsen(U) -> (rho, rho v[, E]) on a grid `factor` times coarser, still sharded."""
    nf = 4 if isothermal else 5

    def _coarse(U):
        U = U[:nf]
        if mode == "stride":
            return U[:, factor // 2::factor, factor // 2::factor, factor // 2::factor]
        c, nx, ny, nz = U.shape
        f = factor
        return U.reshape(c, nx // f, f, ny // f, f, nz // f, f).mean(axis=(2, 4, 6))

    spec = P(None, *AXES)
    local = shard_map(_coarse, mesh=mesh, in_specs=(spec,), out_specs=spec)
    slabs = NamedSharding(mesh, P(None, "x", None, None))
    # gathered within each x-slab (an all-gather over y, z): one file per slab, not per GPU
    return jax.jit(lambda U: jax.lax.with_sharding_constraint(local(U), slabs))


def snip_header(header, factor, mode, isothermal, n_global, box):
    """The geometry and content of a snipshot: everything needed to rebuild positions and energy."""
    nc = n_global // factor
    dxc = box / nc
    pos = (f"cell centres x_i = (i + 0.5) * {dxc!r}, i = 0..{nc - 1}, same for y, z"
           if mode == "mean" else
           f"fine cell centres x_i = ({factor} i + {factor // 2} + 0.5) * {box / n_global!r}")
    return {**header, "format": "luna-snipshot-1", "factor": factor, "mode": mode,
            "coarse_shape": [nc] * 3, "dx_coarse": dxc, "positions": pos,
            "fields": ["rho", "mom"] + ([] if isothermal else ["E"]),
            "energy": ("not stored: E = isothermal_cs2 rho / (gamma - 1) + |mom|^2 / (2 rho)"
                       if isothermal else "stored")}


def _attrs(obj, header):
    for k, v in header.items():
        obj.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v


def write_snipshot(Uc, header, out_dir):
    """The processes holding a slab write part.<pid>.h5; rank 0 adds snip.h5, one virtual global array per field."""
    os.makedirs(out_dir, exist_ok=True)
    pid = jax.process_index()
    owners = {}                                  # slab -> its writer, the replica on the lowest device id
    for dev, index in Uc.sharding.devices_indices_map(Uc.shape).items():
        key = tuple((i.start, i.stop) for i in index)
        if key not in owners or dev.id < owners[key].id:
            owners[key] = dev
    mine = [s for s in Uc.addressable_shards if s.device in owners.values()]
    gshape = Uc.shape[1:]
    fields = header["fields"]
    rows = {"rho": slice(0, 1), "mom": slice(1, 4), "E": slice(4, 5)}

    def write_part(tmp):
        with h5py.File(tmp, "w") as f:
            _attrs(f, header)
            for s in mine:
                g = f.create_group(f"shard_{s.device.id}")
                g.attrs["start"] = [int(i.start or 0) for i in s.index[1:]]
                data = np.asarray(s.data, dtype=np.float32)
                for name in fields:
                    block = data[rows[name]]
                    g.create_dataset(name, data=block[0] if name != "mom" else block)
    if mine:
        write_atomically(os.path.join(out_dir, f"part.{pid:05d}.h5"), write_part)
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices("snipshot_parts")
    if pid != 0:
        return
    def write_index(tmp):
        with h5py.File(tmp, "w") as f:
            _attrs(f, header)
            for name in fields:
                shape = (3,) + gshape if name == "mom" else gshape
                lay = h5py.VirtualLayout(shape=shape, dtype=np.float32)
                for dev in owners.values():
                    index = Uc.sharding.devices_indices_map(Uc.shape)[dev]
                    sl = tuple(slice(i.start or 0, i.stop if i.stop is not None else n)
                               for i, n in zip(index[1:], gshape))
                    bshape = tuple(s.stop - s.start for s in sl)
                    src = h5py.VirtualSource(f"part.{dev.process_index:05d}.h5",
                                             f"shard_{dev.id}/{name}",
                                             shape=((3,) + bshape) if name == "mom" else bshape)
                    lay[((slice(None),) + sl) if name == "mom" else sl] = src
                f.create_virtual_dataset(name, lay)
    write_atomically(os.path.join(out_dir, "snip.h5"), write_index)


def build_frames(mesh, gamma, box, n_global, axis, fields, ranges=None):
    """Return frames(U) -> {field: (uint16 image, [below, above] clipped counts)}, replicated."""
    rng = {k: FRAMES[k][0] for k in fields}
    rng.update(ranges or {})
    mesh_shape = tuple(int(mesh.shape[a]) for a in AXES)
    dx = box / n_global
    others = [d for d in range(3) if d != axis]
    mid = n_global // 2

    def _plane(f):
        """f in the mid-plane normal to `axis`, assembled along the line of sight."""
        n = f.shape[axis]
        owner = jax.lax.axis_index(AXES[axis]) == mid // n
        sl = jax.lax.index_in_dim(f, mid % n, axis=axis, keepdims=False)
        return jax.lax.psum(jnp.where(owner, sl, 0.0), AXES[axis])

    def _frames(U):
        rho, v, _ = primitives(U, gamma)
        rho_mean = jax.lax.psum(jnp.sum(rho), AXES) / float(n_global) ** 3
        out = {}
        for name in fields:
            if name == "column_density":
                x = jnp.log10(jax.lax.psum(jnp.sum(rho, axis=axis) * dx, AXES[axis]) / (rho_mean * box))
            elif name == "slice_s":
                x = jnp.log(_plane(rho) / rho_mean)
            elif name == "slice_log10_vorticity":
                omega, _ = vorticity_divergence(v, dx, mesh_shape, n_global)
                x = jnp.log10(jnp.sqrt(_plane(jnp.sum(omega * omega, axis=0))) + 1e-300)
            else:
                raise ValueError(f"frame field {name!r}: known are {sorted(FRAMES)}")
            lo, hi = rng[name]
            clipped = jnp.stack([jnp.sum(x < lo), jnp.sum(x > hi)]).astype(jnp.float32)
            q = jnp.round((jnp.clip(x, lo, hi) - lo) / (hi - lo) * 65535.0).astype(jnp.uint16)
            out[name] = (q, jax.lax.psum(clipped, tuple(AXES[d] for d in others)))
        return out

    spec = P(None, *AXES)
    img_spec = P(*(AXES[d] for d in others))
    local = shard_map(_frames, mesh=mesh, in_specs=(spec,),
                      out_specs={k: (img_spec, P()) for k in fields})
    replicated = NamedSharding(mesh, P())

    def _gathered(U):
        # one all-gather per image: every rank then holds the frame it may be asked to write
        return {k: (jax.lax.with_sharding_constraint(q, replicated), c) for k, (q, c) in local(U).items()}
    return jax.jit(_gathered), rng


def write_frame(path, image, header, png_max=0):
    """Write a uint16 frame with its header; optionally a colour-mapped quick-look PNG."""
    write_atomically(path, lambda tmp: _write_frame_h5(tmp, image, header))
    if png_max:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        step = max(1, int(np.ceil(image.shape[0] / png_max)))
        png = path[:-3] + ".png"
        write_atomically(png, lambda tmp: plt.imsave(tmp, image[::step, ::step].T, cmap="magma",
                                                     vmin=0, vmax=65535, origin="lower", format="png"))


def _write_frame_h5(tmp, image, header):
    with h5py.File(tmp, "w") as f:
        _attrs(f, header)
        f.create_dataset("image", data=image)
