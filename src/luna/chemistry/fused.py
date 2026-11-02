"""Network 5's fixed-structure hybrid solve (integrate_fixed) in fused Pallas (Triton) kernels, FP64: three per substep.
Same math as the plain-JAX solve (generated f and J, LU, Rodas3, error); its gradient is the plain one (custom_vjp)."""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from luna.chemistry import generated, rates_net5
from luna.ode import rodas3 as ig
from luna.chemistry import tables as tb

BLOCK = 64          # cells per program (one per lane of a 64-wide wavefront)
WARPS = 1           # wavefronts per program


class _View:
    """A table inside the packed buffer, indexed like the array it stands for."""

    def __init__(self, ref, off, arr):
        self.ref, self.off, self.np = ref, off, arr
        self.shape, self.size, self.ndim = arr.shape, arr.size, arr.ndim

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        if all(isinstance(i, (int, np.integer)) for i in idx):
            return float(self.np[idx])                  # a constant
        flat = 0
        for d, i in enumerate(idx):
            i = i + self.shape[d] if isinstance(i, (int, np.integer)) and i < 0 else i
            flat = flat * self.shape[d] + i
        return self.ref[self.off + flat]


@functools.lru_cache(maxsize=None)
def _tables(core):
    """The tables a solve with this net5_core reads: (names, tree, numpy leaves, offsets, packed buffer)."""
    seen = {}
    tb._SEEN = seen
    try:
        with jax.ensure_compile_time_eval():         # one cell, evaluated: the tables are concrete
            z = jnp.zeros(())
            q = rates_net5.Params(*[z + 1.0 if f not in ("abundsi", "dl", "divv") else z
                                    for f in rates_net5.Params._fields])
            core._replace(squeezed=True).rhs_and_jac([z + 1e-3] * 3 + [z + 1e-12], q)
    finally:
        tb._SEEN = None
    names = tuple(sorted(seen))
    leaves, tree = jax.tree_util.tree_flatten([seen[k] for k in names])
    arrs = [np.asarray(v, np.float64) for v in leaves]
    for v, a in zip(leaves, arrs):
        if not np.array_equal(np.asarray(v), a):
            raise TypeError("a table is not exactly representable in FP64")
    offs = np.cumsum([0] + [a.size for a in arrs])
    return names, tree, arrs, tuple(int(o) for o in offs[:-1]), np.concatenate([a.ravel() for a in arrs])


class _kernel_tables:
    """While a kernel is traced: every device_table is its view of the packed buffer ref."""

    def __init__(self, core, ref):
        names, tree, arrs, offs, _ = _tables(core)
        views = [_View(ref, o, a) for o, a in zip(offs, arrs)]
        self.views = dict(zip(names, tree.unflatten(views)))

    def __enter__(self):
        tb._KERNEL_VIEWS = self.views

    def __exit__(self, *exc):
        tb._KERNEL_VIEWS = None


def _core(net):
    """The net5_core a network's lists are, or an error."""
    if not isinstance(net.lists, generated.Core):
        raise TypeError("the fused kernel runs network 5 from net5_core (sources.chemistry.fixed_net)")
    return net.lists


N_OUT = {"jac": 22, "stages": 13, "finish": 6}


def _piece(name, knet, xs, p, h):
    """One kernel's share of a substep of h on per-cell values xs: the list of its outputs."""
    lists = knet.lists
    y, rest = xs[:4], xs[4:]
    if name == "jac":                               # y, warm start -> f0, J, hidden, invalid
        f0, jac, hid, b0 = ig.rosenbrock_jac(lists, knet, y, p._replace(tdust=rest[0]))
        return f0 + [v for r in jac for v in r] + [hid["tdust"], b0]
    if name == "stages":                            # y, f0, J, hidden -> k1, k2, k3, invalid
        f0, jac, td = rest[:4], [rest[4 + 4 * i:8 + 4 * i] for i in range(4)], rest[20]
        ks, b1 = ig.rosenbrock_stages(lists, knet, y, f0, jac, p._replace(tdust=td), h)
        return [v for kk in ks for v in kk] + [b1]
    if name == "finish":                            # y, J, k's, hidden, emax, invalid -> y, emax, invalid
        jac = [rest[4 * i:4 + 4 * i] for i in range(4)]
        ks = [rest[16 + 4 * i:20 + 4 * i] for i in range(3)]
        td, emax, bad = rest[28:]
        y1, err, b2 = ig.rosenbrock_finish(lists, knet, y, jac, ks, p._replace(tdust=td), h)
        return y1 + [jnp.maximum(emax, err), bad | b2]
    raise ValueError(name)


@functools.lru_cache(maxsize=None)
def _call(net, name, n, n_x, kinds, interpret, mat):
    """pallas_call of one piece over n cells: inputs (n_x per-cell values, cell leaves, shared, tables)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    if not interpret:
        from luna.radiation.m1_pallas import select_triton
        select_triton()
    core = _core(net)
    knet = net._replace(lists=core._replace(squeezed=True))
    tree, cut = kinds
    n_cell = sum(cut)

    def kernel(*refs):
        x_refs, c_refs = refs[:n_x], refs[n_x:n_x + n_cell]
        s_ref, t_ref, outs = refs[n_x + n_cell], refs[n_x + n_cell + 1], refs[n_x + n_cell + 2:]
        sl = pl.ds(pl.program_id(0) * BLOCK, BLOCK)
        with _kernel_tables(core, t_ref):
            xs = [r[sl] for r in x_refs]
            cells, leaves, shared = iter([r[sl] for r in c_refs]), [], 0
            for c in cut:
                if c:
                    leaves.append(next(cells))
                else:
                    leaves.append(s_ref[shared])
                    shared += 1
            p, h = jax.tree_util.tree_unflatten(tree, leaves)
            p = p._replace(**{f: v + 0.0 * xs[0] for f, v in net.converge})
            if name == "finish":                    # the invalid flag travels as 0 / 1
                xs[-1] = xs[-1] > 0.5
            out = _piece(name, knet, xs, p, h)
        for r, v in zip(outs, out):
            r[sl] = jnp.where(v, 1.0, 0.0) + 0.0 * xs[0] if v.dtype == jnp.bool_ else v

    shape = jax.ShapeDtypeStruct((n,), jnp.float64, manual_axis_type=mat)   # mat: inside a shard_map
    return pl.pallas_call(kernel, out_shape=(shape,) * N_OUT[name], grid=(n // BLOCK,),
                          interpret=interpret, name=f"chem_{name}",
                          compiler_params=plt.CompilerParams(num_warps=WARPS, num_stages=1))


def _strong(x):
    from luna.radiation.m1_pallas import _strong as s
    return s(x)


def substeps(net, k, y0, p, dt, interpret=False):
    """k substeps for every cell of y0 (N, 4) in the kernels: (y, T_dust, max error norm, invalid)."""
    n = y0.shape[0]
    y0, p, h = ig.pad_cells((y0, p, jnp.asarray(dt, jnp.float64) / k), n, BLOCK)   # as fixed_substeps
    npad = y0.shape[0]
    leaves, tree, cut = ig._split((p, h), npad)
    for v, c in zip(leaves, cut):
        if c and v.size != npad:
            raise ValueError("per-cell leaves must hold one value per cell (one TreeCol pixel)")
    cells = [_strong(v.reshape(npad)) for v, c in zip(leaves, cut) if c]
    shared = [v.reshape(()) for v, c in zip(leaves, cut) if not c]
    shared = _strong(jnp.stack(shared) if shared else jnp.zeros((1,)))
    tab = _strong(jnp.asarray(_tables(_core(net))[4]))
    kinds = (tree, tuple(cut))
    mat = getattr(jax.typeof(y0), "mat", None)

    def run(name, xs):
        return list(_call(net, name, npad, len(xs), kinds, interpret, mat)(*xs, *cells, shared, tab))

    def sub(c):
        y, td, emax, bad = c
        a = run("jac", y + [td])
        f0, jac, td = a[:4], a[4:20], a[20]
        b = run("stages", y + f0 + jac + [td])
        bad = jnp.maximum(bad, jnp.maximum(a[21], b[12]))
        o = run("finish", y + jac + b[:12] + [td, emax, bad])
        return o[:4], td, o[4], o[5]

    z = jnp.zeros((npad,))
    c = ([_strong(y0[:, j]) for j in range(4)], _strong(p.tdust + z), z, z)
    for _ in range(k):
        c = sub(c)
    y, td, emax, bad = c
    return jnp.stack(y, -1)[:n], td[:n], emax[:n], bad[:n] > 0.5


def _forward(net, opt, y0, p, dt, interpret=False):
    """Both passes in the kernels: (y, hidden fields, flagged), as integrate._fixed_impl."""
    n = y0.shape[0]
    y, td, emax, bad = substeps(net, opt.k1, y0, p, dt, interpret)
    flag = (emax > opt.flag_err) | bad
    idx, used, m2 = ig._flagged_chunks(opt, flag)

    def body(j, c):
        sl = jax.lax.dynamic_slice(idx, (j * m2,), (m2,))
        y2, t2, _, _ = substeps(net, opt.k2, *ig._gather((y0, p, dt), n, sl), interpret)
        put = lambda a, v: a.at[sl].set(v, mode="drop")  # noqa: E731
        return put(c[0], y2), put(c[1], t2)
    y, td = jax.lax.fori_loop(0, used, body, (y, td))
    return y, {"tdust": td}, flag


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def solve(net, opt, y0, p, dt):
    """integrate_fixed's solve in the kernels (pass as integrate_fixed(..., solve=solve))."""
    return _forward(net, opt, y0, p, dt)


def _fwd(net, opt, y0, p, dt):
    out = _forward(net, opt, y0, p, dt)
    return out, (y0, p, dt, out[2])


def _bwd(net, opt, res, ct):
    return ig.fixed_vjp(net, opt, *res, ct[0], ct[1])


solve.defvjp(_fwd, _bwd)
