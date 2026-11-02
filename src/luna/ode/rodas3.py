"""Batched, differentiable stiff integrators for per-cell ODEs y' = f(y; p): chemistry, nuclear burning.
Rodas3 (Sandu et al. 1997): adaptive per cell, or fixed structure (integrate_fixed); docs/CHEMISTRY.md.
"""
import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


# Rodas3 (KPP's form): (I/(h g) - J) k_i = f(y + sum_j A_ij k_j) + sum_j C_ij k_j / h,
# y1 = y + sum_i M_i k_i, error estimate k_4 (E = (0, 0, 0, 1)); order 3(2), L-stable.
GAMMA = 0.5
A = ((), (0.0,), (2.0, 0.0), (2.0, 0.0, 1.0))
C = ((), (4.0,), (1.0, -1.0), (1.0, -1.0, -8.0 / 3.0))
M = (2.0, 0.0, 1.0, 1.0)


class Network(NamedTuple):
    """What the integrator needs to know about a network besides its RHS."""
    rhs: Callable                 # rhs(y, p) -> (ydot, aux) for one cell
    rtol: float                   # DVODE's RTOL
    atol: tuple                   # DVODE's ATOL per component (energy last)

    safety: tuple                 # evolve_abundances' first-step factors
    ymax: Callable                # p -> upper bound per component (inf: none)
    ctrl_floor: Callable          # p -> components below this don't set the first step

    hidden: tuple = ()            # Params fields the RHS returns in aux (a warm start, carried)
    converge: tuple = ()          # (field, value) pairs that make the RHS solve them fully
    lists: object = None          # .rhs, .rhs_and_jac on lists of components (integrate_fixed)


class Options(NamedTuple):
    max_steps: int = 2000             # attempted substeps per cell before `failed`
    tiers: tuple = ((2, 1), (6, 2), (16, 8), (48, 32), (None, 128))  # (attempt cap, 1/chunk fraction)
    diff_fields: tuple = None         # Params fields with gradients (None: all but hidden)

    rtol: float = None                # override the network's rtol
    atol: tuple = None                # override the network's atol
    skip_empty_tiers: bool = True     # a tier runs only the chunks holding unfinished cells

    k1: int = 1                       # integrate_fixed: substeps for every cell
    k2: int = 4                       # integrate_fixed: substeps for the flagged cells
    flag_err: float = 0.1             # integrate_fixed: flag above this error norm (or invalid)

    chunk: int = 2 ** 20              # integrate_fixed: cells per first-pass batch
    chunk2: int = 2 ** 16             # integrate_fixed: flagged cells per second-pass batch


class Result(NamedTuple):
    y: jnp.ndarray          # (N, nspec) state at t0 + dt (or where the cell stopped)
    params: object          # Params with the hidden state (T_dust) updated
    h: jnp.ndarray          # step size to start the next call with

    t: jnp.ndarray          # time reached (== dt unless failed)
    n_steps: jnp.ndarray    # accepted substeps
    n_rejected: jnp.ndarray  # rejected attempts (error, physical bounds, invalid state)

    failed: jnp.ndarray     # hit max_steps or a zero step: y is NOT the answer


# --------------------------------------------------------------------------
# small dense linear algebra, unrolled (fuses on the GPU, any n)
# --------------------------------------------------------------------------
def lu_factor(a):
    """LU with partial pivoting of one (n, n) matrix: (packed LU, permutation)."""
    n = a.shape[-1]
    idx = jnp.arange(n)
    perm = idx
    for k in range(n):
        piv = jnp.argmax(jnp.where(idx >= k, jnp.abs(a[:, k]), -1.0))
        sel = idx == piv
        row_p = jnp.sum(jnp.where(sel[:, None], a, 0.0), 0)
        row_k = a[k]
        a = jnp.where((idx == k)[:, None], row_p[None], jnp.where(sel[:, None], row_k[None], a))
        perm_p = jnp.sum(jnp.where(sel, perm, 0))
        perm = jnp.where(idx == k, perm_p, jnp.where(sel, perm[k], perm))
        below = idx > k
        mult = jnp.where(below, a[:, k] / a[k, k], 0.0)
        a = a - mult[:, None] * jnp.where(idx > k, a[k], 0.0)[None, :]
        a = jnp.where((idx == k)[None, :] & below[:, None], mult[:, None], a)
    return a, perm


def lu_solve(lu, b):
    """Solve with lu_factor's output; b is (n,) or (n, m)."""
    a, perm = lu
    n = a.shape[-1]
    onehot = (perm[:, None] == jnp.arange(n)[None, :]).astype(a.dtype)
    x = onehot @ b
    rows = [x[i] for i in range(n)]
    for i in range(n):
        for j in range(i):
            rows[i] = rows[i] - a[i, j] * rows[j]
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            rows[i] = rows[i] - a[i, j] * rows[j]
        rows[i] = rows[i] / a[i, i]
    return jnp.stack(rows)


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------
def _zero_like(x):
    """0.0 computed from x, so a loop carry varies over the mesh axes as x does (shard_map)."""
    return jnp.ravel(x)[0] * 0.0


def _rms(v):
    """Root mean square, summed in index order (no reduction op: same bits for any batch shape)."""
    s = v[0] * v[0]
    for i in range(1, v.shape[0]):
        s = s + v[i] * v[i]
    return jnp.sqrt(s / v.shape[0])


def _error_norm(err, w):
    """Weighted RMS norm of the embedded error estimate."""
    return _rms(err / w)


def _with_hidden(net, p, aux):
    return p._replace(**{k: aux[k] for k in net.hidden})


def _invalid(aux):
    return aux.get("invalid", jnp.asarray(False))


def _f_and_jac(net, y, p):
    """f(y), aux and the Jacobian d f / d y in one forward-mode pass."""
    def fa(yy):
        f, aux = net.rhs(yy, p)
        return f, (f, aux)
    jac, (f, aux) = jax.jacfwd(fa, has_aux=True)(y)
    return (f, aux), jac


def _rodas3(net, y, f0, jac, p, h, w):
    """One Rodas3 step from y: (y1, error estimate, any stage invalid)."""
    n = y.shape[0]
    hg = h * GAMMA
    lu = lu_factor((jnp.eye(n) - hg * jac) * (1.0 / w)[:, None] * w[None, :])   # D^-1 M D
    ks, bad, f = [], jnp.asarray(False), f0
    for i in range(4):
        if i >= 2:                                   # stage 2 sits at y (A_21 = 0): reuse f0
            f, aux = net.rhs(y + sum(a * k for a, k in zip(A[i], ks)), p)
            bad = bad | _invalid(aux)
        r = f + sum(c * k for c, k in zip(C[i], ks)) / h
        ks.append(lu_solve(lu, hg * r / w) * w)
    return y + sum(m * k for m, k in zip(M, ks)), ks[3], bad


def _check(net, rtol, atol, y, p):
    """evolve_abundances' check_abundances: (y clipped, ok, which were clipped)."""
    ymax = net.ymax(p)
    ymax = jnp.stack(ymax, -1) if isinstance(ymax, (list, tuple)) else ymax
    ok = jnp.all(jnp.isfinite(y)) & jnp.all(y >= -atol)
    ok = ok & jnp.all((y <= ymax) | (y - ymax < rtol * y))
    clipped = (y < 0.0) | (y > ymax)
    return jnp.clip(y, 0.0, ymax), ok, clipped


def _first_step(net, y, f0, p, dt):
    """evolve_abundances' first step: min of safety * y / |ydot| above the control floor."""
    use = (f0 != 0.0) & (y > net.ctrl_floor(p))
    est = jnp.where(use, jnp.asarray(net.safety) * y / jnp.where(use, jnp.abs(f0), 1.0), 1e18)
    return jnp.minimum(jnp.min(est), dt)


def _cell(net, opt, sens, y0, p, dt, h0):
    """Integrate one cell over dt. With sens: also d y1 / d(y0, theta)."""
    rtol = net.rtol if opt.rtol is None else opt.rtol
    atol = jnp.asarray(net.atol if opt.atol is None else opt.atol)
    n = y0.shape[0]
    fields = _diff_fields(net, opt, p)
    th0, unravel = ravel_pytree({k: getattr(p, k) for k in fields})
    m = n + th0.shape[0]

    def body(c):
        t, y, pc, h, nacc, nrej, failed, done, s = c
        (f0, aux0), jac = _f_and_jac(net, y, pc)
        ps = _with_hidden(net, pc, aux0)
        h = jnp.where(h > 0.0, h, _first_step(net, y, f0, pc, dt))
        last = h >= dt - t
        he = jnp.where(last, dt - t, h)
        scale = jnp.maximum(rtol * jnp.abs(y) + atol, 1e-300)       # conditions the LU
        y1, err_vec, bad = _rodas3(net, y, f0, jac, ps, he, scale)
        w = jnp.maximum(rtol * jnp.maximum(jnp.abs(y), jnp.abs(y1)) + atol, 1e-300)
        err = _error_norm(err_vec, w)
        ynew, phys, clipped = _check(net, rtol, atol, y1, ps)
        sane = phys & ~bad & ~_invalid(aux0)
        acc = sane & (err <= 1.0)
        fac = jnp.clip(0.9 * jnp.maximum(err, 1e-10) ** (-1.0 / 3.0), 0.2,
                       jnp.where(acc, 5.0, 1.0))
        fac = jnp.where(sane, fac, 0.5)
        hnew = jnp.where(acc & last, jnp.maximum(he * fac, h), he * fac)
        if sens:
            def step(yy, th):
                pt = ps._replace(**unravel(th))
                (ff, _), jj = _f_and_jac(net, yy, pt)
                return _rodas3(net, yy, ff, jj, pt, he, scale)[0]
            dy, dth = jax.jacfwd(step, argnums=(0, 1))(y, th0)
            snew = dy @ s + jnp.concatenate([jnp.zeros((n, n)), dth], 1)
            s = jnp.where(acc, jnp.where(clipped[:, None], 0.0, snew), s)
        t = jnp.where(acc, jnp.where(last, dt, t + he), t)
        y = jnp.where(acc, ynew, y)
        nacc, nrej = nacc + acc, nrej + ~acc
        failed = ((nacc + nrej >= opt.max_steps) | (t + hnew == t)) & ~(acc & last)
        return t, y, ps, hnew, nacc, nrej, failed, (acc & last) | failed, s

    p = p._replace(**{k: jnp.asarray(v, jnp.float64) for k, v in net.converge})
    zero = _zero_like(y0)
    dt, h0 = dt + zero, h0 + zero
    s0 = jnp.eye(n, m) + zero if sens else zero
    izero = zero.astype(jnp.int32)
    c0 = (zero, y0, p, h0, izero, izero, zero > 1.0, dt <= 0.0, s0)
    t, y, pc, h, nacc, nrej, failed, _, s = jax.lax.while_loop(lambda c: ~c[7], body, c0)
    fend, aux = net.rhs(y, pc)
    return (y, _with_hidden(net, pc, aux), h, t, nacc, nrej, failed), (s, fend)


def _diff_fields(net, opt, p):
    if opt.diff_fields is not None:
        return tuple(opt.diff_fields)
    skip = set(net.hidden) | {k for k, _ in net.converge}
    return tuple(k for k in p._fields if k not in skip and getattr(p, k) is not None)


# --------------------------------------------------------------------------
# public entry: tiers of passes in the primal, one sensitivity pass under differentiation
# --------------------------------------------------------------------------
def broadcast_params(p, n):
    """Every Params leaf with a leading cell axis of length n (others are broadcast)."""
    def b(v):
        if v is None:                  # an unset optional field (network 17's photo state)
            return None
        v = jnp.asarray(v, jnp.float64)
        return v if (v.ndim > 0 and v.shape[0] == n) else jnp.broadcast_to(v, (n,) + v.shape)
    return type(p)(*[b(v) for v in p])


@functools.partial(jax.jit, static_argnames=("net", "opt"))
def integrate(y0, p, dt, h0=None, *, net, opt=Options()):
    """Integrate every cell of y0 (N, nspec) over dt [s] (Params leaves (N, ...) or scalars); differentiable."""
    y0 = jnp.asarray(y0, jnp.float64)
    n = y0.shape[0]
    p = broadcast_params(p, n)
    dt = jnp.broadcast_to(jnp.asarray(dt, jnp.float64), (n,))
    h0 = jnp.zeros((n,)) if h0 is None else jnp.broadcast_to(jnp.asarray(h0, jnp.float64), (n,))
    return Result(*_batch(net, opt, y0, p, dt, h0))


def _solve(net, opt, y0, p, dt, h0):
    """One cell, no sensitivities: the outputs of a Result."""
    return _restore(net, p, _cell(net, opt, False, y0, p, dt, h0)[0])


def _restore(net, p_in, out):
    """The caller's Params with only the hidden fields updated."""
    y, pout, *rest = out
    pout = p_in._replace(**{k: getattr(pout, k) for k in net.hidden})
    return (y, pout, *rest)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _batch(net, opt, y0, p, dt, h0):
    n = y0.shape[0]
    tiers = _tiers(opt, n)
    r = Result(*jax.vmap(functools.partial(_solve, net, opt._replace(max_steps=tiers[0][0])))(
        y0, p, dt, h0))
    for cap, m in tiers[1:]:
        o = opt._replace(max_steps=cap)
        if opt.skip_empty_tiers:
            r = Result(*jax.lax.cond(jnp.any(r.failed),
                                     lambda r, o=o, m=m: tuple(_stragglers(net, o, m, Result(*r), dt)),
                                     lambda r: r, tuple(r)))
        else:
            r = _stragglers(net, o, m, r, dt)
    return tuple(r)


def _tiers(opt, n):
    """(attempt cap, cells per chunk) of each pass: all cells, then the unfinished in smaller chunks."""
    out, used = [], 0
    for cap, frac in opt.tiers:
        cap = opt.max_steps - used if cap is None else min(cap, opt.max_steps - used)
        if cap > 0:
            out.append((cap, max(1, -(-n // frac))))
        used += cap
    return out


def _stragglers(net, opt, m, r, dt):
    """Continue the unfinished cells of r, gathered into chunks of m cells (no work for the rest)."""
    n = r.y.shape[0]
    idx = jnp.nonzero(r.failed, size=n, fill_value=n)[0]
    solve = jax.vmap(functools.partial(_solve, net, opt))

    def chunk(r, k):
        sl = jax.lax.dynamic_slice(idx, (k * m,), (m,))
        valid = sl < n
        g = lambda a: a[jnp.minimum(sl, n - 1)]  # noqa: E731
        y, p, h, t, ns, nr, fl = solve(g(r.y), jax.tree_util.tree_map(g, r.params),
                                       jnp.where(valid, g(dt) - g(r.t), 0.0), g(r.h))
        put = lambda a, v: a.at[sl].set(v, mode="drop")  # noqa: E731
        return Result(put(r.y, y), jax.tree_util.tree_map(put, r.params, p), put(r.h, h),
                      put(r.t, g(r.t) + t), put(r.n_steps, g(r.n_steps) + ns),
                      put(r.n_rejected, g(r.n_rejected) + nr), put(r.failed, fl)), None

    if not opt.skip_empty_tiers:
        return jax.lax.scan(chunk, r, jnp.arange(-(-n // m)))[0]
    used = (jnp.sum(r.failed.astype(jnp.int32)) + m - 1) // m       # chunks holding work
    return jax.lax.fori_loop(0, used, lambda k, r: chunk(r, k)[0], r)


def _batch_fwd(net, opt, y0, p, dt, h0):
    """Under differentiation: one pass per cell, carrying d y1 / d(y0, theta)."""
    def one(y0, p, dt, h0):
        out, (s, fend) = _cell(net, opt, True, y0, p, dt, h0)
        return _restore(net, p, out), (s, fend, p, out[6])
    return jax.vmap(one)(y0, p, dt, h0)


def _batch_bwd(net, opt, res, ct):
    def one(s, fend, p, failed, ybar, pout_bar):
        n = ybar.shape[0]
        u = ybar @ s
        _, unravel = ravel_pytree({k: getattr(p, k) for k in _diff_fields(net, opt, p)})
        # the output Params are the input ones (hidden fields aside): their cotangent passes through
        pbar = pout_bar._replace(**{k: jnp.zeros_like(getattr(p, k)) for k in net.hidden})
        pbar = pbar._replace(**{k: getattr(pbar, k) + v for k, v in unravel(u[n:]).items()})
        return u[:n], pbar, jnp.where(failed, 0.0, jnp.sum(ybar * fend)), jnp.zeros(())
    return jax.vmap(one)(*res, ct[0], ct[1])


_batch.defvjp(_batch_fwd, _batch_bwd)


# --------------------------------------------------------------------------
# fixed structure: k1 Rodas3 substeps for every cell, k2 for the flagged ones; no rejection
# --------------------------------------------------------------------------
class FixedResult(NamedTuple):
    y: jnp.ndarray          # (N, nspec) state at t0 + dt
    params: object          # Params with the hidden state (T_dust) updated
    flagged: jnp.ndarray    # (N,) the cells that took the k2 substeps


def lu_factor_lists(a):
    """lu_factor on a matrix given as rows of per-cell values (lists of lists): the same pivots and arithmetic."""
    n = len(a)
    a = [list(r) for r in a]
    perm = [jnp.full(jnp.shape(a[0][0]), i, jnp.int32) for i in range(n)]
    for k in range(n):
        best, piv = jnp.abs(a[k][k]), jnp.full(jnp.shape(a[k][k]), k, jnp.int32)
        for i in range(k + 1, n):                   # argmax: the first of the largest
            take = jnp.abs(a[i][k]) > best
            best, piv = jnp.where(take, jnp.abs(a[i][k]), best), jnp.where(take, i, piv)
        row_k, perm_k = a[k], perm[k]
        a[k] = [_pick(piv, [a[i][j] for i in range(n)], k) for j in range(n)]
        perm[k] = _pick(piv, perm, k)
        for i in range(k + 1, n):
            a[i] = [jnp.where(piv == i, row_k[j], a[i][j]) for j in range(n)]
            perm[i] = jnp.where(piv == i, perm_k, perm[i])
        for i in range(k + 1, n):
            mult = a[i][k] / a[k][k]
            a[i] = [a[i][j] - mult * (a[k][j] if j > k else 0.0) for j in range(n)]
            a[i][k] = mult
    return a, perm


def _pick(piv, vals, k):
    """vals[piv] for piv >= k, as selects."""
    out = vals[k]
    for i in range(k + 1, len(vals)):
        out = jnp.where(piv == i, vals[i], out)
    return out


def lu_solve_lists(lu, b):
    """lu_solve for lu_factor_lists' output and a right-hand side given as a list of per-cell values."""
    a, perm = lu
    n = len(a)
    rows = [_pick(perm[i], b, 0) for i in range(n)]
    for i in range(n):
        for j in range(i):
            rows[i] = rows[i] - a[i][j] * rows[j]
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            rows[i] = rows[i] - a[i][j] * rows[j]
        rows[i] = rows[i] / a[i][i]
    return rows


def _lin(coef, vecs, n):
    """sum_i coef_i vecs_i per component, as Python's sum (from 0, in order)."""
    return [sum(c * v[j] for c, v in zip(coef, vecs)) for j in range(n)]


def _scale(net, y):
    """The LU's and stages' scale per component: max(rtol |y| + atol, 1e-300)."""
    return [jnp.maximum(net.rtol * jnp.abs(y[j]) + net.atol[j], 1e-300) for j in range(len(y))]


def _lu(jac, h, w):
    """LU of D^-1 (I / (h gamma) - J) D h gamma, D = diag(w), as _rodas3 forms it."""
    n = len(w)
    hg = h * GAMMA
    return lu_factor_lists([[((1.0 if i == j else 0.0) - hg * jac[i][j]) * (1.0 / w[i]) * w[j]
                             for j in range(n)] for i in range(n)])


def _stage(lu, c, ks, f, h, w):
    """One stage increment k_i from f at the stage's state and the earlier k's."""
    n = len(w)
    ck = _lin(c, ks, n)
    r = [f[j] + ck[j] / h for j in range(n)]
    sol = lu_solve_lists(lu, [h * GAMMA * r[j] / w[j] for j in range(n)])
    return [sol[j] * w[j] for j in range(n)]


def _rhs_at(lists, y, a, ks, p):
    """f at y + sum_j a_j k_j and whether that state is invalid."""
    n = len(y)
    ya = _lin(a, ks, n)
    f, aux = lists.rhs([y[j] + ya[j] for j in range(n)], p)
    return f, _invalid(aux)


def rosenbrock_jac(lists, net, y, p):
    """f and J at y, the hidden fields they leave, and whether y is invalid."""
    f0, jac, aux0 = lists.rhs_and_jac(y, p)
    return f0, jac, {key: aux0[key] for key in net.hidden}, _invalid(aux0)


def rosenbrock_stages(lists, net, y, f0, jac, p, h):
    """Rodas3's k_1, k_2, k_3 (stage 2 reuses f0: A_21 = 0) and whether stage 3's state is invalid."""
    w = _scale(net, y)
    lu = _lu(jac, h, w)
    ks = [_stage(lu, C[0], [], f0, h, w)]
    ks.append(_stage(lu, C[1], ks, f0, h, w))
    f, bad = _rhs_at(lists, y, A[2], ks, p)
    ks.append(_stage(lu, C[2], ks, f, h, w))
    return ks, bad


def rosenbrock_finish(lists, net, y, jac, ks, p, h):
    """Stage 4 and the step: (y1 clipped to its bounds, embedded error norm, invalid)."""
    n = len(y)
    rtol, atol = net.rtol, net.atol
    w = _scale(net, y)
    f, bad = _rhs_at(lists, y, A[3], ks, p)
    ks = ks + [_stage(_lu(jac, h, w), C[3], ks, f, h, w)]
    step = _lin(M, ks, n)
    y1 = [y[j] + step[j] for j in range(n)]
    e = ks[3]
    w = [jnp.maximum(rtol * jnp.maximum(jnp.abs(y[j]), jnp.abs(y1[j])) + atol[j], 1e-300)
         for j in range(n)]
    s = (e[0] / w[0]) * (e[0] / w[0])
    for j in range(1, n):                           # _rms, in index order
        s = s + (e[j] / w[j]) * (e[j] / w[j])
    ym = net.ymax(p)
    mx = list(ym) if isinstance(ym, (list, tuple)) else [ym[..., j] for j in range(n)]
    ok = y1[0] == y1[0]
    for j in range(n):                              # _check
        ok = ok & jnp.isfinite(y1[j]) & (y1[j] >= -atol[j])
        ok = ok & ((y1[j] <= mx[j]) | (y1[j] - mx[j] < rtol * y1[j]))
    return [jnp.clip(y1[j], 0.0, mx[j]) for j in range(n)], jnp.sqrt(s / n), bad | ~ok


def fixed_substeps(lists, net, k, y, p, dt):
    """k Rodas3 substeps of dt/k from y (a list of values per cell): (y, hidden fields, max error norm, invalid)."""
    h = dt / k

    def sub(c):
        y, hid, emax, bad = c
        f0, jac, hid, b0 = rosenbrock_jac(lists, net, y, p._replace(**hid))
        ph = p._replace(**hid)
        ks, b1 = rosenbrock_stages(lists, net, y, f0, jac, ph, h)
        y, err, b2 = rosenbrock_finish(lists, net, y, jac, ks, ph, h)
        return y, hid, jnp.maximum(emax, err), bad | b0 | b1 | b2

    z = 0.0 * y[0]
    c = (y, {key: getattr(p, key) + z for key in net.hidden}, z, z > 1.0)
    if k == 1:
        c = sub(c)
    else:
        c = jax.lax.fori_loop(0, k, lambda i, c: sub(c), c)
    y, hid, emax, bad = c
    return y, hid, emax, bad


class _Stacked(NamedTuple):
    """Network.lists from net.rhs on one cell's stacked state, J by jacfwd."""
    net: Network

    def rhs(self, y, p):
        f, aux = self.net.rhs(jnp.stack(y), p)
        return [f[j] for j in range(len(y))], aux

    def rhs_and_jac(self, y, p):
        (f, aux), jac = _f_and_jac(self.net, jnp.stack(y), p)
        n = len(y)
        return [f[j] for j in range(n)], [[jac[i, j] for j in range(n)] for i in range(n)], aux


def _lists(net):
    """The network's list form: its own, or net.rhs with jacfwd."""
    return _Stacked(net) if net.lists is None else net.lists


def fixed_cell(net, k, y0, p, dt):
    """One cell, k substeps from y0 (nspec,): (y, hidden fields, max error norm, invalid)."""
    p = p._replace(**{f: jnp.asarray(v, jnp.float64) for f, v in net.converge})
    y, aux, emax, bad = fixed_substeps(_lists(net), net, k, [y0[j] for j in range(y0.shape[0])],
                                       p, dt)
    return jnp.stack(y), {key: aux[key] for key in net.hidden}, emax, bad


def _split(args, n):
    """The leaves of args, their tree, and which leaves carry a leading axis of n cells."""
    leaves, tree = jax.tree_util.tree_flatten(args)
    return leaves, tree, [jnp.ndim(v) > 0 and jnp.shape(v)[0] == n for v in leaves]


def _vcell(net, k, y0, p, dt):
    """fixed_cell over a batch of cells (per-cell leaves of p, dt mapped, the others shared)."""
    leaves, tree, cut = _split((p, dt), y0.shape[0])
    axes = tree.unflatten([0 if c else None for c in cut])
    return jax.vmap(functools.partial(fixed_cell, net, k), in_axes=(0,) + tuple(axes))(
        y0, *tree.unflatten(leaves))


def _map_cells(fn, n, m, args):
    """fn(*args) over n // m chunks of m cells (lax.map); outputs stacked over the chunks."""
    leaves, tree, cut = _split(args, n)
    split = [v.reshape((n // m, m) + v.shape[1:]) for v, c in zip(leaves, cut) if c]

    def one(chunk):
        it = iter(chunk)
        return fn(*tree.unflatten([next(it) if c else v for v, c in zip(leaves, cut)]))
    return jax.lax.map(one, split)


def _cells(a, n):
    """A stacked lax.map output (chunks, m, ...) as (n, ...)."""
    return jax.tree_util.tree_map(lambda v: v.reshape((n,) + v.shape[2:]), a)


def _flagged_chunks(opt, flag):
    """Flagged cell indices padded with n to whole chunks of chunk2, and the number of chunks in use."""
    n = flag.shape[0]
    m2 = min(opt.chunk2, n)
    size = -(-n // m2) * m2
    # not jnp.nonzero: its bincount scatter-adds every unflagged cell onto a few slots (atomics)
    pos = jnp.cumsum(flag.astype(jnp.int32)) - 1
    idx = jnp.full(size, n, jnp.int32).at[jnp.where(flag, pos, size)].set(
        jnp.arange(n, dtype=jnp.int32), mode="drop")
    used = (pos[-1] + m2) // m2
    return idx, used, m2


def _gather(args, n, sl):
    """The per-cell leaves of args at sl; unused slots (sl = n) take distinct real cells, dropped later."""
    leaves, tree, cut = _split(args, n)
    gi = jnp.where(sl < n, sl, jnp.arange(sl.shape[0]))
    return tree.unflatten([v[gi] if c else v for v, c in zip(leaves, cut)])


def _fixed_impl(net, opt, y0, p, dt):
    """Both passes: (y, hidden fields, flagged)."""
    n = y0.shape[0]
    m = min(opt.chunk, n)
    y, hid, emax, bad = _cells(_map_cells(
        lambda *a: _vcell(net, opt.k1, *a), n, m, (y0, p, dt)), n)
    flag = (emax > opt.flag_err) | bad
    idx, used, m2 = _flagged_chunks(opt, flag)

    def body(j, c):
        sl = jax.lax.dynamic_slice(idx, (j * m2,), (m2,))
        y2, h2, _, _ = _vcell(net, opt.k2, *_gather((y0, p, dt), n, sl))
        put = lambda a, v: a.at[sl].set(v, mode="drop")  # noqa: E731
        return put(c[0], y2), jax.tree_util.tree_map(put, c[1], h2)
    y, hid = jax.lax.fori_loop(0, used, body, (y, hid))
    return y, hid, flag


def fixed_vjp(net, opt, y0, p, dt, flag, ybar, hbar):
    """Cotangents of (y0, p, dt) from those of (y, hidden): each cell's VJP through its k1 or k2 substeps."""
    n = y0.shape[0]
    m = min(opt.chunk, n)
    _, _, cut = _split((y0, p, dt), n)

    def cell_vjp(y0c, pc, dtc, yb, hb, k):
        F = lambda *a: _vcell(net, k, *a)[:2]  # noqa: E731
        return jax.vjp(F, y0c, pc, dtc)[1]((yb, hb))

    keep = ~flag
    yb1 = jnp.where(keep[:, None], ybar, 0.0)
    hb1 = jax.tree_util.tree_map(lambda b: jnp.where(keep, b, 0.0), hbar)
    g = _map_cells(lambda *a: cell_vjp(*a, opt.k1), n, m, (y0, p, dt, yb1, hb1))
    gl, gtree = jax.tree_util.tree_flatten(g)
    gl = [v.reshape((n,) + v.shape[2:]) if c else jnp.sum(v, 0) for v, c in zip(gl, cut)]
    idx, used, m2 = _flagged_chunks(opt, flag)

    def body(j, gl):
        sl = jax.lax.dynamic_slice(idx, (j * m2,), (m2,))
        valid = sl < n
        y0g, pg, dtg, yb, hb = _gather((y0, p, dt, ybar, hbar), n, sl)
        yb = jnp.where(valid[:, None], yb, 0.0)
        hb = jax.tree_util.tree_map(lambda b: jnp.where(valid, b, 0.0), hb)
        g2 = jax.tree_util.tree_leaves(cell_vjp(y0g, pg, dtg, yb, hb, opt.k2))
        return [a.at[sl].add(v, mode="drop") if c else a + v for a, v, c in zip(gl, g2, cut)]
    gl = jax.lax.fori_loop(0, used, body, gl)
    return gtree.unflatten(gl)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _fixed(net, opt, y0, p, dt):
    return _fixed_impl(net, opt, y0, p, dt)


def _fixed_fwd(net, opt, y0, p, dt):
    out = _fixed_impl(net, opt, y0, p, dt)
    return out, (y0, p, dt, out[2])


def _fixed_bwd(net, opt, res, ct):
    return fixed_vjp(net, opt, *res, ct[0], ct[1])


_fixed.defvjp(_fixed_fwd, _fixed_bwd)


def pad_cells(args, n, m):
    """args with every per-cell leaf padded (last cell repeated) to a whole number of m-cell chunks."""
    pad = -(-n // m) * m - n
    if pad == 0:
        return args
    leaves, tree, cut = _split(args, n)
    return tree.unflatten([jnp.concatenate([v, jnp.repeat(v[-1:], pad, 0)]) if c else v
                           for v, c in zip(leaves, cut)])


@functools.partial(jax.jit, static_argnames=("net", "opt", "solve"))
def integrate_fixed(y0, p, dt, *, net, opt=Options(), solve=None):
    """Every cell over dt in opt.k1 Rodas3 substeps, the flagged ones again in opt.k2 (docs/CHEMISTRY.md)."""
    net = net._replace(rtol=net.rtol if opt.rtol is None else opt.rtol,
                       atol=net.atol if opt.atol is None else tuple(opt.atol))
    y0 = jnp.asarray(y0, jnp.float64)
    n = y0.shape[0]
    p = type(p)(*[None if v is None else jnp.asarray(v, jnp.float64) for v in p])
    dt = jnp.asarray(dt, jnp.float64)
    y0p, pp, dtp = pad_cells((y0, p, dt), n, min(opt.chunk, n))
    y, hid, flag = (solve or _fixed)(net, opt, y0p, pp, dtp)
    hid = {k: v[:n] for k, v in hid.items()}
    return FixedResult(y[:n], p._replace(**hid), flag[:n])
