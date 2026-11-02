"""Generate src/luna/chemistry/generated/net5_p{npix}.py and net5_core.py (one pixel, no LVG, per-component
lists: the fused kernel's input): network-5 f and its exact Jacobian as straight-line JAX.
    PYTHONPATH=src python tools/chem_sympy/gen_net5.py [--npix 1,12] [--stats out.json]   (SymPy only)"""
import argparse
import json
import os
import sys
import time

import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_net5 import Model, Opaque, Slope  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "src", "luna", "chemistry", "generated")
SHORT = {"tg": "tg", "chem_t": "ct", "cool_t": "kt", "cool_f": "cf", "co": "co", "opac": "op",
         "dust": "du"}


class JaxPrinter(PythonCodePrinter):
    """SymPy -> jax.numpy source; floats printed as exact doubles."""

    def _print_Float(self, e):
        return repr(float(e))

    def _print_Integer(self, e):
        return repr(float(int(e)))

    def _print_Rational(self, e):
        return repr(float(e.p) / float(e.q))

    def _print_Half(self, e):
        return "0.5"

    def _print_Infinity(self, e):
        return "jnp.inf"

    def _print_NegativeInfinity(self, e):
        return "(-jnp.inf)"

    def _print_Exp1(self, e):
        return "jnp.e"

    def _print_Pow(self, e, rational=False):
        b, x = e.args
        bs = self._print(b)
        if x == sp.S.Half:
            return f"jnp.sqrt({bs})"
        if x == -sp.S.Half:
            return f"(1.0 / jnp.sqrt({bs}))"
        if x == -1:
            return f"(1.0 / ({bs}))"
        if x.is_Integer:
            return f"({bs})**{int(x)}"
        return f"({bs})**({self._print(x)})"

    def _print_exp(self, e):
        return f"jnp.exp({self._print(e.args[0])})"

    def _print_log(self, e):
        return f"jnp.log({self._print(e.args[0])})"

    def _print_sign(self, e):
        return f"jnp.sign({self._print(e.args[0])})"

    def _print_Abs(self, e):
        return f"jnp.abs({self._print(e.args[0])})"

    def _print_jmax(self, e):
        return f"jnp.maximum({self._print(e.args[0])}, {self._print(e.args[1])})"

    def _print_jmin(self, e):
        return f"jnp.minimum({self._print(e.args[0])}, {self._print(e.args[1])})"

    def _print_gtw(self, e):
        return f"lk.gtw({self._print(e.args[0])}, {self._print(e.args[1])})"

    def _print_Piecewise(self, e):
        last = e.args[-1]
        if last.cond != sp.true:
            raise ValueError("Piecewise without a default branch")
        out = self._print(last.expr)
        for ex, c in reversed(e.args[:-1]):
            out = f"jnp.where({self._print(c)}, {self._print(ex)}, {out})"
        return out

    def _print_Where(self, e):
        a, b, c = (self._print(x) for x in e.args)
        return f"jnp.where({a}, {b}, {c})"

    def _print_ITE(self, e):
        a, b, c = (self._print(x) for x in e.args)
        return f"jnp.where({a}, {b}, {c})"

    def _print_Relational(self, e):
        op = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[e.rel_op]
        return f"({self._print(e.lhs)} {op} {self._print(e.rhs)})"

    def _print_And(self, e):
        return "(" + " & ".join(self._print(a) for a in e.args) + ")"

    def _print_Or(self, e):
        return "(" + " | ".join(self._print(a) for a in e.args) + ")"

    def _print_Not(self, e):
        return f"(~{self._print(e.args[0])})"

    def _print_BooleanTrue(self, e):
        return "True"

    def _print_BooleanFalse(self, e):
        return "False"


PR = JaxPrinter({"fully_qualified_modules": False})


def code(e):
    return PR._print(e)


def tree_size(exprs):
    """Nodes of the expression trees counted with repetition (the size without CSE), by memoised recursion."""
    memo = {}

    def size(e):
        if e in memo:
            return memo[e]
        s = 1 + sum(size(a) for a in e.args)
        memo[e] = s
        return s
    return sum(size(e) for e in exprs)


def lookup_calls(exprs):
    """Every opaque value / slope application in exprs, nested ones included."""
    out = set()
    for e in exprs:
        out |= {a for a in e.atoms(sp.Function) if isinstance(a, (Opaque, Slope))}
    return out


def plan(exprs):
    """Group the lookup calls into runtime calls; returns (instances in dependency order, call -> symbol)."""
    calls = lookup_calls(exprs)
    inst, names = {}, {}
    for c in sorted(calls, key=sp.default_sort_key):
        k = (c.group, c.args)
        if k not in inst:
            cnt = sum(1 for g, _ in inst if g == c.group)
            inst[k] = dict(group=c.group, args=c.args, name=f"{SHORT[c.group]}{cnt}", vals=set(),
                           slopes=set())
        d = inst[k]
        if isinstance(c, Slope):
            d["slopes"].add(c.key)
            names[c] = sp.Symbol(f"{d['name']}_{c.key[0]}_d{c.key[1]}", real=True)
        else:
            d["vals"].add(c.key)
            names[c] = sp.Symbol(f"{d['name']}_{c.key}", real=True)
    for d in inst.values():
        if d["group"] in ("tg", "co"):       # these return every value at once
            d["vals"] |= {k[0] for k in d["slopes"]}
        d["args_sub"] = [a.xreplace(names) for a in d["args"]]
    own = {d["name"]: d for d in inst.values()}
    order, done = [], set()
    while len(order) < len(own):
        for nm, d in own.items():
            if nm in done:
                continue
            deps = {s.name.split("_")[0] for a in d["args_sub"] for s in a.free_symbols}
            if all(x in done or x not in own or x == nm for x in deps):
                order.append(d)
                done.add(nm)
    return order, names


def emit_lookup(d, lines, core=False):
    """Source lines computing one runtime lookup's arguments and outputs."""
    nm = d["name"]
    reps, red = sp.cse(d["args_sub"], symbols=sp.numbered_symbols(f"_{nm}a"), order="none")
    for s, e in reps:
        lines.append(f"    {s} = {code(e)}")
    a = [code(e) for e in red]
    g, need = d["group"], bool(d["slopes"])
    if g == "tg":
        lines.append(f"    _r = lk.temp_gamma({a[0]}, {a[1]}, {need})")
        outs = ["T", "G"] + (["T_d0", "T_d1", "G_d0", "G_d1"] if need else [])
        lines.append("    " + ", ".join(f"{nm}_{o}" for o in outs) + " = _r")
    elif g in ("chem_t", "cool_t", "cool_f"):
        vk = tuple(sorted(d["vals"]))
        sk = tuple(sorted({k[0] for k in d["slopes"]}))
        lines.append(f"    _v, _s = lk.rows({g!r}, {a[0]}, {vk}, {sk})")
        lines += [f"    {nm}_{k} = _v[{k}]" for k in vk]
        lines += [f"    {nm}_{k}_d0 = _s[{k}]" for k in sk]
    elif g == "co":
        lines.append(f"    _v, _dt, _dn = lk.co({a[0]}, {a[1]}, {need})")
        lines += [f"    {nm}_{k} = _v[{k!r}]" for k in sorted(d["vals"])]
        for k, i in sorted(d["slopes"]):
            src = f"_dt[{k!r}]" if i == 0 else ("0.0" if k == "l0" else f"_dn[{k!r}]")
            lines.append(f"    {nm}_{k}_d{i} = {src}")
    elif g == "opac":
        lines.append(f"    _r = lk.h2_opacity({a[0]}, {a[1]}, {need})")
        lines.append(f"    {nm}_v, {nm}_v_d0, {nm}_v_d1 = _r" if need else f"    {nm}_v = _r")
    elif g == "dust":
        lines.append(f"    {nm}_td = lk.dust_td(n, {a[0]}, {a[1]}, chi, g0_dust, dgr, tdust_in, "
                     "h2_form_ex, h2_form_kin, dust_rtol" + (", dust_iters)" if core else ")"))
    else:
        raise ValueError(g)


def prologue(m, fname, doc):
    lines = [f"def {fname}(y, p):", f'    """{doc}"""',
             "    y = jnp.asarray(y, jnp.float64)",
             "    p = Params(*[jnp.asarray(v, jnp.float64) for v in p])",
             "    y0, y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]"]
    src = dict(n="n_h", tdust_in="tdust")
    lines += [f"    {k} = p.{src.get(k, k)}" for k in m.PARAMS]
    lines.append("    dust_rtol = p.dust_rtol")
    lines += [f"    nh2c{j} = p.n_h2_col[..., {j}]" for j in range(m.npix)]
    lines.append("    _ph = lk.photo_terms(p)")
    lines += [f"    {k} = _ph[{k!r}]" for k in m.PHOTO]
    return lines


def prologue_core(m, fname, doc):
    """Per-component inputs y0..y3 and q (Params fields, one pixel, no pixel axis)."""
    lines = [f"def {fname}(y0, y1, y2, y3, q, dust_iters=None):", f'    """{doc}"""']
    src = dict(n="n_h", tdust_in="tdust")
    lines += [f"    {k} = q.{src.get(k, k)}" for k in m.PARAMS if k != "divv"]
    lines.append("    dust_rtol = q.dust_rtol")
    lines.append("    nh2c0 = q.n_h2_col")
    lines.append("    _ph = lk.photo_terms_1(q)")
    lines += [f"    {k} = _ph[{k!r}]" for k in m.PHOTO]
    return lines


def epilogue_core(r, jac, lines):
    """f, J (lists of rows) and aux; zero where the state is invalid."""
    lines.append("    bad = lk.invalid(y0, y1, y2, y3)")
    lines.append("    _z = jnp.zeros_like(y0)")
    w = [f"jnp.where(bad, 0.0, {e.replace(' + jnp.zeros_like(y0)', ' + _z')})" for e in r]
    lines.append(f"    f = [{', '.join(w[:4])}]")
    k = 4
    if jac:
        rows = [f"[{', '.join(w[4 + 4 * i:8 + 4 * i])}]" for i in range(4)]
        lines.append(f"    J = [{', '.join(rows)}]")
        k = 20
    a = [e.replace(" + jnp.zeros_like(y0)", " + _z") for e in r[k:]]
    lines.append(f"    aux = {{'T': {a[0]}, 'gamma': {a[1]}, 'tdust': {a[2]}, "
                 f"'invalid': bad | ({a[3]} <= 0.0)}}")
    lines.append("    return (f, J, aux)" if jac else "    return (f, aux)")


def emit(m, fname, doc, jac, core=False):
    """One generated function and its statistics."""
    t0 = time.perf_counter()
    outs = list(m.f)
    if jac:
        outs += [fi.diff(yj) for fi in m.f for yj in m.y]
    t_diff = time.perf_counter() - t0
    outs += [m.aux["T"], m.aux["gamma"], m.aux["tdust"], m.t_raw]
    swell = tree_size(outs)
    order, names = plan(outs)
    lines = (prologue_core if core else prologue)(m, fname, doc)
    for d in order:
        emit_lookup(d, lines, core)
    subbed = [o.xreplace(names) for o in outs]
    t1 = time.perf_counter()
    reps, red = sp.cse(subbed, symbols=sp.numbered_symbols("t"), order="none")
    t_cse = time.perf_counter() - t1
    ops = sum(sp.count_ops(e) for _, e in reps) + sum(sp.count_ops(e) for e in red)
    for s, e in reps:
        lines.append(f"    {s} = {code(e)}")
    z = "jnp.zeros_like(y0)"
    r = [f"({code(e)}) + {z}" for e in red]
    stats = dict(cse_temporaries=len(reps), cse_ops=int(ops), tree_nodes_without_cse=int(swell),
                 lookups={d["name"]: dict(group=d["group"], values=len(d["vals"]),
                                          slopes=len(d["slopes"])) for d in order},
                 diff_s=t_diff, cse_s=t_cse)
    if core:
        epilogue_core(r, jac, lines)
        return lines, dict(stats, lines=len(lines))
    lines.append("    bad = _validate(y)[4]")
    lines.append(f"    f = jnp.where(bad[..., None], 0.0, jnp.stack([{', '.join(r[:4])}], -1))")
    k = 4
    if jac:
        rows = [f"jnp.stack([{', '.join(r[4 + 4 * i:8 + 4 * i])}], -1)" for i in range(4)]
        lines.append(f"    J = jnp.where(bad[..., None, None], 0.0, jnp.stack([{', '.join(rows)}], -2))")
        k = 20
    lines.append(f"    aux = {{'T': {r[k]}, 'gamma': {r[k + 1]}, 'tdust': {r[k + 2]}, "
                 f"'invalid': bad | ({r[k + 3]} <= 0.0)}}")
    lines.append("    return (f, J, aux)" if jac else "    return (f, aux)")
    return lines, dict(stats, lines=len(lines))


HEADER = '''"""Generated by tools/chem_sympy/gen_net5.py (SymPy {ver}): network 5, n_pix = {npix}. Do not edit.
f equals rates_net5.rhs (fixes="chemcool", eos="table"); J is its exact Jacobian d f_i / d y_j.
"""
# ruff: noqa: F841
import jax.numpy as jnp

from luna.chemistry.generated import lookups as lk
from luna.chemistry.rates_net5 import Params, _validate
'''


HEADER_CORE = '''"""Generated by tools/chem_sympy/gen_net5.py (SymPy {ver}): network 5, one pixel, divv = 0. Do not edit.
f equals rates_net5.rhs(lvg=False) per component; J is its exact Jacobian; inputs and outputs are lists.
"""
# ruff: noqa: F841
import jax.numpy as jnp

from luna.chemistry.generated import lookups as lk
'''


def generate_core():
    """Write net5_core.py (one pixel, divv = 0, lists in and out); returns the statistics per function."""
    stats, body = {}, []
    for dust in ("implicit", "frozen"):
        m = Model(1, dust, lvg=False)
        for jac in (False, True):
            fname = f"{'rhs_and_jac' if jac else 'rhs'}_{dust}"
            doc = (f"{'(f, J, aux)' if jac else '(f, aux)'} of network 5 without LVG, T_dust "
                   f"{'solved' if dust == 'implicit' else 'held at q.tdust'}, as lists.")
            lines, st = emit(m, fname, doc, jac, core=True)
            stats[fname] = st
            body += [""] + [""] + lines
            print("core", fname, {k: v for k, v in st.items() if k != "lookups"}, flush=True)
    path = os.path.join(OUT, "net5_core.py")
    with open(path, "w") as fh:
        fh.write(HEADER_CORE.format(ver=sp.__version__) + "\n".join(body) + "\n")
    stats["file_bytes"] = os.path.getsize(path)
    return stats


def generate(npix):
    """Write net5_p{npix}.py; returns the statistics per function."""
    stats, body = {}, []
    for dust in ("implicit", "frozen"):
        t0 = time.perf_counter()
        m = Model(npix, dust)
        t_build = time.perf_counter() - t0
        for jac in (False, True):
            fname = f"{'rhs_and_jac' if jac else 'rhs'}_{dust}"
            doc = (f"{'(f, J, aux)' if jac else '(f, aux)'} of network 5, T_dust "
                   f"{'solved, implicit derivative' if dust == 'implicit' else 'held at p.tdust'}.")
            t1 = time.perf_counter()
            lines, st = emit(m, fname, doc, jac)
            st.update(build_s=t_build, total_s=time.perf_counter() - t1 + t_build)
            stats[fname] = st
            body += [""] + [""] + lines
            print(fname, {k: v for k, v in st.items() if k != "lookups"}, flush=True)
    path = os.path.join(OUT, f"net5_p{npix}.py")
    with open(path, "w") as fh:
        fh.write(HEADER.format(ver=sp.__version__, npix=npix) + "\n".join(body) + "\n")
    stats["file_bytes"] = os.path.getsize(path)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npix", default="1,12")
    ap.add_argument("--stats", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()
    res = {f"npix_{k}": generate(int(k)) for k in a.npix.split(",") if k}
    res["core"] = generate_core()
    res["wall_s"] = time.perf_counter() - t0
    res["sympy"] = sp.__version__
    print(f"total {res['wall_s']:.1f} s", flush=True)
    if a.stats:
        json.dump(res, open(a.stats, "w"), indent=1)


if __name__ == "__main__":
    main()
