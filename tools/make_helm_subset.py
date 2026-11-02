"""Cut the Timmes Helmholtz table to the nodes the Microphysics reference tests read; every other node is NaN.

Same 541 x 201 grid and node values, so `_cell` and the Hermite weights are unchanged and every kept cell
evaluates bitwise as with the full table (checked by --verify); a state outside the kept cells reads NaN and
`evaluate` returns valid = False (loud, never a silent value). Kept: the box rho Ye in [1e5, 2e9] (CJ, ZND
and the He burn), every T, plus one node on each side, and the 2 x 2 nodes of every helmholtz_reference.json state.
The table's derivative entries are stored per node (Timmes), not differenced by LUNA, so cutting whole nodes
loses nothing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from luna.eos.helmholtz import HelmholtzTable, _cell, evaluate, load_table  # noqa: E402
from luna.reference_data import reference_path, reference_root  # noqa: E402

OUT = "tables/helmholtz/helm_table_subset.npz"   # in luna-reference-solutions (then run its make_manifest.sh)
BOX = dict(rho_ye=(1e5, 2e9), temperature=(1e3, 1e13))   # all T: the burn's T inversion brackets from 1e3


def node_mask(table):
    """(201, 541) bool: the nodes kept."""
    rho_n, T_n = np.asarray(table.density), np.asarray(table.temperature)
    mask = np.zeros((T_n.size, rho_n.size), bool)
    ri = np.where((rho_n >= BOX["rho_ye"][0]) & (rho_n <= BOX["rho_ye"][1]))[0]
    tj = np.where((T_n >= BOX["temperature"][0]) & (T_n <= BOX["temperature"][1]))[0]
    mask[max(tj[0] - 1, 0):tj[-1] + 2, max(ri[0] - 1, 0):ri[-1] + 2] = True
    ref = np.asarray(json.loads(Path(reference_path("microphysics/helmholtz_eos/helmholtz_reference.json")).read_text())["inputs"])
    i = np.asarray(_cell(table.density, jnp.asarray(ref[:, 0] * ref[:, 3] / ref[:, 2]))[0])
    j = np.asarray(_cell(table.temperature, jnp.asarray(ref[:, 1]))[0])
    for a, b in zip(i, j):
        mask[b:b + 2, a:a + 2] = True
    return mask


def subset(table, mask):
    """HelmholtzTable with the unkept nodes NaN."""
    m = jnp.asarray(mask)
    return table._replace(free=jnp.where(m[..., None, None], table.free, jnp.nan),
                          dpdr=jnp.where(m[..., None, None], table.dpdr, jnp.nan))


def load_subset(path=None):
    """The subset as a HelmholtzTable (the grid exactly as luna.eos.helmholtz.load_table builds it)."""
    d = np.load(path or reference_root() / OUT)
    return HelmholtzTable(jnp.asarray(np.logspace(-12, 15, 541)), jnp.asarray(np.logspace(3, 13, 201)),
                          jnp.asarray(d["free"]), jnp.asarray(d["dpdr"]))


def verify(full, sub, n=20000, seed=0):
    """Bitwise: evaluate on random box states and the reference states, and the ten CJ solves."""
    from luna.nuclear.detonation import prescribed_ash_cj
    rng = np.random.default_rng(seed)
    lr = rng.uniform(np.log(2 * BOX["rho_ye"][0]), np.log(BOX["rho_ye"][1]), n)
    lt = rng.uniform(np.log(BOX["temperature"][0]), np.log(BOX["temperature"][1]), n)
    comps = [(4.0, 2.0), (56.0, 28.0), (13.714285714285714, 6.857142857142857)]
    worst = nvalid = 0
    for coulomb in (False, True):
        for A, Z in comps:
            args = (np.exp(lr) * A / Z, np.exp(lt), A, Z)
            a, b = (evaluate(t, *args, coulomb=coulomb) for t in (full, sub))
            for x, y in zip(a, b):
                x, y = np.asarray(x), np.asarray(y)
                same = (x == y) | (np.isnan(x) & np.isnan(y)) if x.dtype.kind == "f" else x == y
                worst += int(np.sum(~same))
            nvalid += int(np.asarray(a.valid).sum())
    ref = json.loads(Path(reference_path("microphysics/helmholtz_eos/helmholtz_reference.json")).read_text())
    inp = np.asarray(ref["inputs"])
    for coulomb in (0, 1):
        r = inp[inp[:, -1] == coulomb]
        a, b = (evaluate(t, *r[:, :4].T, coulomb=bool(coulomb)) for t in (full, sub))
        worst += sum(int(np.sum(np.asarray(x) != np.asarray(y))) for x, y in zip(a, b))
    print(f"evaluate: {worst} differing values over {6 * n} box states ({nvalid} valid on the full table) and "
          f"{len(inp)} reference states")
    cj = json.loads(Path(reference_path("microphysics/detonation_cj/detonation_cj_reference.json")).read_text())
    diff = 0
    for row in cj["inputs"]:
        a, b = (prescribed_ash_cj(t, row["density"], row["temperature"], row["fuel"], row["ash"],
                                  coulomb=row["coulomb"]) for t in (full, sub))
        diff += int(tuple(a) != tuple(b))
    print(f"prescribed_ash_cj: {diff} of {len(cj['inputs'])} solves differ")
    return worst + diff


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("table", type=Path)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raw = a.table.read_bytes()
    full = load_table(a.table)
    mask = node_mask(full)
    sub = subset(full, mask)
    prov = dict(source=str(a.table), sha256=hashlib.sha256(raw).hexdigest(), md5=hashlib.md5(raw).hexdigest(),
                box=BOX, kept_nodes=int(mask.sum()), generator="tools/make_helm_subset.py")
    np.savez_compressed(reference_root() / OUT, free=np.asarray(sub.free), dpdr=np.asarray(sub.dpdr), mask=mask,
                        provenance=json.dumps(prov))
    print(json.dumps(prov))
    if a.verify:
        bad = verify(full, load_subset())
        print("VERIFY", "bitwise identical" if bad == 0 else f"{bad} differences")


if __name__ == "__main__":
    main()
