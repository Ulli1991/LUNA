"""Forward- and reverse-mode derivatives of LUNA across processes and nodes.

    srun ... python tools/adj_multinode.py probe problem=jeans nx=64 mesh=2,2,1
    srun -n1 python tools/adj_multinode.py ref   problem=jeans nx=128 out=ref.npz
    srun ... python tools/adj_multinode.py check problem=jeans nx=128 mesh=4,2,1 ref=ref.npz
    srun ... python tools/adj_multinode.py bench problem=hydro nx=512 mesh=4,2,1 ops=fwd,jvp
    srun ... python tools/adj_multinode.py coll

probe  each program (forward, JVP, VJP) called `calls` times, a line before
       and after every call: a hang or crash shows as the last "start".
ref    one device: the rollout, its gradient, J^T w (and J v) to an .npz.
check  the same on the mesh, then one CHECK PASS|FAIL line: N devices vs 1,
       <J v, w> vs <v, J^T w> (hydro), d(mass)/dU = (1,0,0,0,0).
bench  forward / JVP / VJP time, first-call time, peak memory per device.
coll   no LUNA: a ring ppermute + psum over every device.

Problems: `jeans` and `cooling` are scripts/examples/autodiff/ic_inversion_run's
(self-gravity; OU turbulence + cooling), `hydro` is the jeans box without
gravity. The state gets fixed broadband perturbations of rho, v and p
(state=flat: rho only, a static contact mode). FP64, minmod, fixed dt.
Extra `Field=value` pairs go to the Config. Launch recipe and numbers:
docs/AUTODIFF.md, "Across nodes".
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOL = 1.0e-11          # round-off in FP64; the CPU-mesh tests use the same bound


def _args(argv):
    kv = dict(a.split("=", 1) for a in argv if "=" in a)
    p = dict(problem="jeans", nx="64", n_steps="10", mesh="1,1,1", calls="3",
             ops="fwd,jvp,vjp", reps="2", out="", ref="", amp="0.05", ou="1",
             state="generic")
    for k in list(kv):
        if k in p:
            p[k] = kv.pop(k)
    return p, kv


def _config(p, extra):
    from dataclasses import replace
    from luna.setup.cli import apply, overrides
    from scripts.examples.autodiff.ic_inversion_run import config
    nx = int(p["nx"])
    base = "jeans" if p["problem"] == "hydro" else p["problem"]
    cfg = config(base, nx=nx, n_steps=int(p["n_steps"]))
    cfg = replace(cfg, LIMITER=0, enable_x64=True,
                  MESH_SHAPE=tuple(int(m) for m in p["mesh"].split(",")),
                  write_snapshots=False, enable_diagnostics=False,
                  enable_checkpointing=False)
    if p["problem"] == "hydro":
        cfg = replace(cfg, SELF_GRAVITY=False)
    if p["ou"] == "0":                       # cooling without the OU driving
        cfg = replace(cfg, ou_config=None)
    return apply(cfg, overrides(extra))


def _perturbation(shape, amp, seed=1):
    """A fixed broadband log-density field, numpy, the same on every process."""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftn(rng.standard_normal(shape))
    k = np.sqrt(sum(np.meshgrid(*[np.fft.fftfreq(n) ** 2 for n in shape[:-1]],
                                np.fft.rfftfreq(shape[-1]) ** 2, indexing="ij")))
    f *= np.exp(-(k / 0.25) ** 2)                 # smooth at the grid scale
    d = np.fft.irfftn(f, s=shape, axes=(0, 1, 2))
    return amp * d / d.std()


def build(p, extra):
    """(cfg, f, U0): f(U) is the differentiable rollout of the production step."""
    import jax
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    from luna.autodiff.rollout import build_simulate
    cfg = resolve(_config(p, extra))
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    jmesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, jmesh, kind, layout, False)
    physics = assemble(cfg, jmesh, kind, layout, init, False)
    U = init.U
    # A generic state: rho and p times exp(delta), v plus delta x c_s, five
    # independent fields built on the host (identical on every process and
    # every layout) and placed shard by shard. A density perturbation alone
    # is a static contact mode: the rollout would not move.
    amp, g = float(p["amp"]), float(cfg.gamma)
    d = np.stack([_perturbation(U.shape[1:], amp, seed=1 + i) for i in range(5)])
    if p["state"] == "flat":                 # rho alone (the first runs' state)
        d[1:] = 0.0
    dS = jax.make_array_from_callback(d.shape, U.sharding, lambda idx: d[idx])

    def generic(U, d):
        jnp = jax.numpy
        rho = U[0]
        v = U[1:4] / rho
        pr = (g - 1.0) * (U[4] - 0.5 * rho * jnp.sum(v ** 2, axis=0))
        cs = jnp.sqrt(g * pr / rho)
        rho, pr, v = rho * jnp.exp(d[0]), pr * jnp.exp(d[4]), v + d[1:4] * cs
        return jnp.concatenate([rho[None], rho * v,
                                (pr / (g - 1.0) + 0.5 * rho * jnp.sum(v ** 2, axis=0))[None]])
    U0 = jax.jit(generic, out_shardings=U.sharding)(U, dS)
    simulate, _, _, _, theta = build_simulate(init, cfg, physics, False)
    return cfg, (lambda U: simulate(theta, U)), U0


def _host_array(a, sharding):
    import jax
    return jax.make_array_from_callback(a.shape, sharding, lambda idx: a[idx])


def _gather(x):
    """A (sharded, maybe multi-process) array as one numpy array everywhere."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    rep = NamedSharding(x.sharding.mesh, P())
    return np.asarray(jax.jit(lambda a: a, out_shardings=rep)(x))


def _peak():
    import jax
    try:
        return int((jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use", -1))
    except Exception:
        return -1


def _all_peaks():
    """Every device's high-water mark, in GiB (a collective)."""
    from jax.experimental import multihost_utils
    v = multihost_utils.process_allgather(np.array([_peak()], np.int64))
    return [x / 2 ** 30 for x in np.ravel(v)]


def _say(*a):
    import jax
    if jax.process_index() == 0:
        print(*a, flush=True)


def _host_vectors(shape, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape) for _ in range(2)]


def _vectors(U0, seed):
    """Random v, w of U0's shape and sharding, from the host (layout-free)."""
    return [_host_array(a, U0.sharding) for a in _host_vectors(U0.shape, seed)]


def _fsum_dot(a, b):
    import math
    return math.fsum((np.asarray(a) * np.asarray(b)).ravel())


def _forward_mode(p):
    """The gravity solve and the cooling rate are custom_vjp functions (an
    implicit adjoint; a gather-free slope): JAX has no JVP through them, so
    forward mode is the hydro-only problem's."""
    return p["problem"] == "hydro"


def _programs(f):
    import jax
    import jax.numpy as jnp

    def loss(U):
        return 0.5 * jnp.mean(f(U) ** 2)

    fwd = jax.jit(loss)
    jvp = jax.jit(lambda U, v: jax.jvp(f, (U,), (v,))[1])
    vjp = jax.jit(lambda U, w: jax.vjp(f, U)[1](w)[0])
    grad = jax.jit(jax.value_and_grad(loss))
    mass = jax.jit(jax.grad(lambda U: jnp.sum(f(U)[0])))
    return fwd, jvp, vjp, grad, mass


# --------------------------------------------------------------------------

def probe(p, extra):
    import jax
    cfg, f, U0 = build(p, extra)
    fwd, jvp, vjp, grad, _ = _programs(f)
    v, w = _vectors(U0, 3)
    progs = {"fwd": lambda: fwd(U0), "jvp": lambda: jvp(U0, v),
             "vjp": lambda: vjp(U0, w), "grad": lambda: grad(U0)}
    _say(f"PROBE problem={p['problem']} nx={cfg.Nx} steps={p['n_steps']} "
         f"mesh={cfg.MESH_SHAPE} devices={jax.device_count()} "
         f"processes={jax.process_count()} jax={jax.__version__} "
         f"HLB={os.environ.get('HIP_LAUNCH_BLOCKING')} "
         f"XLA_FLAGS={os.environ.get('XLA_FLAGS')!r}")
    ops = [o for o in p["ops"].split(",") if _forward_mode(p) or o != "jvp"]
    for op in ops:
        for k in range(int(p["calls"])):
            _say(f"  start {op} call {k}")
            t0 = time.perf_counter()
            jax.block_until_ready(progs[op]())
            _say(f"  done  {op} call {k}  {time.perf_counter() - t0:.3f} s")
    _say(f"PROBE OK peak GiB per device {['%.1f' % g for g in _all_peaks()]}")


def collective(p, extra):
    """No LUNA: one ring ppermute + psum over every device, `calls` times.
    Tells a launch-mode problem of the collectives from one of the solver."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.utils.shmap import shard_map
    n = jax.device_count()
    mesh = Mesh(np.array(jax.devices()), ("x",))
    perm = [(i, (i + 1) % n) for i in range(n)]

    def body(a):
        b = jax.lax.ppermute(a, "x", perm)
        return b + jax.lax.psum(jnp.sum(a), "x")

    f = jax.jit(shard_map(body, mesh=mesh, in_specs=P("x"), out_specs=P("x")))
    a = jax.jit(lambda: jnp.ones((n * 2 ** 20,)),
                out_shardings=NamedSharding(mesh, P("x")))()
    _say(f"COLL devices={n} processes={jax.process_count()} "
         f"HLB={os.environ.get('HIP_LAUNCH_BLOCKING')}")
    for k in range(int(p["calls"])):
        _say(f"  start coll call {k}")
        t0 = time.perf_counter()
        a = jax.block_until_ready(f(a) / n)
        _say(f"  done  coll call {k}  {time.perf_counter() - t0:.3f} s")
    _say("COLL OK")


def ref(p, extra):
    import jax
    cfg, f, U0 = build(p, extra)
    fwd, jvp, vjp, grad, _ = _programs(f)
    v, w = _vectors(U0, 3)
    _, g = grad(U0)
    out = dict(U0=np.asarray(U0), UT=np.asarray(jax.jit(f)(U0)), grad=np.asarray(g),
               JTw=np.asarray(vjp(U0, w)))
    if _forward_mode(p):
        out["Jv"] = np.asarray(jvp(U0, v))
    np.savez(p["out"], **out)
    print(f"REF written {p['out']} nx={cfg.Nx} steps={p['n_steps']}", flush=True)


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def check(p, extra):
    """N devices vs 1, the transpose identity and the mass identity; one
    PASS/FAIL line. The identities are summed on the device (sharded)."""
    import jax
    import jax.numpy as jnp
    cfg, f, U0 = build(p, extra)
    fwd, jvp, vjp, grad, mass = _programs(f)
    v, w = _vectors(U0, 3)
    t0 = time.perf_counter()
    _, g = grad(U0)
    JTw = vjp(U0, w)
    gm = mass(U0)
    one = jnp.zeros_like(gm).at[0].set(1.0)
    err_mass = float(jax.jit(lambda a, b: jnp.max(jnp.abs(a - b)))(gm, one))
    res = {"mass": err_mass}
    arrays = {"U0": U0, "UT": jax.jit(f)(U0), "grad": g, "JTw": JTw}
    if _forward_mode(p):
        Jv = jvp(U0, v)
        dot = jax.jit(lambda a, b: jnp.vdot(a, b))
        lhs, rhs = float(dot(Jv, w)), float(dot(v, JTw))
        arrays["Jv"] = Jv
    host = {k: _gather(a) for k, a in arrays.items()}
    t = time.perf_counter() - t0
    ok = True
    if jax.process_index() == 0:
        print(f"CHECK problem={p['problem']} nx={cfg.Nx} steps={p['n_steps']} "
              f"mesh={cfg.MESH_SHAPE} devices={jax.device_count()} "
              f"processes={jax.process_count()} jax={jax.__version__} ({t:.1f} s)")
        print(f"  max |d(mass)/dU - (1,0,0,0,0)| = {res['mass']:.2e}")
        if "Jv" in arrays:
            # <Jv,w> vs <v,J^T w> is judged with exactly rounded sums (fsum on
            # the host): a device vdot's own rounding is eps x the condition
            # number sum|a_i b_i| / |sum a_i b_i| of the dot (1e5 was measured
            # for random v, w), which is the dot's error, not the adjoint's
            vh, wh = _host_vectors(host["U0"].shape, 3)
            lx, rx = _fsum_dot(host["Jv"], wh), _fsum_dot(vh, host["JTw"])
            res["transpose"] = abs(lx - rx) / max(abs(lx), abs(rx))
            cond = float(np.sum(np.abs(host["Jv"] * wh)) / abs(lx))
            print(f"  <Jv,w>={lx:.15e}  <v,J^T w>={rx:.15e}  rel {res['transpose']:.2e}"
                  f"  (exact sums; condition {cond:.1e}; device vdot: rel "
                  f"{abs(lhs - rhs) / max(abs(lhs), abs(rhs)):.2e})")
            if p["ref"] and "Jv" in np.load(p["ref"]):
                r = np.load(p["ref"])
                l1, r1 = _fsum_dot(r["Jv"], wh), _fsum_dot(vh, r["JTw"])
                print(f"  1 device, exact sums: rel {abs(l1 - r1) / max(abs(l1), abs(r1)):.2e}")
        if p["ref"]:
            r = np.load(p["ref"])
            res["U0 vs 1 device"] = _rel(host["U0"], r["U0"])
            for k in (k for k in ("UT", "grad", "Jv", "JTw") if k in arrays):
                res[f"{k} vs 1 device"] = _rel(host[k], r[k])
                print(f"  {k:4s} N devices vs 1: max|diff|/max|ref| = {res[f'{k} vs 1 device']:.2e}")
        bad = {k: x for k, x in res.items() if not x < TOL}
        ok = not bad
        print(("CHECK PASS" if ok else f"CHECK FAIL {bad}")
              + f" (tolerance {TOL:.0e}, all: "
              + ", ".join(f"{k}={x:.2e}" for k, x in res.items()) + ")", flush=True)
    return ok


def bench(p, extra):
    import jax
    import statistics
    cfg, f, U0 = build(p, extra)
    fwd, jvp, vjp, grad, _ = _programs(f)
    key = jax.random.PRNGKey(3)
    v = jax.jit(lambda k: jax.random.normal(k, U0.shape, U0.dtype),
                out_shardings=U0.sharding)(key)
    progs = {"fwd": lambda: fwd(U0), "jvp": lambda: jvp(U0, v),
             "grad": lambda: grad(U0)}
    _say(f"BENCH problem={p['problem']} nx={cfg.Nx} steps={p['n_steps']} "
         f"mesh={cfg.MESH_SHAPE} devices={jax.device_count()} "
         f"processes={jax.process_count()} jax={jax.__version__}")
    times = {}
    for op in p["ops"].split(","):
        t0 = time.perf_counter()
        jax.block_until_ready(progs[op]())
        c = time.perf_counter() - t0
        ts = []
        for _ in range(int(p["reps"])):
            t0 = time.perf_counter()
            jax.block_until_ready(progs[op]())
            ts.append(time.perf_counter() - t0)
        times[op] = statistics.median(ts)
        pk = _all_peaks()
        _say(f"  {op:4s} {times[op]:.3f} s (first call {c:.1f} s, reps {['%.3f' % t for t in ts]}) "
             f"peak GiB so far max {max(pk):.1f} per device {['%.1f' % g for g in pk]}")
    if "fwd" in times:
        for op in ("jvp", "grad"):
            if op in times:
                _say(f"  ratio {op}/fwd = {times[op] / times['fwd']:.2f}")
    _say("BENCH OK")


def main(argv):
    mode = argv[0]
    p, extra = _args(argv[1:])
    import jax
    if mode in ("probe", "check", "bench", "coll"):
        if os.environ.get("JAX_PLATFORMS", "").startswith("cpu"):
            # CPU processes need a cross-process collective library
            jax.config.update("jax_cpu_collectives_implementation", "gloo")
        from luna.setup.jax_init import initialize_jax_distributed
        initialize_jax_distributed(enable_x64=True)
    jax.config.update("jax_enable_x64", True)
    ok = {"probe": probe, "ref": ref, "check": check, "bench": bench,
          "coll": collective}[mode](p, extra)
    return 0 if ok in (None, True) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
