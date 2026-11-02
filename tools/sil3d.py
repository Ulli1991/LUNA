"""Solver in the loop at scale: a 3D CNN flux corrector for driven isothermal
turbulence, trained by reverse mode through long coarse rollouts.

    srun ... python tools/sil3d.py ref     out=DIR nx=1024 f=4 mach=5 mesh=4,2,1
    srun ... python tools/sil3d.py train   ref=DIR out=DIR mesh=2,2,2 unroll=8,32,128
    srun ... python tools/sil3d.py offline ref=DIR out=DIR mesh=2,2,2
    srun ... python tools/sil3d.py eval    ref=DIR weights=W.npz n0=... steps=...
    srun ... python tools/sil3d.py cost    nx=256 mesh=1,1,1 unroll=16

The problem (`config`): a periodic unit box, rho0 = cs = 1, isothermal by
LUNA's projection, solenoidal OU forcing at k = 1..2 tuned to an rms Mach
number. The fine run steps adaptively and lands on every coarse step dt_c;
there it records the OU amplitudes and, where asked, writes the state pooled
f^3 (conservative). The coarse run replays the recorded amplitudes, so both
feel the same driving. Minmod, FP64. docs/AUTODIFF.md, "Solver in the loop
at scale".
"""
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CS, RHO0, GAMMA = 1.0, 1.0, 5.0 / 3.0
SPEC = ("x", "y", "z")


# ---------------------------------------------------------------------------
# the problem
# ---------------------------------------------------------------------------

def config(nx, mach, seed=7, mesh=(1, 1, 1), acc_coef=2.0, **extra):
    """Driven isothermal turbulence at rms Mach ~`mach` on an nx^3 box.
    t_turn = L / (2 v_rms); the OU correlation time is t_turn and the
    acceleration acc_coef v_rms^2 / L (calibrated: docs/AUTODIFF.md)."""
    from luna.hydro.turbulence import OUDriverConfig
    from luna.ic_generation import driven_turbulence_3d
    from luna.template_config import Config
    v = mach * CS
    t_turn = 0.5 / v
    ou = OUDriverConfig(k_min=1.0, k_max=2.0, solenoidal_weight=1.0,
                        autocorrelation_time=t_turn, rms_acceleration=acc_coef * v * v,
                        seed=seed, spectrum="parabolic", Lx=1.0, Ly=1.0, Lz=1.0)
    return replace(
        Config(), test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=1.0, Ly=1.0, Lz=1.0, gamma=GAMMA,
        problem=driven_turbulence_3d.Params(rho0=RHO0, P0=RHO0 * CS * CS), isothermal_cs2=CS * CS,
        ORDER=1, LIMITER=0, enable_x64=True, MESH_SHAPE=tuple(mesh),
        ou_config=ou, ou_seed=seed, autodiff_mode=True,
        write_snapshots=False, enable_diagnostics=False, enable_checkpointing=False,
        PLOT_AFTER_SIM=False, **extra)


def t_turn(mach):
    return 0.5 / (mach * CS)


def assemble_problem(cfg, is_main=False):
    """(cfg, init, physics): the production build of `cfg`."""
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    cfg = resolve(cfg)
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    jmesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, jmesh, kind, layout, False)
    physics = assemble(cfg, jmesh, kind, layout, init, is_main)
    return cfg, init, physics


class Replay:
    """The OU kick with recorded amplitudes: the carry holds the table A of
    the amplitudes at every coarse step, A[state.step] before the hydro and
    A[state.step + 1] after (the same trapezoid as the live driver)."""
    name = "forcing"

    def __init__(self, kick):
        self._kick = kick

    def _a(self, state, k):
        import jax
        return jax.lax.dynamic_index_in_dim(state.sources[self.name], k, 0, False)

    def before(self, state, h, dt):
        return state.update(U=self._kick(state.U, self._a(state, state.step), h))

    def after(self, state, h, dt):
        return state.update(U=self._kick(state.U, self._a(state, state.step + 1), h))


def coarse_problem(cfg, meta, depth, is_main=False):
    """(cfg, init, with_params): the production coarse step with the live
    forcing swapped for `Replay` and the corrector stage last (innermost)."""
    from luna.setup.assemble import build_corrector
    from luna.sources.ml_corrector import cnn3d_init
    import jax
    cfg, init, physics = assemble_problem(cfg, is_main)
    live = next(s for s in physics.stages if s.name == "forcing")
    dummy = cnn3d_init(jax.random.PRNGKey(0), depth=depth)
    corr = build_corrector(cfg, physics.mesh, is_main, params=dummy, meta=meta)
    base = [Replay(live._kick) if s is live else s for s in physics.stages]

    def with_params(theta):
        return replace(physics, stages=base + [corr.with_params(theta)])

    return cfg, init, with_params


def state0(physics, init, U0, table):
    """The rollout's opening State: U0, t = 0, step 0, the forcing table."""
    import jax.numpy as jnp
    from luna.setup.evolve import initial_state
    st = initial_state(replace(init, U=U0), physics)
    return st.update(t=jnp.asarray(0.0, U0.dtype), a=jnp.asarray(1.0, U0.dtype),
                     step=jnp.asarray(0, jnp.int32)).with_source("forcing", table)


def observed(U):
    """(4, ...) what the loss compares: ln rho, v / cs."""
    import jax.numpy as jnp
    return jnp.concatenate([jnp.log(U[0] / RHO0)[None], U[1:4] / U[0] / CS])


def make_loss(with_params, init, *, n_steps, every, dt, norm, ckpt="revolve", snaps=16):
    """loss(theta, U0, targets, table): the mean over j = 1..n_steps/every of
    |obs(U_{j every}) - targets[j]|^2 / norm (norm = var ln rho, var v per
    component, weighted 1 : 3), rolled out from U0 on the coarse grid."""
    import jax
    import jax.numpy as jnp
    from luna.setup.evolve import step
    from luna.time_integration.revolve import revolve_rollout
    if n_steps % every:
        raise ValueError(f"n_steps={n_steps} not a multiple of every={every}")
    J = n_steps // every
    w = jnp.asarray([1.0 / norm[0]] + [1.0 / (3.0 * norm[1])] * 3)

    def mse(U, tgt):
        d = observed(U) - tgt
        return jnp.sum(w * jnp.mean(d * d, axis=(1, 2, 3)))

    def one(carry, theta, targets):
        st, L = carry
        st = step(st, dt, st.t + dt, with_params(theta))
        k = st.step
        L = L + jax.lax.cond(
            k % every == 0,
            lambda: mse(st.U, jax.lax.dynamic_index_in_dim(targets, k // every, 0, False)),
            lambda: jnp.zeros((), st.U.dtype))
        return st, L

    phys0 = with_params(None)                      # structure only: no corrector use

    if ckpt == "revolve":
        roll = revolve_rollout(one, n_steps, snaps, aux=True)

        def loss(theta, U0, targets, table):
            st = state0(phys0, init, U0, table)
            _, L = roll(theta, (st, jnp.zeros((), U0.dtype)), targets)
            return L / J
    else:
        def loss(theta, U0, targets, table):
            st = state0(phys0, init, U0, table)

            @jax.checkpoint
            def body(c, _):
                return one(c, theta, targets), None
            (_, L), _ = jax.lax.scan(body, (st, jnp.zeros((), U0.dtype)), None,
                                    length=n_steps)
            return L / J
    return loss


def make_forward(with_params, init, n_chunk, dt):
    """adv(theta, st) -> st after n_chunk coarse steps (loop bound hidden)."""
    import jax
    from luna.setup.evolve import step
    from luna.time_integration.revolve import _opaque

    @jax.jit
    def adv(theta, st):
        phys = with_params(theta)
        return jax.lax.fori_loop(0, _opaque(n_chunk),
                                 lambda _, s: step(s, dt, s.t + dt, phys), st)
    return adv


# ---------------------------------------------------------------------------
# reference data on disk: meta.json, forcing.npy, snap/cNNNNNN.npy
# (4, nc, nc, nc) float32 [rho, rho v] pooled f^3
# ---------------------------------------------------------------------------

def _snap(ref, n):
    return os.path.join(ref, "snap", f"c{n:06d}.npy")


def load_meta(ref):
    with open(os.path.join(ref, "meta.json")) as fh:
        return json.load(fh)


def _U_from(a):
    """(5, ...) float64 isothermal state from the stored (4, ...) block."""
    a = a.astype(np.float64)
    ke = 0.5 * (a[1] ** 2 + a[2] ** 2 + a[3] ** 2) / a[0]
    return np.concatenate([a, (CS * CS * a[0] / (GAMMA - 1.0) + ke)[None]])


def load_state(ref, n, sharding):
    """The pooled reference state at coarse step n, read shard by shard."""
    import jax
    mm = np.load(_snap(ref, n), mmap_mode="r")
    shape = (5,) + mm.shape[1:]
    return jax.make_array_from_callback(
        shape, sharding, lambda idx: _U_from(np.asarray(mm[(slice(None),) + idx[1:]])))


def load_targets(ref, ns, sharding):
    """(len(ns), 4, ...) observed() of the reference at coarse steps ns."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    mms = [np.load(_snap(ref, n), mmap_mode="r") for n in ns]
    shape = (len(ns), 4) + mms[0].shape[1:]
    sh = NamedSharding(sharding.mesh, P(None, None, *SPEC))

    def cb(idx):
        out = []
        for mm in mms[idx[0]]:
            a = np.asarray(mm[(slice(None),) + idx[2:]]).astype(np.float64)
            out.append(np.concatenate([np.log(a[0] / RHO0)[None], a[1:4] / a[0] / CS]))
        return np.stack(out)[:, idx[1]]
    return jax.make_array_from_callback(shape, sh, cb)


def gather(x):
    """A sharded (maybe multi-process) array as numpy on every process."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    rep = NamedSharding(x.sharding.mesh, P())
    return np.asarray(jax.jit(lambda a: a, out_shardings=rep)(x))


# ---------------------------------------------------------------------------
# statistics (host, numpy): spectra, PDF, structure functions
# ---------------------------------------------------------------------------

S_BINS = np.linspace(-8.0, 6.0, 141)
LAGS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])


def _shell(power):
    n = power.shape[0]
    k = np.fft.fftfreq(n) * n
    kk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    kb = np.rint(kk).astype(int).ravel()
    p = np.bincount(kb, weights=power.ravel(), minlength=n)
    return p[1:n // 2 + 1]


def statistics(U):
    """Density and velocity power spectra, ln-rho PDF, velocity structure
    functions S_p(l) (p = 1, 2, 3; |dv| along the axes), and scalars."""
    rho = U[0]
    v = U[1:4] / rho
    n = rho.shape[0]
    fr = np.fft.fftn(rho / rho.mean() - 1.0) / n ** 3
    pr = _shell(np.abs(fr) ** 2)
    pv = sum(_shell(np.abs(np.fft.fftn(v[i]) / n ** 3) ** 2) for i in range(3))
    s = np.log(rho / RHO0)
    pdf, _ = np.histogram(s, bins=S_BINS, density=True)
    lags = LAGS[LAGS < n // 2]
    sf = np.zeros((3, len(lags)))
    for j, l in enumerate(lags):
        acc = np.zeros(3)
        for ax in range(3):
            dv = np.sqrt(np.sum((np.roll(v, -l, axis=ax + 1) - v) ** 2, axis=0))
            acc += [np.mean(dv), np.mean(dv ** 2), np.mean(dv ** 3)]
        sf[:, j] = acc / 3.0
    return dict(Prho=pr, Pv=pv, pdf=pdf, sf=sf, lags=lags,
                slice=s[:, :, n // 2].astype(np.float32),
                mach=float(np.sqrt(np.mean(np.sum(v * v, axis=0))) / CS),
                sigma_s=float(np.std(s)), rho_max=float(rho.max()), rho_min=float(rho.min()))


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def _args(argv, **defaults):
    kv = dict(a.split("=", 1) for a in argv if "=" in a)
    p = dict(defaults)
    for k in list(kv):
        if k in p:
            p[k] = kv.pop(k)
    if kv:
        raise SystemExit(f"unknown arguments: {sorted(kv)}")
    return p


def _mesh(p):
    return tuple(int(m) for m in p["mesh"].split(","))


def _say(*a):
    import jax
    if jax.process_index() == 0:
        print(*a, flush=True)


def _peak_gib():
    import jax
    st = jax.local_devices()[0].memory_stats() or {}
    return st.get("peak_bytes_in_use", 0) / 2 ** 30


def _ranges(spec):
    """"a:b:s,c:d:t" -> sorted set of coarse steps."""
    out = set()
    for part in filter(None, spec.split(",")):
        a, b, s = (int(x) for x in part.split(":"))
        out.update(range(a, b, s))
    return sorted(out)


def _where_bad(U):
    """Every process: per shard, the non-finite cells, min rho and p, and the
    first bad cell's global index."""
    import jax
    for s in U.addressable_shards:
        a = np.asarray(s.data)
        rho = a[0]
        p = (GAMMA - 1.0) * (a[4] - 0.5 * np.sum(a[1:4] ** 2, axis=0) / rho)
        bad = ~np.isfinite(a).all(axis=0) | ~(p > 0) | ~(rho > 0)
        first = None
        if bad.any():
            loc = np.argwhere(bad)[0]
            first = tuple(int(sl.start or 0) + int(i) for sl, i in zip(s.index[1:], loc))
        print(f"[bad] process {jax.process_index()} device {s.device} index {s.index[1:]} "
              f"nonfinite {int((~np.isfinite(a)).sum())} bad cells {int(bad.sum())} "
              f"rho_min {np.nanmin(rho):.3e} p_min {np.nanmin(p):.3e} first bad {first}",
              flush=True)


def mode_ref(argv):
    """The fine run: adaptive steps landing on every coarse step; OU
    amplitudes recorded at every one, pooled states written at `rec`."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import PartitionSpec as P
    from luna.setup.evolve import initial_state, step
    from luna.utils.shmap import shard_map
    p = _args(argv, out="", nx="1024", f="4", mach="5", seed="7", mesh="1,1,1",
              acc_coef="2.0", cflc="0.3", vfac="3.0", n_end="0", rec="", diag_every="16",
              wall_h="23.0", fuse="1")
    nx, f, mach = int(p["nx"]), int(p["f"]), float(p["mach"])
    nc = nx // f
    out = p["out"]
    t_wall0 = time.time()
    cfg, init, physics = assemble_problem(config(nx, mach, int(p["seed"]), _mesh(p),
                                                 float(p["acc_coef"])), jax.process_index() == 0)
    # the coarse step: a fixed CFL cflc on the coarse grid at a signal speed
    # cs (1 + vfac mach)
    dtc = float(p["cflc"]) / nc / (CS * (1.0 + float(p["vfac"]) * mach))
    n_end = int(p["n_end"])
    rec = set(_ranges(p["rec"]))
    if jax.process_index() == 0:
        os.makedirs(os.path.join(out, "snap"), exist_ok=True)
    meta = dict(nx=nx, f=f, nc=nc, mach=mach, seed=int(p["seed"]), dtc=dtc, cs=CS,
                rho0=RHO0, gamma=GAMMA, acc_coef=float(p["acc_coef"]), cflc=float(p["cflc"]),
                vfac=float(p["vfac"]), t_turn=t_turn(mach), n_end=n_end, rec=p["rec"],
                mesh=p["mesh"])
    sh = physics.hydro.sharding
    spec = P(None, *SPEC)

    def _pool(U):
        c, a, b, d = U.shape
        return U[:4].reshape(4, a // f, f, b // f, f, d // f, f).mean(axis=(2, 4, 6)) \
            .astype(jnp.float32)
    pool = jax.jit(shard_map(_pool, mesh=sh.mesh, in_specs=spec, out_specs=spec))
    # fuse=1: the whole step one program (fast, ~270 B/cell); fuse=0: the
    # production loop's form, each stage its own program (~150 B/cell)
    if p["fuse"] == "1":
        jstep = jax.jit(lambda st, dt, t: step(st, dt, t, physics), donate_argnums=0)
    else:
        def jstep(st, dt, t):
            return step(st, dt, t, physics)
    cdt = jax.jit(physics.hydro.compute_dt)

    @jax.jit
    def diag(U):
        rho = U[0]
        v2 = jnp.sum(U[1:4] ** 2, axis=0) / rho ** 2
        return jnp.stack([jnp.sqrt(jnp.mean(v2)), jnp.max(jnp.sqrt(v2)) + CS,
                          jnp.max(rho), jnp.min(rho)])

    # restart: the fine state and the OU carry at coarse step n
    rst = os.path.join(out, "restart")
    st = initial_state(init, physics)
    n = 0
    A = []
    if os.path.exists(os.path.join(rst, "info.json")):
        with open(os.path.join(rst, "info.json")) as fh:
            info = json.load(fh)
        n = int(info["n"])
        mm = np.load(os.path.join(rst, "U.npy"), mmap_mode="r")
        U = jax.make_array_from_callback(mm.shape, sh, lambda idx: np.asarray(mm[idx]))
        a_ou = jnp.asarray(np.load(os.path.join(rst, "a.npy")))
        key = jnp.asarray(np.load(os.path.join(rst, "key.npy")))
        A = list(np.load(os.path.join(out, "forcing.npy"))[:n + 1])
        st = st.update(U=U, t=n * dtc, step=int(info["step"])).with_source("forcing", (a_ou, key))
        _say(f"restart at coarse step {n}, t = {n * dtc:.5f}")
    else:
        A = [np.asarray(st.sources["forcing"][0])]
    _say(f"REF nx={nx} f={f} nc={nc} mach={mach} dtc={dtc:.4e} t_turn={t_turn(mach):.4f} "
         f"n_end={n_end} ({n_end * dtc / t_turn(mach):.2f} t_turn) rec={len(rec)} snapshots "
         f"devices={jax.device_count()} processes={jax.process_count()}")
    log = []
    t0, fine0 = time.time(), 0
    while n < n_end:
        target = (n + 1) * dtc
        t = float(st.t)
        while True:
            dt = float(cdt(st.U))
            if not np.isfinite(dt) or dt <= 0:
                _where_bad(st.U)
                raise RuntimeError(f"dt={dt} at t={t}, coarse step {n}, fine step {fine0}")
            if t + dt >= target * (1.0 - 1e-12):
                st = jstep(st, target - t, target)
                fine0 += 1
                break
            st = jstep(st, dt, t + dt)
            t += dt
            fine0 += 1
        n += 1
        A.append(np.asarray(st.sources["forcing"][0]))
        if n in rec:
            path = _snap(out, n)
            c = pool(st.U)
            if jax.process_index() == 0:
                np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                          shape=(4, nc, nc, nc)).flush()
            multihost_utils.sync_global_devices(f"snap{n}")
            mm = np.lib.format.open_memmap(path, mode="r+")
            for s in c.addressable_shards:
                if s.replica_id == 0:
                    mm[s.index] = np.asarray(s.data)
            mm.flush()
            del mm
            multihost_utils.sync_global_devices(f"snap{n}w")
        if n % int(p["diag_every"]) == 0 or n == n_end:
            d = [float(x) for x in diag(st.U)]
            log.append([n, n * dtc] + d + [fine0])
            el = time.time() - t0
            _say(f"  n={n:6d} t={n * dtc:.5f} ({n * dtc / t_turn(mach):.3f} t_turn) "
                 f"Mach={d[0]:.3f} smax={d[1]:.2f} (CFL_c={d[1] * dtc * nc:.3f}) "
                 f"rho=[{d[3]:.3e},{d[2]:.3e}] fine steps {fine0} "
                 f"{el / max(fine0, 1):.3f} s/fine step")
            if not np.all(np.isfinite(d)):
                raise RuntimeError("non-finite state")
        if (time.time() - t_wall0) / 3600.0 > float(p["wall_h"]):
            break
    if jax.process_index() == 0:
        np.save(os.path.join(out, "forcing.npy"), np.stack(A))
        old = []
        if os.path.exists(os.path.join(out, "diag.npy")):
            old = list(np.load(os.path.join(out, "diag.npy")))
        np.save(os.path.join(out, "diag.npy"), np.array(old + log))
        with open(os.path.join(out, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
    if n < n_end:                                  # out of wall time: write a restart
        if jax.process_index() == 0:
            os.makedirs(rst, exist_ok=True)
            np.lib.format.open_memmap(os.path.join(rst, "U.npy"), mode="w+",
                                      dtype=np.float64, shape=st.U.shape).flush()
        multihost_utils.sync_global_devices("rst")
        mm = np.lib.format.open_memmap(os.path.join(rst, "U.npy"), mode="r+")
        for s in st.U.addressable_shards:
            if s.replica_id == 0:
                mm[s.index] = np.asarray(s.data)
        mm.flush()
        del mm
        multihost_utils.sync_global_devices("rstw")
        if jax.process_index() == 0:
            a_ou, key = st.sources["forcing"]
            np.save(os.path.join(rst, "a.npy"), np.asarray(a_ou))
            np.save(os.path.join(rst, "key.npy"), np.asarray(key))
            with open(os.path.join(rst, "info.json"), "w") as fh:
                json.dump(dict(n=n, step=int(st.step)), fh)
        _say(f"REF PAUSED at n={n}: restart written")
        return True
    _say(f"REF DONE n={n} fine steps {fine0} wall {time.time() - t0:.0f} s")
    return True


def _coarse_setup(p, meta, depth):
    cfgc = config(meta["nc"], meta["mach"], meta["seed"], _mesh(p), meta["acc_coef"])
    amp = float(p.get("amp", 1.0))
    m = dict(rho0=RHO0, c=CS, amp=amp)
    return coarse_problem(cfgc, m, depth, False) + (m,)


def _norm(ref, meta, n, sharding):
    import jax.numpy as jnp
    t = load_targets(ref, [n], sharding)[0]
    return (float(jnp.var(t[0])), float(jnp.mean(jnp.var(t[1:4], axis=(1, 2, 3)))))


def mode_train(argv, offline=False):
    """Adam on the SiL loss; `unroll` is the curriculum of rollout lengths,
    `iters` the iterations at each. offline: one-step supervised (unroll=1,
    every=1): the same network and loss with no solver in the loop."""
    import jax
    import jax.numpy as jnp
    from luna.sources.ml_corrector import (adam_init, adam_step, cnn3d_init, param_count,
                                           save_corrector3d)
    p = _args(argv, ref="", out="", mesh="1,1,1", unroll="8,32,128", iters="200,200,200",
              every="4", width="16", depth="3", amp="1.0", lr="3e-4", seed="0",
              train="", snaps="16", ckpt="revolve", init="", log_every="10", kind="sep")
    ref, out = p["ref"], p["out"]
    meta = load_meta(ref)
    depth, width = int(p["depth"]), int(p["width"])
    cfgc, init, with_params, m = _coarse_setup(p, meta, depth)
    sh = init.U.sharding
    lo, hi = (int(x) for x in p["train"].split(":"))          # training window of n0
    A = np.load(os.path.join(ref, "forcing.npy"))
    dt = meta["dtc"]
    norm = _norm(ref, meta, lo, sh)
    unrolls = [1] if offline else [int(x) for x in p["unroll"].split(",")]
    iters = [int(x) for x in p["iters"].split(",")]
    every = 1 if offline else int(p["every"])
    key = jax.random.PRNGKey(int(p["seed"]))
    theta = cnn3d_init(key, width=width, depth=depth, kind=p["kind"])
    if p["init"]:
        from luna.sources.ml_corrector import load_corrector3d
        theta, _ = load_corrector3d(p["init"])
    # replicated on the mesh from the start, as the updates come back: a
    # single-device first theta recompiles the gradient on iteration 1
    rep = jax.sharding.NamedSharding(sh.mesh, jax.sharding.PartitionSpec())
    theta = jax.tree.map(lambda x: jax.make_array_from_callback(
        x.shape, rep, lambda idx, h=np.asarray(x): h[idx]), theta)
    opt = adam_init(theta)
    rng = np.random.default_rng(int(p["seed"]))
    if jax.process_index() == 0:
        os.makedirs(out, exist_ok=True)
    _say(f"TRAIN {'offline' if offline else 'sil'} nc={meta['nc']} mach={meta['mach']} "
         f"params={param_count(theta)} unroll={unrolls} iters={iters} every={every} "
         f"window n0 in [{lo},{hi}) norm={norm} devices={jax.device_count()} "
         f"processes={jax.process_count()}")
    hist = []
    it_all = 0
    for K, n_it in zip(unrolls, iters):
        J = K // every
        loss = make_loss(with_params, init, n_steps=K, every=every, dt=dt, norm=norm,
                         ckpt=p["ckpt"], snaps=int(p["snaps"]))
        vg = jax.jit(jax.value_and_grad(loss))
        for it in range(n_it):
            n0 = int(rng.integers(lo, hi - K + 1))
            # every rank draws the same n0 (same seed); the data is read per shard
            t_io = time.time()
            U0 = load_state(ref, n0, sh)
            tg = load_targets(ref, [n0 + j * every for j in range(J + 1)], sh)
            table = jnp.asarray(A[n0:n0 + K + 1])
            t_io = time.time() - t_io
            t1 = time.time()
            L, g = vg(theta, U0, tg, table)
            L = float(L)
            t_g = time.time() - t1
            if not np.isfinite(L):
                _say(f"  it {it_all}: non-finite loss at n0={n0}, skipped")
                it_all += 1
                continue
            gn = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g))))
            g = jax.tree.map(lambda x: x * min(1.0, 1.0 / (gn + 1e-30)), g)
            lr = float(p["lr"]) * min(1.0, (it_all + 1) / 20.0)
            theta, opt = adam_step(g, theta, opt, lr)
            hist.append([it_all, K, n0, L, gn, t_g, t_io, _peak_gib()])
            if it % int(p["log_every"]) == 0 or it == n_it - 1:
                _say(f"  it {it_all:5d} K={K:4d} n0={n0:6d} loss={L:.5e} |g|={gn:.3e} "
                     f"grad {t_g:.2f} s io {t_io:.2f} s peak {_peak_gib():.1f} GiB")
            it_all += 1
        if jax.process_index() == 0:
            save_corrector3d(os.path.join(out, f"weights_K{K}.npz"), theta, **m)
            np.save(os.path.join(out, "hist.npy"), np.array(hist))
    if jax.process_index() == 0:
        save_corrector3d(os.path.join(out, "weights.npz"), theta, **m)
        np.save(os.path.join(out, "hist.npy"), np.array(hist))
    _say("TRAIN DONE")
    return True


def mode_eval(argv):
    """Coarse rollouts from the reference state at n0: bare, and with each
    set of weights; statistics every `every` steps (and of the reference
    where it has a snapshot) into out.npz."""
    import jax
    from luna.sources.ml_corrector import load_corrector3d
    p = _args(argv, ref="", out="", mesh="1,1,1", weights="", n0="0", steps="1000",
              every="100", depth="3")
    ref = p["ref"]
    meta = load_meta(ref)
    names = ["coarse"] + [w for w in p["weights"].split(",") if w]
    depth = int(p["depth"])
    if len(names) > 1:
        depth = len(load_corrector3d(names[1])[0])
    cfgc, init, with_params, _ = _coarse_setup(p, meta, depth)
    sh = init.U.sharding
    A = np.load(os.path.join(ref, "forcing.npy"))
    n0, steps, every = int(p["n0"]), int(p["steps"]), int(p["every"])
    have = {int(f[1:7]) for f in os.listdir(os.path.join(ref, "snap"))}
    adv = make_forward(with_params, init, every, meta["dtc"])
    res = {}
    for name in names:
        theta = None if name == "coarse" else load_corrector3d(name)[0]
        U0 = load_state(ref, n0, sh)
        st = state0(with_params(theta), init, U0, jax.numpy.asarray(A[n0:n0 + steps + 1]))
        rows = []
        t0 = time.time()
        for j in range(steps // every + 1):
            if j:
                st = adv(theta, st)
            U = gather(st.U)
            ok = bool(np.all(np.isfinite(U)))
            n = n0 + j * every
            row = dict(n=n, ok=ok)
            if ok and jax.process_index() == 0:
                row.update(statistics(U))
                if n in have:
                    R = _U_from(np.load(_snap(ref, n)))
                    ob, obr = np.asarray(observed(U)), np.asarray(observed(R))
                    row["err"] = [float(np.mean((ob[0] - obr[0]) ** 2)),
                                  float(np.mean(np.sum((ob[1:] - obr[1:]) ** 2, axis=0)))]
            rows.append(row)
            _say(f"  {os.path.basename(name)} n={n} ok={ok} "
                 f"mach={row.get('mach', float('nan')):.3f} "
                 f"sigma_s={row.get('sigma_s', float('nan')):.3f} err={row.get('err')} "
                 f"{time.time() - t0:.0f} s")
            if not ok:
                break
        res[name] = rows
    if jax.process_index() == 0:
        refrows = []
        for j in range(steps // every + 1):
            n = n0 + j * every
            if n in have:
                refrows.append(dict(n=n, **statistics(_U_from(np.load(_snap(ref, n))))))
        res["reference"] = refrows
        import pickle
        with open(p["out"], "wb") as fh:
            pickle.dump(dict(meta=meta, n0=n0, every=every, res=res), fh)
    _say("EVAL DONE")
    return True


def mode_cost(argv):
    """Time the coarse step with and without the corrector, the corrector
    alone, and the value-and-gradient of a K-step SiL loss (synthetic
    targets), on the production coarse problem."""
    import jax
    import jax.numpy as jnp
    from luna.setup.evolve import step
    from luna.sources.ml_corrector import cnn3d_init, param_count
    p = _args(argv, nx="256", mesh="1,1,1", mach="5", width="16", depth="3",
              unroll="16", every="4", snaps="16", ckpt="revolve", reps="3", grad="1",
              kind="sep")
    nx, width, depth = int(p["nx"]), int(p["width"]), int(p["depth"])
    meta = dict(rho0=RHO0, c=CS, amp=1.0)
    cfgc, init, with_params = coarse_problem(config(nx, float(p["mach"]), 7, _mesh(p)),
                                             meta, depth)
    theta = randomize_last(cnn3d_init(jax.random.PRNGKey(1), width=width, depth=depth,
                                      kind=p["kind"]), 0.01)
    sh = init.U.sharding
    # a turbulent-looking state: random velocities at Mach ~ mach, rho lognormal
    rng = np.random.default_rng(0)
    host = np.empty((5, nx, nx, nx))
    s = 0.5 * rng.standard_normal((nx, nx, nx))
    host[0] = np.exp(s)
    v = float(p["mach"]) * rng.standard_normal((3, nx, nx, nx)) / np.sqrt(3)
    host[1:4] = host[0] * v
    host[4] = host[0] / (GAMMA - 1) + 0.5 * host[0] * np.sum(v * v, axis=0)
    U0 = jax.make_array_from_callback(host.shape, sh, lambda idx, h=host: h[idx])
    del host
    A = jnp.zeros((int(p["unroll"]) + 1,) + tuple(
        np.asarray(init_forcing_shape(cfgc))), jnp.complex128)
    dt = 0.3 / nx / (1.0 + 3.0 * float(p["mach"]))
    phys_c = with_params(theta)
    phys_0 = replace(phys_c, stages=phys_c.stages[:-1])
    st = state0(phys_c, init, U0, A)
    corr = phys_c.stages[-1]
    reps = int(p["reps"])

    def timeit(fn, *a):
        jax.block_until_ready(fn(*a))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*a))
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    f_bare = jax.jit(lambda s_: step(s_, dt, s_.t + dt, phys_0).U)
    f_corr = jax.jit(lambda th, s_: step(s_, dt, s_.t + dt, with_params(th)).U)
    f_only = jax.jit(lambda th, U: corr._apply(th, U, dt))
    t_bare = timeit(f_bare, st)
    t_corr = timeit(f_corr, theta, st)
    t_only = timeit(f_only, theta, st.U)
    _say(f"COST nx={nx} mesh={p['mesh']} devices={jax.device_count()} width={width} "
         f"depth={depth} params={param_count(theta)}")
    _say(f"  step bare {t_bare * 1e3:.2f} ms  step+corrector {t_corr * 1e3:.2f} ms  "
         f"corrector alone {t_only * 1e3:.2f} ms  ratio corrector/bare {t_only / t_bare:.2f}")
    if p["grad"] == "1":
        K, every = int(p["unroll"]), int(p["every"])
        J = K // every
        tg = jax.jit(lambda U: jnp.stack([observed(U)] * (J + 1)),
                     out_shardings=jax.sharding.NamedSharding(sh.mesh, jax.sharding.PartitionSpec(None, None, *SPEC)))(U0)
        loss = make_loss(with_params, init, n_steps=K, every=every, dt=dt, norm=(1.0, 1.0),
                         ckpt=p["ckpt"], snaps=int(p["snaps"]))
        fwd = jax.jit(loss)
        vg = jax.jit(jax.value_and_grad(loss))
        t_f = timeit(fwd, theta, U0, tg, A)
        t_g = timeit(vg, theta, U0, tg, A)
        L, g = vg(theta, U0, tg, A)
        _say(f"  K={K} loss fwd {t_f:.3f} s  value+grad {t_g:.3f} s  ratio {t_g / t_f:.2f}  "
             f"per step {t_g / K * 1e3:.1f} ms  peak {_peak_gib():.1f} GiB  loss {float(L):.4e} "
             f"|g| {float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g)))):.3e}")
    _say("COST DONE")
    return True


def randomize_last(params, scale, seed=2):
    """The zero last layer's mixing weights drawn at `scale` (a corrector
    that does something, for costs and tests)."""
    import jax
    last = list(params[-1])
    j = 0 if len(last) == 2 else 1
    last[j] = scale * jax.random.normal(jax.random.PRNGKey(seed), last[j].shape, last[j].dtype)
    return params[:-1] + [tuple(last)]


def init_forcing_shape(cfg):
    from luna.hydro.turbulence import OUDriver
    return (OUDriver(cfg.ou_config).n_modes, 3)


def main(argv):
    mode = argv[0]
    import jax
    if os.environ.get("JAX_PLATFORMS", "").startswith("cpu") and \
            int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        from luna.setup.jax_init import initialize_jax_distributed
        initialize_jax_distributed(enable_x64=True)
    jax.config.update("jax_enable_x64", True)
    fn = {"ref": mode_ref, "train": mode_train, "eval": mode_eval, "cost": mode_cost,
          "offline": lambda a: mode_train(a, offline=True)}[mode]
    return 0 if fn(argv[1:]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
