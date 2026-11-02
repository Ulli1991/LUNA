"""One step of the driven-turbulence box, the stages compiled apart (the
production loop's form) vs together in one program, from the same state.

    srun ... python tools/fusediff.py n=1024 mesh=2,2,2 warm=30 [barrier=0]

`warm` unfused steps first (a flow that moves), then per variant: the cells
that differ from the unfused step, per field, on every shard. Variants: "all"
(the whole step one jit, as sil3d ref fuse=1), "kick+hydro", "hydro+after"
(hydro, the second kick, the projection), "hydro" (a control), and the finer
bisection of "kick+hydro" below. barrier=0 uses the hydro step without
assemble's optimization barrier (the code before 6fbf00b). Exit status 1
when any cell differs. docs/AUTODIFF.md, "What the reference found".
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv):
    kv = dict(a.split("=", 1) for a in argv)
    n = int(kv.get("n", 1024))
    mesh_shape = tuple(int(m) for m in kv.get("mesh", "2,2,2").split(","))
    warm = int(kv.get("warm", 30))
    barrier = kv.get("barrier", "1") == "1"
    import jax
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        from luna.setup.jax_init import initialize_jax_distributed
        initialize_jax_distributed(enable_x64=True)
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from luna.setup.evolve import initial_state, step
    from tools.sil3d import assemble_problem, config
    rank = jax.process_index()
    cfg, init, phys = assemble_problem(config(n, 5.0, 7, mesh_shape, 2.55))
    if not barrier:
        from dataclasses import replace
        from luna.setup.hydro import create_solver_distributed
        raw = create_solver_distributed(cfg, phys.hydro.grid, phys.mesh)[1]
        phys = replace(phys, hydro=replace(phys.hydro, step=raw))
    st = initial_state(init, phys)
    cdt = jax.jit(phys.hydro.compute_dt)
    t = 0.0
    for _ in range(warm):
        dt = float(cdt(st.U))
        st = step(st, dt, t + dt, phys)
        t += dt
    dt = float(cdt(st.U))
    h = 0.5 * dt
    stages = phys.stages
    projections = phys.projections

    def before(s):
        for stg in stages:
            s = stg.before(s, h, dt)
        return s

    def hydro(s):
        return s.update(U=phys.hydro.step(s.U, dt))

    def after(s):
        s = s.update(t=t + dt, a=phys.clock.scale(t + dt))
        for stg in reversed(stages):
            s = stg.after(s, h, dt)
        U = s.U
        for p in projections:
            U = p(U, s.a)
        return s.update(U=U, step=s.step + 1)

    ref = after(hydro(before(st))).U                     # stage by stage
    kicked = before(st).U                                # the first kick alone, its own program
    from jax.sharding import PartitionSpec as P
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap
    from luna.utils.shmap import shard_map
    halo = make_halo_shardmap(phys.mesh, 3)
    spec = P(None, "x", "y", "z")
    halo_rt = shard_map(lambda u: halo(u, 0, 3, width=2)[:, 2:-2, 2:-2, 2:-2],
                        mesh=phys.mesh, in_specs=spec, out_specs=spec)
    local_id = shard_map(lambda u: u * 1.0, mesh=phys.mesh, in_specs=spec, out_specs=spec)
    # (variant, reference) for the finer bisection of "kick+hydro"
    extra = {
        "kick": (lambda s: jax.jit(lambda x: before(x).U)(s), kicked),
        "kick+copy": (lambda s: jax.jit(lambda x: before(x).U * 1.0)(s), kicked),
        "kick+shmap": (lambda s: jax.jit(lambda x: local_id(before(x).U))(s), kicked),
        "kick+halo": (lambda s: jax.jit(lambda x: halo_rt(before(x).U))(s), kicked),
        "kick+hydro0": (lambda s: jax.jit(lambda x: phys.hydro.step(before(x).U, 0.0))(s), kicked),
        # the kick materialised before the hydro reads it, in one program
        "kick-bar-hydro": (lambda s: after(jax.jit(lambda x: hydro(x.update(
            U=jax.lax.optimization_barrier(before(x).U))))(s)).U, ref),
    }
    variants = {
        "all": lambda s: jax.jit(lambda x: after(hydro(before(x))))(s).U,
        "kick+hydro": lambda s: after(jax.jit(lambda x: hydro(before(x)))(s)).U,
        "hydro+after": lambda s: jax.jit(lambda x: after(hydro(x)))(before(s)).U,
        "hydro": lambda s: after(jax.jit(hydro)(before(s))).U,
    }
    if rank == 0:
        print(f"FUSEDIFF n={n} mesh={mesh_shape} processes={jax.process_count()} warm={warm} "
              f"barrier={int(barrier)} t={t:.5f} dt={dt:.3e}", flush=True)

    from functools import partial

    def _cmp_local(a, b, c):
        """This shard's field c: cells off by more than 1e-12 of the value or
        of the field's scale (round-off where momentum crosses zero is not a
        miscompile), and max |diff|."""
        diff = jnp.abs(a[c] - b[c])
        tol = 1e-12 * (jnp.abs(b[c]) + jnp.max(jnp.abs(b[c])))
        return (jnp.sum(diff > tol)[None],
                jnp.max(diff)[None])

    dev_axes = P(("x", "y", "z"))
    # one field per program: all five at once ran out of memory at 1024^3
    cmp_g = {c: jax.jit(shard_map(partial(_cmp_local, c=c), mesh=phys.mesh,
                                  in_specs=(spec, spec), out_specs=(dev_axes, dev_axes)))
             for c in range(5)}

    ndiff = 0
    for name in kv.get("variants", "all,kick+hydro,hydro+after,hydro").split(","):
        if name in extra:
            fn, r = extra[name]
            out = fn(st)
        else:
            out, r = variants[name](st), ref
        if not out.sharding.is_equivalent_to(r.sharding, out.ndim):
            # a whole-step jit picks its own output layout; the shard_map
            # compare would gather it
            out = jax.jit(lambda a: a, out_shardings=r.sharding)(out)
        res =[cmp_g[c](out, r) for c in range(5)]
        del out
        lines = []
        for i, sc in enumerate(res[0][0].addressable_shards):
            c = [int(np.asarray(res[f][0].addressable_shards[i].data)[0]) for f in range(5)]
            m = [float(np.asarray(res[f][1].addressable_shards[i].data)[0]) for f in range(5)]
            ndiff += sum(c)
            lines.append(f"  [{name}] rank {rank} {sc.device} {'SAME' if sum(c) == 0 else 'DIFF'} "
                         f"cells per field {c} max |diff| {['%.1e' % x for x in m]}")
        print("\n".join(lines), flush=True)
    print(f"FUSEDIFF rank {rank} done: {'DIFF' if ndiff else 'SAME'}", flush=True)
    return 1 if ndiff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
