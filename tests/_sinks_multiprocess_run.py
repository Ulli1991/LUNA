"""One standalone sink run through `luna.main.run`, for
tests/test_sinks_multiprocess.py: one process, or one of several (SLURM_*
and JAX_COORDINATOR_ADDRESS in the environment, CPU devices, gloo).

    python tests/_sinks_multiprocess_run.py ic_dir=D output_dir=O mesh=2,1,1

Rank 0 prints one RESULT line: steps, gas mass, and the final snapshot's sinks.
Every rank prints one PROGRAMS line: the programs over devices of more than one
process that it compiled, in order. On GPU compiling such a program waits for
the other processes (a rank-0-only reduction hung in backend_compile_and_load);
on CPU it does not, so the test compares these lists instead.
"""
import faulthandler
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, L, T_END = 32, 1.0, 0.2


def config(ic_dir, output_dir, mesh, rules="luna"):
    from dataclasses import replace
    from luna.template_config import Config
    return replace(
        Config(), test_problem="Jeans-3D", SPATIAL_DIMENSION=3,
        Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, gamma=5.0 / 3.0,
        BOUNDARY=0, POISSON_BOUNDARY=0, SELF_GRAVITY=True, GRAVITATIONAL_G=1.0,
        ORDER=1, LIMITER=0, riemann_solver="hllc", enable_x64=True,
        MESH_SHAPE=tuple(int(m) for m in mesh.split(",")),
        ic_dir=ic_dir, output_dir=output_dir, clear_output_dir=True,
        t_end=T_END, time_bet_snaps=T_END, output_cadence=1, diag_interval=1,
        enable_checkpointing=False, PLOT_AFTER_SIM=False,
        sinks_enabled=True, sink_max_count=4, sink_creation_density=5.0,
        sink_accretion_radius=0.0, sink_softening=0.0, sink_creation_separation=0.0,
        **(FLASH_RULES if rules == "flash" else {}))


# FLASH-Torch's sink rules (docs/TORCH.md "Sink rules: LUNA and FLASH")
FLASH_RULES = dict(sink_accretion="federrath", sink_creation_checks="flash", sink_merging=True)


def main(argv):
    kv = dict(a.split("=", 1) for a in argv)
    # a hung rank shows where it waits (the test kills it after this)
    faulthandler.dump_traceback_later(float(kv.get("dump_after", 1e9)), exit=False)
    import jax
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1 and \
            os.environ.get("JAX_PLATFORMS", "").startswith("cpu"):
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
    import jax.numpy as jnp
    import numpy as np
    from luna.setup.cli import launch
    programs = _record_multiprocess_compiles()
    res = launch(config(kv["ic_dir"], kv["output_dir"], kv["mesh"], kv.get("rules", "luna")))
    m_gas = float(jnp.sum(res.extra["U"][0])) * (L / N) ** 3     # every rank: a collective
    if jax.process_index() == 0:
        snaps = sorted(f for f in os.listdir(kv["output_dir"]) if f.startswith("sinks_"))
        s = np.load(os.path.join(kv["output_dir"], snaps[-1]))
        act = s["active"] > 0.5
        print("RESULT " + repr(dict(
            n_steps=int(res.n_steps), t=float(res.t), snap=snaps[-1], m_gas=m_gas,
            n_sinks=int(act.sum()), m_sinks=float(s["mass"].sum()),
            mass=s["mass"].tolist(), pos=s["pos"].tolist(), vel=s["vel"].tolist(),
            active=s["active"].tolist(), processes=jax.process_count(),
            devices=jax.device_count())), flush=True)
    print(f"PROGRAMS {jax.process_index()} {programs!r}", flush=True)
    return 0


def _record_multiprocess_compiles():
    """The names of the programs compiled over more than one process's devices,
    appended as JAX compiles them (a wrapper on its compile entry point)."""
    from jax._src import compiler
    orig, names = compiler.compile_or_get_cached, []

    def wrapped(backend, computation, devices, *a, **k):
        if len({d.process_index for d in devices.flat}) > 1:
            names.append(str(computation.operation.attributes["sym_name"]).strip('"'))
        return orig(backend, computation, devices, *a, **k)

    compiler.compile_or_get_cached = wrapped
    return names


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    sys.exit(main(sys.argv[1:]))
