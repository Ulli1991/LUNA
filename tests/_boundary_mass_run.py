"""One outflow run with isolated gravity and sinks, for
tests/test_boundary_mass_budget.py: one process, or one of several (SLURM_*
and JAX_COORDINATOR_ADDRESS in the environment, CPU devices, gloo).

    python tests/_boundary_mass_run.py boundary=4 mesh=2,1,1 [steps=20 budget=1 out=U.npy]

Rank 0 prints one RESULT line: the initial and final gas mass, the sinks'
mass, and the (2, 6) face in/outflow the run accumulated.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config(boundary, mesh, budget=True):
    from dataclasses import replace
    from tests.test_wall_m3_ic import _m3
    # 1e4 Msun, not M3's 1e3: since ff7ec03 sink creation needs FLASH's bound
    # check (thermal energy included), which the 1e3 Msun cloud fails for 40 steps
    return replace(
        _m3(N=32, mass_msun=1.0e4, turb_v0_kms=0.616), MESH_SHAPE=tuple(mesh), BOUNDARY=boundary, SELF_GRAVITY=True,
        POISSON_BOUNDARY=3, POISSON_N_LEVELS=3, ORDER=1, LIMITER=0,
        enable_x64=True, sinks_enabled=True, sink_max_count=4,
        sink_creation_density=5.0, particle_boundary="outflow", write_snapshots=False,
        t_end=1.0, boundary_mass_budget=budget)


def main(argv):
    kv = dict(a.split("=", 1) for a in argv)
    import jax
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1 and \
            os.environ.get("JAX_PLATFORMS", "").startswith("cpu"):
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
    from luna.setup.jax_init import initialize_jax_distributed
    initialize_jax_distributed(enable_x64=True)
    import jax.numpy as jnp
    import numpy as np
    from amuse_luna.stepper import build
    budget = bool(int(kv.get("budget", 1)))
    cfg = config(int(kv["boundary"]), [int(m) for m in kv["mesh"].split(",")], budget)
    dV = (cfg.Lx / cfg.Nx) * (cfg.Ly / cfg.Ny) * (cfg.Lz / cfg.Nz)

    def masses(st):
        s = st.species["sinks"]
        return (float(jnp.sum(st.U[0])) * dV,           # every rank: collectives
                float(jnp.sum(s.mass * s.active)), int(jnp.sum(s.active)))

    st = build(cfg)
    m0 = masses(st)
    for _ in range(int(kv.get("steps", 20))):
        st.step_once(st.compute_dt())
    m_gas, m_sinks, n_sinks = masses(st)
    if "out" in kv:                                     # one process only
        np.save(kv["out"], np.asarray(st.U))
    acc = np.asarray(st.sources["boundary_mass"]) if budget else np.zeros((2, 6))
    if jax.process_index() == 0:
        print("RESULT " + repr(dict(
            m0=m0[0] + m0[1], m_gas=m_gas, m_sinks=m_sinks, n_sinks=n_sinks,
            steps=int(st.step), budget=acc.tolist(), processes=jax.process_count(),
            devices=jax.device_count())), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    sys.exit(main(sys.argv[1:]))
