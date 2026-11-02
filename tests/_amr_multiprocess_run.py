"""One sharded AMR run (3D Sod) for tests/test_amr_multiprocess.py, as one process or
one of several: `python tests/_amr_multiprocess_run.py output_dir=O`. Rank 0 prints
RESULT (steps, last block snapshot); every rank prints PROGRAMS (its multi-process compiles)."""
import faulthandler
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NX, NT, LEVELS, T_END = 32, 16, 1, 0.03


def config(output_dir):
    from dataclasses import replace
    sys.path.insert(0, ROOT)
    from scripts.validation.runs.sod3d_amr_run import config as sod
    return replace(sod(nx=NX, nt=NT, t_end=T_END), amr_levels=LEVELS, amr_shard=True,
                   amr_dense_snapshots=False, output_dir=output_dir)


def main(argv):
    kv = dict(a.split("=", 1) for a in argv)
    faulthandler.dump_traceback_later(float(kv.get("dump_after", 1e9)), exit=False)
    import jax
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1 and \
            os.environ.get("JAX_PLATFORMS", "").startswith("cpu"):
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
    import numpy as np
    from luna.setup.cli import launch
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from _sinks_multiprocess_run import _record_multiprocess_compiles
    programs = _record_multiprocess_compiles()
    res = launch(config(kv["output_dir"]))
    if jax.process_index() == 0:
        snaps = sorted(f for f in os.listdir(kv["output_dir"]) if f.startswith("blocks_"))
        s = np.load(os.path.join(kv["output_dir"], snaps[-1]))
        levels = int(s["levels"])
        print("RESULT " + repr(dict(
            n_steps=int(res.n_steps), t=float(res.t), snap=snaps[-1], levels=levels,
            coords=[s[f"L{L}_coords"].tolist() for L in range(levels)],
            data=[s[f"L{L}_data"].tolist() for L in range(levels)],
            processes=jax.process_count(), devices=jax.device_count())), flush=True)
    print(f"PROGRAMS {jax.process_index()} {programs!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    sys.exit(main(sys.argv[1:]))
