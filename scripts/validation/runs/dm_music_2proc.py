"""Multi-process check of the DM-only blocks path (pytest launches no processes).

The chunked IC read -- each rank reading only the index range its own block
covers -- existed for the slab path alone until 2026-09-19. This exercises it
on the blocks path: 2 ranks, chunked read, then the global particle count.

  srun -n2 ... python scripts/validation/runs/dm_music_2proc.py out=OUT

A chunked read that drops rows agrees with its OWN baseline all the way to
a_end -- the loop's conservation watchdog compares against a baseline taken
after assembly, so it cannot see it; the count against the IC can.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from luna.setup.cli import args                   # noqa: E402

BOX, NP, AMP, A_I, A_END = 25000.0, 32, 0.02, 0.02, 0.1


def run(out, dump_s):
    import faulthandler
    faulthandler.dump_traceback_later(dump_s, repeat=True, file=sys.stderr)
    import jax
    from luna.setup.jax_init import initialize_jax_distributed
    initialize_jax_distributed(enable_x64=True)
    pid = jax.process_index()
    ic = os.path.join(out, "ic.hdf5")
    os.makedirs(out, exist_ok=True)
    if pid == 0 and not os.path.exists(ic):
        from _music_growing_mode import growing_mode_ic
        growing_mode_ic(ic + ".tmp")
        os.rename(ic + ".tmp", ic)
    while not os.path.exists(ic):
        time.sleep(1)

    from dataclasses import replace
    from luna import main
    from luna.template_config import Config
    cfg = replace(Config(), run="dm", CosmologicalIntegration=True, music_ic_path=ic,
                  Nx=NP, Ny=NP, Nz=NP, a_end=A_END, dloga_max=0.02,
                  output_dir=os.path.join(out, "run"),
                  PLOT_AFTER_SIM=False, decomposition="blocks", write_snapshots=False)
    res = main.run(cfg).extra
    n, m = res["n_particles"], res["mass"]
    print(f"[rank {pid}] a={res['a']:.4f} steps={res['n_steps']} n={n} mass={m:.8e}", flush=True)
    assert n == NP ** 3, f"rank {pid}: {n} particles of {NP ** 3} -- chunked read lost rows"
    print(f"[rank {pid}] OK: every particle accounted for", flush=True)


if __name__ == "__main__":
    a = args(out="", dump_s=200)
    if not a["out"]:
        raise SystemExit("out=<dir> is required")
    run(a["out"], a["dump_s"])
