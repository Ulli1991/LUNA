"""Multi-process validation of the coupled MUSIC gas+DM path (pytest cannot
launch processes): 2 ranks, chunked IC read, the block layout, then the k-mode
amplitude of the gas density against linear growth from the ranks' local
x-profiles (no collective in the analysis).

Recipe (viper apudev, CPU backend = code-path check; swap in the GPU stack for
a platform check):
  srun -p apudev --account=<your_account> --constraint=apu -N1 -n2 --gres=gpu:2 bash -c '
    export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1
    unset ROCR_VISIBLE_DEVICES
    export JAX_COORDINATOR_ADDRESS=$(hostname):$((20000 + SLURM_JOB_ID % 40000))
    python scripts/validation/runs/coupled_music_2proc.py mode=run out=OUT > OUT_rank$SLURM_PROCID.log 2>&1'
  python scripts/validation/runs/coupled_music_2proc.py mode=analyze out=OUT
Expected (2026-09-17, 32^3, a 0.02 -> 0.1): 0.9865, identical to the
single-process value of tests/test_cosmo_hydro_music_blocks.py. (The slab twin
this replaced measured 0.9442 and has been deleted.)
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np                           # noqa: E402

from luna.setup.cli import args                   # noqa: E402

BOX, NP, AMP, A_I = 25000.0, 32, 0.02, 0.02
k = 2.0 * np.pi / BOX


def analyze(out):
    from luna.cosmology import Cosmology
    prof = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("xprof_rank"):
            d = np.load(os.path.join(out, f))
            prof[int(d["lo"])] = (d["rho"], float(d["a"]))
    rho = np.concatenate([prof[lo][0] for lo in sorted(prof)])
    a_f = prof[sorted(prof)[0]][1]
    assert rho.shape[0] == NP, rho.shape
    x = (np.arange(NP) + 0.5) * (BOX / NP)
    dgas = rho / rho.mean() - 1.0
    amp = 2.0 * np.mean(dgas * np.cos(k * x))
    cosmo = Cosmology(Omega_m=0.3, Omega_L=0.7, h=0.6711)
    a_lin = -AMP * cosmo.growth_factor(a_f) / cosmo.growth_factor(A_I)
    print(f"[2proc {os.path.basename(out)}] a_f={a_f:.4f} gas mode/linear = {amp / a_lin:.4f}")


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
    cfg = replace(Config(), run="gas+dm", CosmologicalIntegration=True, SPATIAL_DIMENSION=3,
                  music_ic_path=ic, Nx=NP, Ny=NP, Nz=NP, a_end=0.1, dloga_max=0.02,
                  output_dir=os.path.join(out, "run"),
                  PLOT_AFTER_SIM=False, hydro_fused=True, decomposition="blocks")
    res = main.run(cfg)
    U, posS, a, step = res.extra["U"], res.extra["dm_pos"], res.t, res.n_steps
    # local x-profile of the gas density from this process's shard(s) only
    shards = U.addressable_shards
    rho_loc = np.concatenate([np.asarray(s.data)[0].mean(axis=(1, 2)) for s in
                              sorted(shards, key=lambda s: s.index[1].start or 0)])
    lo = min((s.index[1].start or 0) for s in shards)
    np.savez(os.path.join(out, f"xprof_rank{pid}.npz"), rho=rho_loc, lo=lo, a=float(a))
    n_loc = int(np.sum(np.any(np.asarray(posS.addressable_shards[0].data) != 0.0, axis=1)))
    print(f"[rank {pid}] run: steps={step} a={float(a):.4f} local gas cells "
          f"x[{lo}:{lo + rho_loc.size}] local active DM={n_loc}", flush=True)


if __name__ == "__main__":
    a = args(mode="run", out="", dump_s=150)
    if not a["out"]:
        raise SystemExit("out=<dir> is required")
    analyze(a["out"]) if a["mode"] == "analyze" else run(a["out"], a["dump_s"])
