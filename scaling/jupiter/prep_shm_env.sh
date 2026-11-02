# Source from inside an sbatch script AFTER module load, INSTEAD of
# activating the scratch venv. Broadcasts the venv+repo tarball to /dev/shm
# on every allocated node and re-points the environment there, so N-thousand
# ranks import Python from node-local RAM instead of hammering GPFS (the
# ~14 min startup at 4096 ranks). Build the tarball with make_shm_tarball.sh.
SHM_TAR=${LUNA_SCRATCH:?set LUNA_SCRATCH to your scratch directory}/shm_env/luna_env.tar
t0=$SECONDS
sbcast --force "$SHM_TAR" /dev/shm/luna_env.tar
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    tar -xf /dev/shm/luna_env.tar -C /dev/shm
echo "[shm-env] broadcast+untar $((SECONDS - t0))s"
# interpreter + stdlib + site-packages all from /dev/shm: bypass the venv
# (its pyvenv.cfg points at the GPFS module tree) via PYTHONHOME/PYTHONPATH
export PYTHONHOME=/dev/shm/luna_env/python
export PYTHONPATH=/dev/shm/luna_env/jax_env_cuda/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}
export PATH="/dev/shm/luna_env/python/bin:$PATH"
export LD_LIBRARY_PATH="/dev/shm/luna_env/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# node-local XLA cache: no GPFS write contention; 4 ranks/node share it
export LUNA_XLA_CACHE_DIR=/dev/shm/xla_cache
export LUNA_REPO_DIR=/dev/shm/luna_env/LUNA
