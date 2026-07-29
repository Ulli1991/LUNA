# Example SLURM submission scripts

Two adaptable templates, one per GPU vendor. Copy one, adjust the `#SBATCH`
resources, `module load` lines, environment, and paths for your system, then
`sbatch` it.

| Script | System | What it runs |
|--------|--------|--------------|
| `example_production_3d_multinode.sh` | NVIDIA (MPCDF Raven, A100) | a 1024³ `shard_map` production run across 8 nodes / 32 GPUs |
| `example_amd_mi300a_2gpu.sh`         | AMD (MPCDF viper, MI300A / ROCm) | a multi-GPU run on one node, one JAX process per GPU |

Any run mode (validation problems, cosmology, autodiff tasks) uses the same
pattern — point the `python -m scripts...` line at the entry script you want
(see `scripts/validation/runs/` and `docs/ENV.md` for the knobs). Site-specific
launchers are deliberately kept out of the repo.
