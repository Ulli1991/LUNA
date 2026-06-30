# Example SLURM submission scripts

Adaptable templates for the main ways LUNA is run on a SLURM cluster
(developed on MPCDF Raven, NVIDIA A100 GPUs). Copy one, adjust the `#SBATCH`
resources, `module load` lines, environment, and paths for your system, then
`sbatch` it.

| Script | What it runs |
|--------|--------------|
| `example_validation_1gpu.sh`         | a hydrodynamics test problem (Sedov 3D) on a single GPU |
| `example_production_3d_multinode.sh` | a 1024³ `shard_map` production run across 8 nodes / 32 GPUs |
| `example_cosmo_dm_4gpu.sh`           | a grid-sharded dark-matter N-body run on 4 GPUs (parameterised via `--export`) |
| `example_autodiff_inverse_1gpu.sh`   | a reverse-mode initial-condition inversion (differentiable solver) on one GPU |

All entry points run through `scripts/` (see the `python -m scripts...` line in
each). The full set of project-internal submit scripts is kept out of the repo.
