# third_party/jaxamr — reference copy (do not import at runtime)

Upstream: **JAX-AMR** by Haocheng Wen, Faxuan Luo, Hanbing Zou —
<https://github.com/JA4S/JAX-AMR> (MIT, see `LICENSE`).
Paper: Wen et al. 2025, *JANC*, arXiv:2504.13750.

This is a frozen, read-only snapshot kept for **attribution** and for
**parity testing** of LUNA's own block-AMR implementation (`amr/`). LUNA does
not depend on this package at runtime.

## Why LUNA re-implements rather than forks

We cherry-pick the good ideas into a LUNA-native, dimension-generic `amr/`
package instead of forking, because:

- LUNA already owns the hard 3D physics JAX-AMR lacks (3D MUSCL-Hancock in
  `hydro/`, 3D multigrid Poisson in `gravity/poisson_multigrid.py`). JAX-AMR is
  only needed for the *block bookkeeping*.
- We need 3D, and the upstream code hand-unrolls per-axis (2D only). A single
  ndim-parameterised code path is cleaner and is itself the "make it faster"
  improvement the project called for.
- We must keep the Poisson coupling open between a one-way Dirichlet scheme and
  a force-conserving composite scheme; owning the block layer lets us add the
  coarse-fine primitives the latter needs.

## What we keep (the genuinely good ideas)

- **Fixed-shape blocks padded to a max-per-level + a NaN sentinel block.** The
  JAX-tractable core insight: no ragged trees fighting XLA.
- Pipeline: refinement grid-mask → block-mask → block-info (neighbor topology)
  → gather block data.
- Block prolong/restrict; the refine-buffer halo; "preserve unaltered blocks"
  across re-adaptation.

## What we change

- **Morton-code** block identity + unaltered-block matching (O(N log N))
  instead of upstream `compare_coords` (O(N²) pairwise, `amr.py:356`).
- **δ density-contrast** refinement criterion (upstream is gradient-of-density,
  `amr.py:118`).
- An immutable `AMRConfig` struct threaded through, replacing upstream's global
  mutable module state (`set_amr()` sets `Lx, Ly, n_block, …` as globals).
- Dimension-generic (`ndim` / axis tuples) instead of hand-unrolled 2D.

## Known issues in the upstream snapshot (do not "fix" here — reference only)

- `amraux.py:4` `from tkinter import N` → crashes headless; `__init__` upstream
  does `from .amraux import *`, so `import jaxamr` fails on a cluster node.
- `amraux.py:82` hardcodes `.reshape((4, nx, ny))` (n_vars pinned to 4).
- `get_ghost_block_data` relies on `neighbor == -1` wrapping to the last (NaN)
  block — correct by construction but silently breaks on any `max_blk_num`
  off-by-one.
