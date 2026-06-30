"""Multi-GPU unit test for solve_poisson_fmg_shardmap.

Setup: 64³ uniform grid on a 2×2×2 device mesh (= 8 GPUs). Source is a
single cos(kx) mode with zero mean. Analytic Φ = -(4πG·amp/k²)·cos(kx).

We compare the FMG-shardmap output Φ to the analytic field cell-by-cell
and report:
  - max |Φ_num - Φ_analytic|  (absolute error)
  - max |Φ_num - Φ_analytic| / max|Φ_analytic|  (relative error)
  - correlation between Φ_num and Φ_analytic (should be ≈ -1 if attractive)

Pass: rel err < 1% (discretisation floor of 7-point stencil at N=64).
"""

import pytest
# Runs on 8 devices (conftest provides 8 virtual CPU devices) or 1; the
# shard_map FMG Poisson is the same code path the multi-GPU runs use.
pytestmark = pytest.mark.slow


import os
import jax
import jax.numpy as jnp
import numpy as np

# Bootstrap distributed JAX, matching the LUNA hydro runs.
if "JAX_COORDINATOR_ADDRESS" in os.environ:
    jax.distributed.initialize(
        coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ.get("SLURM_NTASKS", "1")),
        process_id=int(os.environ.get("SLURM_PROCID", "0")),
    )

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

from gravity.poisson_multigrid import solve_poisson_fmg_shardmap


def main():
    N_GLOBAL = 64
    L = 1.0
    DX = L / N_GLOBAL
    AMP = 1e-4
    K = 2.0 * jnp.pi / L
    G = 7.5
    FOURPIG = 4.0 * jnp.pi * G

    is_main = jax.process_index() == 0
    n_devices = len(jax.devices())
    if is_main:
        print(f"# devices: {n_devices}")
        print(f"Test: N={N_GLOBAL}, L={L}, k={float(K):.4f}, G={G}, amp={AMP}")

    # Build 2x2x2 (or 1x1x1 if 1 GPU) mesh.
    if n_devices == 1:
        mesh_shape = (1, 1, 1)
    elif n_devices == 8:
        mesh_shape = (2, 2, 2)
    else:
        pytest.skip(f"needs 1 or 8 devices; got {n_devices}")
    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, ("x", "y", "z"))
    spec = P("x", "y", "z")
    sharding = NamedSharding(mesh, spec)

    if is_main:
        print(f"Mesh shape: {mesh_shape}")

    # Construct rho = ρ₀ (1 + amp·cos(kx)) globally. Each process builds the full
    # global array (it's only 64³ floats = 2 MB), then we device_put with sharding.
    x_centres = (jnp.arange(N_GLOBAL) + 0.5) * DX
    X, Y, Z = jnp.meshgrid(x_centres, x_centres, x_centres, indexing="ij")
    rho_global = 1.0 + AMP * jnp.cos(K * X)        # background + perturbation
    rho_sharded = jax.device_put(rho_global, sharding)

    # Analytic Φ for the perturbation (after subtracting the mean).
    # ∇²Φ = 4πG·(ρ - ρ₀) = 4πG·amp·cos(kx) → Φ_analytic = -(4πG·amp/k²)·cos(kx)
    phi_amp = FOURPIG * AMP / (K * K)
    phi_analytic = -phi_amp * jnp.cos(K * X)
    phi_analytic_sharded = jax.device_put(phi_analytic, sharding)

    if is_main:
        print(f"Φ amplitude (analytic): {float(phi_amp):.6e}")

    # Call the shard_map FMG solver. It does its own mean subtraction.
    phi_num = solve_poisson_fmg_shardmap(
        rho_sharded,
        DX,
        FOURPIG,
        boundary=0,         # periodic
        mesh=mesh,
        n_levels=None,      # auto
        n_vcycles_per_level=1,
        n_extra_vcycles=0,
        n_pre=2,
        n_post=2,
        n_coarsest=50,
        subtract_mean=True,
    )

    # Stay in JAX (sharded). jnp aggregations over a NamedSharded array
    # reduce globally → the resulting scalars are replicated on every
    # process, and float() syncs them to the host on rank 0.
    phi_num_mean = jnp.mean(phi_num)
    phi_anal_mean = jnp.mean(phi_analytic_sharded)
    phi_num_zm = phi_num - phi_num_mean
    phi_anal_zm = phi_analytic_sharded - phi_anal_mean

    diff = phi_num_zm - phi_anal_zm
    abs_err = jnp.max(jnp.abs(diff))
    max_anal = jnp.max(jnp.abs(phi_anal_zm))
    max_num = jnp.max(jnp.abs(phi_num_zm))
    rel_err = abs_err / max_anal
    dot = jnp.sum(phi_num_zm * phi_anal_zm)
    norm_n = jnp.sqrt(jnp.sum(phi_num_zm * phi_num_zm))
    norm_a = jnp.sqrt(jnp.sum(phi_anal_zm * phi_anal_zm))
    corr = dot / (norm_n * norm_a)

    if is_main:
        print(f"max |Φ_num|                  = {float(max_num):.6e}")
        print(f"max |Φ_analytic|             = {float(max_anal):.6e}")
        print(f"max |Φ_num - Φ_analytic|     = {float(abs_err):.6e}")
        print(f"max |Φ_num - Φ_analytic|/|Φ| = {float(rel_err):.6e}")
        print(f"corr(Φ_num, Φ_analytic)      = {float(corr):.6f}  (should be ≈ +1)")

        passed = float(rel_err) < 1e-2
        print(f"{'PASS' if passed else 'FAIL'}: rel err {'<' if passed else '>'} 1e-2")

    # ------------------------------------------------------------------
    # Now test the GRADIENT computation under shard_map. That's the next
    # step in the self_grav pipeline. Analytic: gx = -∂Φ/∂x ∝ -sin(kx).
    # ------------------------------------------------------------------
    from functools import partial
    from jax.experimental.shard_map import shard_map
    from gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap

    halo_factory = make_scalar_halo_3d_shardmap(mesh)
    halo_fn_sm = halo_factory(0)  # periodic

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh, in_specs=(spec,), out_specs=spec)
    def grad_x_shardmap(phi_local):
        phi_pad = halo_fn_sm(phi_local)
        return -(phi_pad[2:, 1:-1, 1:-1] - phi_pad[:-2, 1:-1, 1:-1]) * (0.5 / DX)

    gx_num = grad_x_shardmap(phi_num)
    gx_analytic = -(FOURPIG * AMP / K) * jnp.sin(K * X)
    gx_anal_sh = jax.device_put(gx_analytic, sharding)

    gx_max_num = jnp.max(jnp.abs(gx_num))
    gx_max_anal = jnp.max(jnp.abs(gx_anal_sh))
    gx_diff = jnp.max(jnp.abs(gx_num - gx_anal_sh))
    gx_rel = gx_diff / gx_max_anal
    gx_corr = jnp.sum(gx_num * gx_anal_sh) / (
        jnp.sqrt(jnp.sum(gx_num * gx_num)) * jnp.sqrt(jnp.sum(gx_anal_sh * gx_anal_sh))
    )

    if is_main:
        print()
        print("--- gradient test ---")
        print(f"max |gx_num|                = {float(gx_max_num):.6e}")
        print(f"max |gx_analytic|           = {float(gx_max_anal):.6e}")
        print(f"max |gx_num - gx_analytic|  = {float(gx_diff):.6e}")
        print(f"max |gx_diff|/|gx|          = {float(gx_rel):.6e}")
        print(f"corr(gx_num, gx_analytic)   = {float(gx_corr):.6f}")
        gx_passed = float(gx_rel) < 1e-2
        print(f"{'PASS' if gx_passed else 'FAIL'}: gradient rel err {'<' if gx_passed else '>'} 1e-2")

    assert float(rel_err) < 1e-2, f"FMG-shardmap Poisson rel err = {float(rel_err):.3e}"
    assert float(gx_rel) < 1e-2, f"shard_map gradient rel err = {float(gx_rel):.3e}"


def test_poisson_multigrid_shardmap():
    """FMG-shardmap Poisson + gradient vs analytic single-mode field."""
    main()


if __name__ == "__main__":
    main()
