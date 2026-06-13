import sys
sys.path.insert(0, "/raven/u/uli/jax-hydro")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scripts.autodiff.sedov_ad import simulate_state, build_ic, _step

Nx, L, gamma = 32, 1.0, 5.0/3.0
n, t_end = 150, 0.06
dt = t_end / n
inv_dx = jnp.asarray(Nx / L, dtype=jnp.float64)
kw = dict(gamma=gamma, limiter=2, riemann_solver=1, order=1, inv_dx=inv_dx)

step = jax.jit(lambda u: _step(u, dt, **kw))
grad_probe = jax.jit(jax.grad(lambda u: jnp.sum(step(u) ** 2)))

U = build_ic(0.5, 1.0, Nx, L, gamma)
bad_k, bad_U = None, None
for k in range(n):
    g = grad_probe(U)
    if bool(jnp.any(~jnp.isfinite(g))):
        bad_k, bad_U = k, U
        print(f"first NaN single-step gradient at step k={k}", flush=True)
        break
    U = step(U)
else:
    print("no single-step NaN found in 150 steps (cotangent-dependent?)", flush=True)

if bad_k is not None:
    jax.config.update("jax_debug_nans", True)
    try:
        jax.grad(lambda u: jnp.sum(_step(u, dt, **kw) ** 2))(bad_U)
    except Exception:
        import traceback
        traceback.print_exc()
