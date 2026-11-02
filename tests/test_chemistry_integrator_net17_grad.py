"""The stiff integrator driving network 17: gradients through the fixed network with the Newton EOS."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.ode import rodas3 as ig  # noqa: E402


from luna.chemistry import networks  # noqa: E402
from tools.chem_integrator import validate_net17 as val  # noqa: E402


@pytest.fixture(scope="module")
def reference():
    return val.load()


GRAD_CASE = 214                       # n = 1e4, T0 = 100 K, G0 = 1.7, A_V = 1, Z = 1e-4
GRAD_COLS = (0, 1, 6, 7)              # x_H2, x_H+, x_HD, e (and zeta)


@pytest.fixture(scope="module")
def grad_problem(reference):
    names, cp, t, y0, ref, keep, steady, fixed = reference
    sel = np.asarray([GRAD_CASE])
    y = jnp.asarray(ref[sel, 20, :8])        # a mid-trajectory state (kept output 20)
    p = ig.broadcast_params(val.params(cp[sel], fixed), 1)
    tp, xh2, xhd = val.rates_net17.photo_state(y, p, "fixed")
    p = p._replace(photo_temp=tp, photo_x_h2=xh2, photo_x_hd=xhd)
    opt = ig.Options(rtol=1e-8, atol=tuple(1e-4 * a for a in networks.NET17.atol),
                     diff_fields=("zeta",), tiers=((None, 1),))

    def f(y, z):
        r = ig.integrate(y, p._replace(zeta=z), 1e4 * 3.15576e7, net=networks.NET17_FIXED, opt=opt)
        temp = val.rates_net17.rhs(r.y, r.params, fixes17="fixed")[1]["T"]
        return jnp.sum(jnp.log(temp) + jnp.log(r.y[:, 0]))
    z = p.zeta
    ff = jax.jit(f)
    fd = []
    for j in GRAD_COLS + (8,):
        e = np.zeros(9)
        e[j] = 1e-4 * (float(y[0, j]) if j < 8 else float(z[0]))
        fd.append((float(ff(y + e[:8], z + e[8])) - float(ff(y - e[:8], z - e[8]))) / 2e-4)
    return f, y, z, np.array(fd)


def _grad_error(grad_problem):
    """x dF/dx by AD against central differences, relative to the largest."""
    f, y, z, fd = grad_problem
    gy, gz = (np.asarray(g) for g in jax.jit(jax.grad(f, argnums=(0, 1)))(y, z))
    ad = np.concatenate([(gy * np.asarray(y))[0, list(GRAD_COLS)], gz * np.asarray(z)])
    err = np.max(np.abs(ad - fd)) / np.max(np.abs(ad))
    print(f"x dF/dx AD {ad}\n          FD {fd}\n error {err:.1e}")
    return err


def test_gradient_newton_eos_matches_fd(grad_problem):
    assert _grad_error(grad_problem) < 1e-4


def test_teeth_gradient_without_eos_derivative(grad_problem, monkeypatch):
    from luna.chemistry import thermo
    orig = thermo._residual

    def stopped(u, lnr, x, c1):
        g, dg = orig(u, lnr, x, c1)
        return jax.lax.stop_gradient(g), jax.lax.stop_gradient(dg)
    monkeypatch.setattr(thermo, "_residual", stopped)
    jax.clear_caches()
    assert _grad_error(grad_problem) > 1e-2
    jax.clear_caches()
