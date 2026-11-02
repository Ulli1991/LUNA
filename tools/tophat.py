"""The top-hat cross-check: parametric closed-FRW interior against the ODE (luna.cosmology.exact_tophat)."""
import numpy as np

from luna.cosmology.exact_tophat import TopHat


if __name__ == "__main__":
    th = TopHat(0.01, 1.0 / 101.0)
    a_ta = th.a_turnaround()
    a = np.array([0.02, 0.05, 0.5 * a_ta, a_ta, 1.3 * a_ta, 1.55 * a_ta])
    p, o = th.one_plus_delta(a), th.integrate(a)
    print(f"delta_i 0.01 at a_i 1/101: th_i {th.th_i:.6e}, turnaround a {a_ta:.6f}, 1 + delta there {th.one_plus_delta([a_ta])[0]:.6f} "
          f"(9 pi^2 / 16 = {9 * np.pi ** 2 / 16:.6f})")
    for ai, pi, oi in zip(a, p, o):
        print(f"  a {ai:.5f}  parametric {pi:.12f}  ODE {oi:.12f}  rel diff {abs(pi / oi - 1):.1e}")
