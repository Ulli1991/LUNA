"""A16 cross-check: LUNA's adiabatic-expansion and top-hat runs against the exact answers, with mini-ramses and Arepo
on the same ICs printed alongside (arepo|ramses/cosmo_a16_xcheck in luna-reference-solutions, made by tools/a16_xcheck.py; context only, never gated:
both codes lack the (3 gamma - 5) internal-energy term, so their gamma 7/5 p~ follows a^-2 / a^-3(gamma-1) off)."""

import numpy as np
import pytest

from luna.reference_data import reference_path
import tests.test_cosmo_adiabatic_expansion as ad
import tests.test_cosmo_tophat_collapse as tc
from tests._helpers import check, gpu_only

CODES = ("ramses", "arepo")


def refs():
    """{code: npz} of the reduced mini-ramses and Arepo runs."""
    return {c: np.load(reference_path(f"{c}/cosmo_a16_xcheck/{c}.npz")) for c in CODES}


def context(z, key, a):
    """The other code's row at target a (interpolated between its bracketing snapshots), or None."""
    if f"{key}_targets" not in z:
        return None
    rows = z[f"{key}_targets"]
    r = rows[np.isclose(rows[:, 0], a)]
    return None if not len(r) or np.isnan(r[0, 3]) else r[0]


@pytest.mark.parametrize("gamma", list(ad.BOUND), ids=["5/3", "7/5"])
def test_adiabatic_xcheck(gamma):
    """LUNA p~ against a^(5 - 3 gamma) at a = 0.1 .. 1 (8^3), mini-ramses and Arepo printed next to it."""
    z = refs()
    key = f"adiabatic_g{'53' if gamma > 1.5 else '75'}"
    no_term = {a: (a / ad.A_I) ** -(5 - 3 * gamma) - 1 for a in ad.A_SNAP}
    for a, ep, ev, er in ad.run(gamma):
        others = " ".join(f"{c} {r[4]:.3e}" if (r := context(z[c], key, a)) is not None else f"{c} -" for c in CODES)
        print(f"CONTEXT adiabatic gamma={gamma:.4g} a={a}: max |p~/exact - 1| LUNA {ep:.3e} | {others} | "
              f"no (3 gamma - 5) term {abs(no_term[a]):.3e}")
        check(f"adiabatic xcheck gamma={gamma:.4g} a={a} |p~/exact - 1|", ep, ad.BOUND[gamma]["p"])
        check(f"adiabatic xcheck gamma={gamma:.4g} a={a} |v~/v0 - 1|", ev, ad.BOUND[gamma]["v"])
        check(f"adiabatic xcheck gamma={gamma:.4g} a={a} |rho~ - 1|", er, 1e-13)


def test_references_carry_provenance():
    """Both reference files name their code version and hold every run the comparison prints."""
    for c, z in refs().items():
        assert {"ramses": "mini-ramses 55daaa5", "arepo": "Arepo public 1fc135a"}[c] in str(z["readme"]), c
        for run in ("adiabatic_g53", "adiabatic_g75", "tophat_n64", "tophat_n128"):
            assert f"{run}_series" in z and np.isfinite(z[f"{run}_series"]).all(), (c, run)


@gpu_only
@pytest.mark.parametrize("n", [64, 128])
def test_tophat_xcheck_gpu(n):
    """LUNA core rho~ and v~ slope against closed FRW (turnaround a = 0.5, 1 + delta = 19.9 at a = 0.65), others alongside."""
    z = refs()
    gates, li, dp = tc.run(n)
    for a in tc.A_SNAP:
        others = " ".join(f"{c} {r[3]:+.3e}/{r[4]:+.3e}" if (r := context(z[c], f"tophat_n{n}", a)) is not None
                          else f"{c} -" for c in CODES)
        print(f"CONTEXT tophat N={n} a={a}: |core|/|slope| LUNA {gates[a][0]:.3e}/{gates[a][1]:.3e} | signed {others}")
    tc.gate(n, gates, li, dp)
