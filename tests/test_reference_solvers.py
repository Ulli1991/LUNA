"""The exact solvers that tests bound LUNA against, checked against published numbers to their printed digits.

Every bound is half a unit in the last printed digit of the publication (stated per check), not a measured
tolerance. Each section has teeth: the same check with a perturbed input (gamma) is asserted to fail.

- Exact Riemann solver (`luna.hydro.exact_riemann`): Toro (2009) Table 4.2 star states of the Table 4.1 tests,
  the test-1 profile against the closed-form wave relations, and Toro's vacuum-generating case (sec. 4.6).
- Sedov (`luna.hydro.exact_sedov`): Kamm & Timmes 2007 (LA-UR-07-2849) Tables 1-3 (similarity functions,
  gamma = 1.4, j = 1, 2, 3), Table 4 (energy integrals, alpha) and Table 5 (shock position and state at t = 1);
  the values as transcribed in ExactPack `exactpack/tests/test_sedov.py` (github.com/lanl/ExactPack, read
  2026-10-29), against a scipy.quad control of the energy integral.
- Guderley (`luna.hydro.exact_guderley`): the similarity exponent alpha against Ramsey, Kamm & Bolstad 2012
  (LA-UR-09-05853) Table 1, all eight (j, gamma), by the phase-plane classification on alpha -+ 5e-10.
- Noh (`tests/test_hydro_hard_noh_multid.py`, `tests/test_hydro_hard_noh_planar.py`): Noh 1987 (JCP 72, 78)
  closed form for gamma = 5/3: shock speed 1/3, post-shock rho = 4, 16, 64 and p = 4/3, 16/3, 64/3.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from luna.hydro import exact_guderley as eg
from luna.hydro import exact_riemann as er
from luna.hydro import exact_sedov as se
from tests._helpers import check

G14 = 1.4


def half_unit(printed):
    """Half a unit in the last printed digit of the decimal string `printed`."""
    mant = printed.lower().split("e")[0]
    exp = int(printed.lower().split("e")[1]) if "e" in printed.lower() else 0
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return 0.5 * 10.0 ** (exp - decimals)


# ---------------------------------------------------------------------------------------------------------- Riemann
# Toro (2009) Table 4.2 (sec. 4.3.3) as printed, (p*, u*, rho*_L, rho*_R); er.TORO_STAR holds the same numbers.
TORO_42 = {1: ("0.30313", "0.92745", "0.42632", "0.26557"), 2: ("0.00189", "0.00000", "0.02185", "0.02185"),
           3: ("460.894", "19.5975", "0.57506", "5.99924"), 4: ("46.0950", "-6.19633", "5.99242", "0.57511"),
           5: ("1691.64", "8.68975", "14.2823", "31.0426")}


# Toro Table 4.1 test 5 data as printed (6 digits: the rounded star states of tests 3 and 4); tests 1-4 are exact.
TORO_41_5 = (("5.99924", "19.5975", "460.894"), ("5.99242", "-6.19633", "46.0950"))


def toro_star(WL, WR, gamma):
    """(p*, u*, rho*_L, rho*_R) of the exact solver."""
    w = er.waves(WL, WR, gamma)
    return np.array([w["p"], w["u"], w["rho_L"], w["rho_R"]])


def toro_input_bound(k, gamma):
    """Per quantity, |d q / d input| times half a unit of each printed input, summed (test 5 only; else 0)."""
    if k != 5:
        return np.zeros(4)
    x = [float(v) for W in TORO_41_5 for v in W]
    q0, out = toro_star(tuple(x[:3]), tuple(x[3:]), gamma), np.zeros(4)
    for i, sv in enumerate(v for W in TORO_41_5 for v in W):
        h = 1e-6 * abs(x[i])
        xp = list(x)
        xp[i] += h
        out += np.abs((toro_star(tuple(xp[:3]), tuple(xp[3:]), gamma) - q0) / h) * half_unit(sv)
    return out


def toro_misses(k, gamma):
    """Per quantity |computed - Table 4.2| / (half a unit in its printed digit + propagated input rounding)."""
    WL, WR = er.TORO[k][:2]
    q = toro_star(WL, WR, gamma)
    allow = np.array([half_unit(s) for s in TORO_42[k]]) + toro_input_bound(k, gamma)
    return [abs(v - float(s)) / b for v, s, b in zip(q, TORO_42[k], allow)]


@pytest.mark.parametrize("k", range(1, 6))
def test_riemann_toro_table_4_2(k):
    """p*, u*, rho*_L, rho*_R of Toro test k within half a unit of Table 4.2's last printed digit (test 5: plus
    the Table 4.1 input rounding propagated linearly)."""
    assert er.TORO_STAR[k] == tuple(float(s) for s in TORO_42[k])
    for name, m in zip(("p*", "u*", "rho*_L", "rho*_R"), toro_misses(k, G14)):
        check(f"Toro test {k} {name} |exact - Table 4.2| / half unit", m, 1.0)


def test_riemann_toro_teeth_wrong_gamma():
    """Teeth: gamma = 1.41 misses Table 4.2 by more than half a unit in every test."""
    for k in range(1, 6):
        worst = max(toro_misses(k, 1.41))
        print(f"MEASURED teeth gamma=1.41 Toro test {k} worst miss / half unit = {worst:.3e}")
        assert worst > 1.0, f"Toro test {k} passes with gamma = 1.41"


def fan_misses(g_solver):
    """Max |u - a - xi|, |p rho^-g / (p_L rho_L^-g) - 1|, |J+ - J+_L| over test 1's left fan, the solver run at
    g_solver and the relations evaluated at gamma = 1.4."""
    WL, WR = er.TORO[1][:2]
    g = G14
    w = er.waves(WL, WR, g_solver)
    xi = np.linspace(*w["left"], 201)[1:-1]
    rho, u, p = er.sample(WL, WR, g_solver, xi)
    a, aL = np.sqrt(g * p / rho), np.sqrt(g * WL[2] / WL[0])
    return (float(np.abs(u - a - xi).max()), float(np.abs(p / rho ** g * WL[0] ** g / WL[2] - 1).max()),
            float(np.abs(u + 2 * a / (g - 1) - 2 * aL / (g - 1)).max()))


def test_riemann_toro_test1_profile_teeth():
    """Teeth: the profile sampled with gamma = 1.41 breaks the gamma = 1.4 fan relations."""
    m = fan_misses(1.41)
    print(f"MEASURED teeth Toro 1 fan relations, solver at gamma 1.41: {m}")
    assert min(m) > 1e-14


def test_riemann_toro_test1_profile():
    """Test 1 sampled at t = 0.25: left fan (u - a = xi, isentropic, J+ constant), star plateaus equal the star
    states, the right shock obeys Rankine-Hugoniot, the outer states are the initial ones."""
    WL, WR, x0, t = er.TORO[1]
    g = G14
    w = er.waves(WL, WR, g)
    head, tail = w["left"]
    S = w["right"][0]
    aL = np.sqrt(g * WL[2] / WL[0])
    check("Toro 1 fan head + a_L", abs(head + aL), 1e-15)
    xi = np.linspace(head, tail, 201)[1:-1]
    rho, u, p = er.sample(WL, WR, g, xi)
    a = np.sqrt(g * p / rho)
    check("Toro 1 fan max |u - a - xi|", float(np.abs(u - a - xi).max()), 1e-14)
    check("Toro 1 fan max |p rho^-g / (p_L rho_L^-g) - 1|", float(np.abs(p / rho ** g * WL[0] ** g / WL[2] - 1).max()),
          1e-14)
    check("Toro 1 fan max |J+ - J+_L|", float(np.abs(u + 2 * a / (g - 1) - 2 * aL / (g - 1)).max()), 1e-14)
    plateaus = {(tail + 0.5 * (w["u"] - tail)): (w["rho_L"], w["u"], w["p"]),
                (w["u"] + 0.5 * (S - w["u"])): (w["rho_R"], w["u"], w["p"]),
                (head - 0.1): WL, (S + 0.1): WR}
    for x, want in plateaus.items():
        got = [v[0] for v in er.sample(WL, WR, g, np.array([x]))]
        check(f"Toro 1 plateau xi={x:.4f} max |W - W_exact|", max(abs(a_ - b) for a_, b in zip(got, want)), 0.0)
    down, up = (w["rho_R"], w["u"], w["p"]), WR

    def flux(r, v, q):
        return np.array([r * v, r * v * v + q, v * (q / (g - 1) + 0.5 * r * v * v + q)])

    def cons(r, v, q):
        return np.array([r, r * v, q / (g - 1) + 0.5 * r * v * v])
    rh = np.abs(flux(*down) - flux(*up) - S * (cons(*down) - cons(*up))).max()
    check("Toro 1 right shock max |[F] - S [U]|", float(rh), 1e-14)
    # Table 4.2 quantities through the sampled profile (not only through waves())
    for name, m in zip(("p*", "u*", "rho*_L", "rho*_R"),
                       [abs(v - float(s)) / half_unit(s) for v, s in zip(
                           (plateaus[next(iter(plateaus))][2], plateaus[next(iter(plateaus))][1],
                            er.sample(WL, WR, g, np.array([0.5 * (tail + w["u"])]))[0][0],
                            er.sample(WL, WR, g, np.array([0.5 * (S + w["u"])]))[0][0]), TORO_42[1])]):
        check(f"Toro 1 sampled {name} |exact - Table 4.2| / half unit", m, 1.0)


VAC_L, VAC_R = (1.0, -4.0, 0.4), (1.0, 4.0, 0.4)       # Toro sec. 4.6: u_R - u_L = 8 > 2 (a_L + a_R) / (g - 1)


def test_riemann_vacuum_refused():
    """A vacuum-generating pair (Toro eq. 4.40 violated) is refused, not answered with a floor pressure; the
    test-2 pair, which satisfies eq. 4.40 (4 < 7.48), is not."""
    aL = np.sqrt(G14 * VAC_L[2] / VAC_L[0])
    assert VAC_R[1] - VAC_L[1] > 4 * aL / (G14 - 1)
    with pytest.raises(ValueError, match="vacuum"):
        er.star(VAC_L, VAC_R, G14)
    with pytest.raises(ValueError, match="vacuum"):
        er.sample(VAC_L, VAC_R, G14, np.zeros(3))
    er.star(*er.TORO[2][:2], G14)


# ------------------------------------------------------------------------------------------------------------- Sedov
# Kamm & Timmes 2007 Tables 1-3 (gamma = 1.4, omega = 0): rows (V, lam, f, g, h), 4 decimals; as transcribed in
# ExactPack exactpack/tests/test_sedov.py (TestSedovFunctionsTable1-3).
KT_TABLES = {
    1: ("0.5500 0.5400 0.5300 0.5200 0.5100 0.5000 0.4950 0.4900 0.4850 0.4820 0.4800 0.4790 0.4780 0.4770 0.4765 "
        "0.4763 0.4762 0.4762",
        "0.9797 0.9420 0.9013 0.8565 0.8050 0.7419 0.7029 0.6553 0.5925 0.5396 0.4912 0.4589 0.4161 0.3480 0.2810 "
        "0.2320 0.1680 0.1040",
        "0.9699 0.9157 0.8598 0.8017 0.7390 0.6677 0.6263 0.5780 0.5173 0.4682 0.4244 0.3957 0.3580 0.2988 0.2410 "
        "0.1989 0.1440 0.0891",
        "0.8620 0.6662 0.5159 0.3981 0.3020 0.2201 0.1823 0.1453 0.1075 0.0826 0.0641 0.0535 0.0415 0.0263 0.0153 "
        "0.0095 0.0042 0.0013",
        "0.9159 0.7917 0.6922 0.6119 0.5458 0.4905 0.4661 0.4437 0.4230 0.4112 0.4037 0.4001 0.3964 0.3929 0.3911 "
        "0.3905 0.3901 0.3900"),
    2: ("0.4166 0.4100 0.4050 0.4000 0.3950 0.3900 0.3820 0.3770 0.3720 0.3670 0.3640 0.3620 0.3600 0.3585 0.3578 "
        "0.3575 0.3573 0.3572 0.3572 0.3571 0.3571 0.3571 0.3571",
        "0.9998 0.9802 0.9644 0.9476 0.9295 0.9096 0.8725 0.8442 0.8094 0.7629 0.7242 0.6894 0.6390 0.5745 0.5180 "
        "0.4748 0.4222 0.3654 0.3000 0.2500 0.2000 0.1500 0.1000",
        "0.9996 0.9645 0.9374 0.9097 0.8812 0.8514 0.7999 0.7638 0.7226 0.6720 0.6327 0.5990 0.5521 0.4943 0.4448 "
        "0.4074 0.3620 0.3133 0.2572 0.2143 0.1714 0.1286 0.0857",
        "0.9972 0.7651 0.6281 0.5161 0.4233 0.3450 0.2427 0.1892 0.1415 0.0974 0.0718 0.0545 0.0362 0.0208 0.0123 "
        "0.0079 0.0044 0.0021 0.0008 0.0003 0.0001 0.0000 0.0000",
        "0.9984 0.8658 0.7829 0.7122 0.6513 0.5982 0.5266 0.4884 0.4545 0.4241 0.4074 0.3969 0.3867 0.3794 0.3760 "
        "0.3746 0.3737 0.3732 0.3730 0.3729 0.3729 0.3729 0.3729"),
    3: ("0.3300 0.3250 0.3200 0.3120 0.3060 0.3000 0.2950 0.2915 0.2890 0.2870 0.2860 0.2857 0.2857 0.2857 0.2857 "
        "0.2857",
        "0.9913 0.9773 0.9622 0.9342 0.9080 0.8747 0.8359 0.7950 0.7493 0.6788 0.5794 0.4560 0.3600 0.2960 0.2000 "
        "0.1040",
        "0.9814 0.9529 0.9238 0.8745 0.8335 0.7872 0.7398 0.6952 0.6497 0.5844 0.4971 0.3909 0.3086 0.2537 0.1714 "
        "0.0891",
        "0.8388 0.6454 0.4984 0.3248 0.2275 0.1508 0.0968 0.0620 0.0379 0.0174 0.0052 0.0009 0.0001 0.0000 0.0000 "
        "0.0000",
        "0.9116 0.7992 0.7082 0.5929 0.5238 0.4674 0.4273 0.4021 0.3857 0.3732 0.3672 0.3656 0.3655 0.3655 0.3655 "
        "0.3655"),
}
KT_HALF = 0.5e-4                                       # half a unit of the 4th decimal


def kt_row_miss(j, gamma, row):
    """min over the exact curve of max_k |q_k - row_k| / KT_HALF, q = (V, lam, f, g, h): <= 1 iff a point of the
    closed form rounds to the printed row (V is itself printed to 4 decimals). The curve is searched in
    ln(V - V0) (lam ~ (V - V0)^|a2| at the origin), down to V - V0 = 1e-300."""
    V0, V2 = se.v_range(gamma, j)
    top = min(row[0] + KT_HALF, V2) - V0
    if top <= 0:
        return np.inf

    def miss(u):
        dV = np.exp(np.atleast_1d(u))
        q = np.vstack([V0 + dV, *se._similarity_dv(dV, gamma, j)])
        return np.abs(q - np.asarray(row)[:, None]).max(0) / KT_HALF
    u = np.linspace(np.log(1e-300), np.log(top), 20001)
    m = miss(u)
    i = int(np.argmin(m))
    lo, hi = u[max(i - 1, 0)], u[min(i + 1, len(u) - 1)]
    best = minimize_scalar(lambda x: miss(x)[0], bounds=(lo, hi), method="bounded", options=dict(xatol=1e-14))
    return float(min(m[i], best.fun))


@pytest.mark.parametrize("j", (1, 2, 3))
def test_sedov_kamm_timmes_tables_1_3(j):
    """Every printed row (V, lam, f, g, h) of KT Table j is the rounding of a point on the closed form."""
    rows = np.array([[float(v) for v in col.split()] for col in KT_TABLES[j]]).T
    worst = max(kt_row_miss(j, G14, r) for r in rows)
    check(f"Sedov j={j} KT Table {j} worst row miss / half unit (4th decimal)", worst, 1.0)


def test_sedov_kamm_timmes_tables_teeth():
    """Teeth: gamma = 1.41 puts rows of every table off the curve by more than half a unit."""
    for j in (1, 2, 3):
        rows = np.array([[float(v) for v in col.split()] for col in KT_TABLES[j]]).T
        worst = max(kt_row_miss(j, 1.41, r) for r in rows)
        print(f"MEASURED teeth gamma=1.41 KT Table {j} worst row miss / half unit = {worst:.3e}")
        assert worst > 1.0, f"KT Table {j} passes with gamma = 1.41"


def alpha_control(gamma, j):
    """(kinetic, thermal) by scipy.quad (QAGS) directly in dV = V - V0 (the endpoint singularity left to QAGS),
    independent of the sigma substitution in exact_sedov.energy_integrals."""
    V0, V2 = se.v_range(gamma, j)
    a2 = -(gamma - 1) / (2.0 * (gamma - 1) + j)
    a1 = (j + 2) * gamma / (2.0 + j * (gamma - 1)) * (2.0 * j * (2.0 - gamma) / (gamma * (j + 2) ** 2) - a2)
    e = (2.0 + j * (gamma - 1)) / 2.0

    def dens(dV, part):
        lam, f, g, h = se._similarity_dv(dV, gamma, j)
        V = V0 + dV
        dlam = lam * (-2.0 / (j + 2) / V - a2 / dV + a1 * e / (1.0 - e * V))
        w = (0.5 * (gamma + 1) / (gamma - 1) * g * (2.0 / (gamma + 1)) ** 2 * f ** 2 if part == 0
             else (2.0 / (gamma + 1)) / (gamma - 1) * h)
        return w * lam ** (j - 1) * dlam
    return tuple(se._ALPHA_PREFACTOR[j] * quad(dens, 0.0, V2 - V0, args=(k,), epsabs=0.0, epsrel=1e-12,
                                               limit=500)[0] for k in (0, 1))


# KT Table 4 (gamma = 1.4, t = 1): J1 (eval1) printed; alpha = c_j (J1 + 2 J2 / (gamma - 1)), c = 1/2, pi, 2 pi,
# in KT's planar convention (one side of the slab; LUNA's planar alpha is the whole slab, twice KT's).
KT_J1 = {1: "0.197928", 2: "6.54053e-02", 3: "2.96269e-02"}
KT_C = {1: 0.5, 2: np.pi, 3: 2.0 * np.pi}
KT_PLANAR = {1: 0.5, 2: 1.0, 3: 1.0}
# Table 5 spherical E0 = 0.851072 puts r2 = 1.0 at t = 1 (rho0 = 1), i.e. alpha_sph = 0.851072 to its 6 digits.
KT_E0_SPH = "0.851072"


def test_sedov_alpha_against_kamm_timmes():
    """alpha against the control quadrature (round-off of quad, 1e-10 relative); J1 within half a unit of KT Table 4;
    the spherical alpha within half a unit of KT Table 5's E0 = 0.851072 (r2 = 1 at t = 1)."""
    for j in (1, 2, 3):
        kin, th = se.energy_integrals(G14, j)
        ck, ct = alpha_control(G14, j)
        alpha = se.sedov_alpha(G14, j)
        check(f"Sedov j={j} |alpha / alpha_control - 1|", abs(alpha / (ck + ct) - 1), 1e-10)
        J1 = KT_PLANAR[j] * kin / KT_C[j]
        J2 = KT_PLANAR[j] * th * (G14 - 1) / (2 * KT_C[j])
        print(f"MEASURED Sedov j={j} alpha={alpha:.9f} (KT convention {KT_PLANAR[j] * alpha:.9f}) J1={J1:.9g} "
              f"J2={J2:.9g}")
        check(f"Sedov j={j} |J1 - KT Table 4| / half unit", abs(J1 - float(KT_J1[j])) / half_unit(KT_J1[j]), 1.0)
    check("Sedov j=3 |alpha - KT E0 0.851072| / half unit",
          abs(se.sedov_alpha(G14, 3) - float(KT_E0_SPH)) / half_unit(KT_E0_SPH), 1.0)


def test_sedov_alpha_teeth():
    """Teeth: gamma = 1.41 misses KT's J1 (every j) and E0 by more than half a unit."""
    for j in (1, 2, 3):
        J1 = KT_PLANAR[j] * se.energy_integrals(1.41, j)[0] / KT_C[j]
        miss = abs(J1 - float(KT_J1[j])) / half_unit(KT_J1[j])
        print(f"MEASURED teeth gamma=1.41 j={j} J1 miss / half unit = {miss:.3e}")
        assert miss > 1.0
    assert abs(se.sedov_alpha(1.41, 3) - float(KT_E0_SPH)) > half_unit(KT_E0_SPH)


# KT Table 5 planar and cylindrical rows (rho2, u2, e2, p2) at r2 = 0.5, 0.75, t = 1, rho0 = 1.
KT_T5 = {1: (0.5, ("6.0", "2.77778e-01", "3.85802e-02", "9.25926e-02")),
         2: (0.75, ("6.0", "3.125e-01", "4.88281e-02", "1.17188e-01"))}


def test_sedov_post_shock_teeth():
    """Teeth: the post-shock state at gamma = 1.41 misses KT Table 5 (rho2 = 6 at gamma = 1.4) in both rows."""
    for j, (r2, printed) in KT_T5.items():
        rho = se.solution(np.array([r2]), 1.0, se.sedov_alpha(1.41, j) * r2 ** (j + 2), 1.0, 1.41, j)[0]
        miss = abs(rho[0] - float(printed[0])) / half_unit(printed[0])
        print(f"MEASURED teeth Sedov j={j} gamma=1.41 rho2 miss / half unit = {miss:.3e}")
        assert miss > 1.0 + 1e-9


@pytest.mark.parametrize("j", (1, 2))
def test_sedov_post_shock_state_table_5(j):
    """solution() at R_s = r2 (E0 = alpha r2^(j+2)): rho2, u2, e2, p2 within half a unit of KT Table 5 (plus 1e-9 of
    it: the cylindrical p2 = 0.1171875 is a rounding tie, printed 1.17188e-01)."""
    r2, printed = KT_T5[j]
    E0 = se.sedov_alpha(G14, j) * r2 ** (j + 2)
    rho, u, p, Rs = se.solution(np.array([r2]), 1.0, E0, 1.0, G14, j)
    check(f"Sedov j={j} |R_s - r2|", abs(Rs - r2), 1e-14)
    got = (rho[0], u[0], p[0] / ((G14 - 1) * rho[0]), p[0])
    for name, v, sv in zip(("rho2", "u2", "e2", "p2"), got, printed):
        check(f"Sedov j={j} {name} |exact - KT Table 5| / half unit", abs(v - float(sv)) / half_unit(sv), 1.0 + 1e-9)


# ---------------------------------------------------------------------------------------------------------- Guderley
@pytest.mark.parametrize("key", list(eg.ALPHA_RKB), ids=[f"j{j}-g{g:.3f}" for j, g in eg.ALPHA_RKB])
def test_guderley_exponent_rkb_table_1(key):
    """The phase-plane classification flips between alpha -+ 5e-10 of RKB Table 1: the eigenvalue lies within half
    a unit of the 9th printed decimal (independent of exponent()'s bracket); exponent() lies in that interval."""
    j, g = key
    a = eg.ALPHA_RKB[key]
    lo, hi = eg._classify(1.0 / (a - 5e-10), g, j), eg._classify(1.0 / (a + 5e-10), g, j)
    print(f"MEASURED Guderley j={j} gamma={g:.4f} classify(alpha -+ 5e-10) = {lo}, {hi}")
    assert lo != hi, f"no eigenvalue within 5e-10 of RKB alpha {a}"
    check(f"Guderley j={j} gamma={g:.4f} |1/exponent - RKB| / 5e-10", abs(1.0 / eg.exponent(j, g) - a) / 5e-10, 1.0)


def test_guderley_teeth_wrong_gamma():
    """Teeth: with gamma = 1.41 the classification does not flip within -+ 5e-10 of the gamma = 1.4 alpha."""
    for j in (2, 3):
        a = eg.ALPHA_RKB[(j, 1.4)]
        lo, hi = eg._classify(1.0 / (a - 5e-10), 1.41, j), eg._classify(1.0 / (a + 5e-10), 1.41, j)
        print(f"MEASURED teeth Guderley j={j} gamma=1.41 classify = {lo}, {hi}")
        assert lo == hi


# --------------------------------------------------------------------------------------------------------------- Noh
def test_noh_against_noh_1987():
    """The Noh exact used by the Noh tests, gamma = 5/3, rho0 = 1, |u0| = 1, p0 -> 0: shock speed 1/3, post-shock
    rho = 4, 16, 64 and p = 4/3, 16/3, 64/3, pre-shock rho = (1 + t / r)^(d - 1) (Noh 1987 sec. 4, closed form);
    exact rationals, bound FP64 round-off (4 ulp)."""
    import tests.test_hydro_hard_noh_multid as nm
    import tests.test_hydro_hard_noh_planar as npl
    assert (nm.G, nm.RHO0, abs(nm.U0)) == (5.0 / 3.0, 1.0, 1.0) and (npl.G, npl.RHO0, abs(npl.U0)) == (5 / 3, 1, 1)
    eps = 4 * np.finfo(float).eps
    check("Noh shock speed |D - 1/3| (multid, planar)", max(abs(nm.D - 1 / 3), abs(npl.D - 1 / 3)), eps)
    check("Noh planar |rho_s - 4|, |p_s - 4/3|", max(abs(npl.RHO_S - 4), abs(npl.P_S - 4 / 3)), 4 * eps)
    for d, (r_want, p_want) in {1: (4.0, 4 / 3), 2: (16.0, 16 / 3), 3: (64.0, 64 / 3)}.items():
        check(f"Noh d={d} |rho_s / {r_want:g} - 1|, |p_s / p - 1|",
              max(abs(nm.rho_s(d) / r_want - 1), abs(nm.p_s(d) / p_want - 1)), eps)
        r, t = np.array([0.5, 0.8]), 0.6
        rho = nm.exact_point(d, r, t)[0]
        check(f"Noh d={d} pre-shock |rho / (1 + t/r)^(d-1) - 1|", float(np.abs(rho / (1 + t / r) ** (d - 1) - 1).max()),
              eps)


def test_noh_teeth_wrong_gamma(monkeypatch):
    """Teeth: the Noh helpers with the gamma = 7/5 compression (6) miss rho_s = 4, 16, 64 by far more than round-off."""
    import tests.test_hydro_hard_noh_multid as nm
    monkeypatch.setattr(nm, "COMP", (1.4 + 1.0) / (1.4 - 1.0))
    for d, want in ((1, 4.0), (2, 16.0), (3, 64.0)):
        miss = abs(nm.rho_s(d) / want - 1)
        print(f"MEASURED teeth Noh d={d} gamma=7/5 |rho_s / {want:g} - 1| = {miss:.3e}")
        assert miss > 4 * np.finfo(float).eps
