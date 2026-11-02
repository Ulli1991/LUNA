"""Exact compressible linear theory of single-mode Richtmyer-Meshkov for two ideal gases.

Reflected rarefaction: Cobos-Campos & Wouchuk, PRE 96, 013102 (2017) [CW17].
Reflected shock:       Cobos-Campos & Wouchuk, PRE 93, 053111 (2016) [CW16].

Geometry (both papers): incident shock in fluid b, transmitted into fluid a,
R0 = rho_a0/rho_b0.  Velocities in units of k*psi0*|Di|; ripple psi(t) ~ psi_inf + dv_inf*t,
same sign convention as the pre-shock amplitude psi0 > 0.

Typos found and resolved against the papers' worked examples are listed in TYPOS.

    base_flow(ga, gb, R0, Mi)        one-dimensional shock refraction (Riemann problem)
    rm_linear(ga, gb, R0, Mi, nit)   asymptotic velocity, psi_inf, iterates, approximations
    rm_setup(...)                    dimensional wrapper for a (gamma, rho, p0, Mach, direction) setup
    psi_history(setup, t)            psi_i(t)/psi0 and its velocity from t = 0+ (Bessel series, CW17 (119)/(129))

Time history: F's are solved exactly as power series in w = exp(-q) of the fluid with the larger post-shock
sound speed (its Bessel coefficients; series in the slower fluid diverge inside |w| < 1), and checked against a
numerical Bromwich inversion of H(s) = F_a1 - F_a2 evaluated from the functional equations at complex s.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

sh, ch = np.sinh, np.cosh

TYPOS = [
    "CW17 Eq.(46) and App.B: printed '1/kappa_t = 7.83164' is kappa_t = (dp/drho)_H/c_af^2 = 4r/((g+1-(g-1)r)^2 (p_f/p_0)), r = rho_af/rho_a0.",
    "CW17 Eq.(51a): sign; psi_rh = 1 + (c_b1 - U1)/|Di| (all fronts: psi/psi0 = 1 - w_front/D_i, signed lab speeds).",
    "CW17 Eq.(58): first bracket must be 2(psi_rt-psi_rh)M1/(zeta_rh-M1)+2psi_rt (else w0 does not reproduce v0_yb); alpha_j taken from v_init(zeta) directly.",
    "CW17 Eq.(71) prefactor c_b1/Di is right (B8 reproduced); Eq.(72)/(75) c_bf/Di is not used.",
    "CW17 App.B: chi_t = -artanh(beta_t) (printed -arctan); q_b = arsinh(N sinh q_a) (printed arcsin/sin).",
    "CW17 Eq.(47): first term needs factor sinh(chi_t) as in Eq.(80a)/CW16 (A1); B10 reproduced only with it.",
    "CW17 Eq.(A5): last term coefficient is phi_a2(0)-eps_a2(0), not phi_a2'(0); psi_inf from consistent derivative.",
    "CW16 Eq.(A4): alpha_m10 = (M^2+1)/(2 M^2 beta) (power 2 missing), alpha_a11 = +1/(2 beta_t); "
    "reflected side alpha_b10 = -(Mr^2+1)/(2Mr^2 beta_r), alpha_b11 = -1/(2 beta_r) (fixed by D9).",
    "CW16 Eq.(A1)/(C4): eta_r arguments are q_b + chi_r (as in A6/D9), not q_b - chi_r.",
    "CW16 Eq.(4): c_b1/c_b0 has (g-1)M^2+2, not +1.",
    "CW16 Eq.(32): 4(D_r+U_i)/(U1-U_i) means |reflected shock speed rel. to fluid b_f| / |U1-U|.",
    "CW16 (D12): F_b^[5] printed 0.012538 is 0.0122538; (D16) v_ib printed = u_i - F_b contradicts Eqs.(30),(31) (u_i + F_b).",
    "CW17 Eq.(123b) and (B28): stray factor (2n+1); H_ai = sum (f_a1 - f_a2)/s^(2n+1) as in (84b),(85). B28 prints "
    "(2n+1) h_i0: 0.07193 = 3 x 0.0239777; B30a (u_ai Taylor) uses the true h_i0 and is reproduced.",
    "CW17 (B30a): tau^7 coefficient printed 0.000000165 is 0.00000165 (= h^(7)/7! = 0.008324617/5040).",
    "CW17 (B19a): s^-5 constant printed 0.0010 is 0.0100; s^-1 f_a1 coefficient printed 1.0437, ours "
    "1 - lambda_t eps_a2(0) = 1.04386 (all other B18a/B19a numbers reproduced): unresolved, likely misprint.",
    "CW17 (B18b),(B19b): wrong subscripts; P = pi1/2s^2 + (pi3-3pi1)/8s^4 + (pi5-5pi3+10pi1)/32s^6 "
    "+ (pi7-7pi5+21pi3-35pi1)/128s^8, H = om1/2s + (om3-om1)/8s^3 + (om5-3om3+2om1)/32s^5 + (om7-5om5+9om3-5om1)/128s^7.",
    "CW17 (B22),(129): use u_i^[0] in place of the exact u_i = -sum omega_n; we use the exact series. "
    "B22's tau_a = 2.28014351/ns t needs D_i = 49.79 um/ns (c_b0 = 3.25425), not 3.254 x 15.3.",
    "CW16 (D7): pi_1^a printed -0.1518432 is -0.1584321 (digits permuted); (D8) p_t1 = -0.033885 needs -0.1584321.",
    "CW17 p.27 (Fig.18 text): gamma_a = 1.8, gamma_b = 1.45 swapped; App. B and Fig. 3 use gamma_a = 1.45, gamma_b = 1.8.",
]


def base_flow(ga, gb, R0, Mi):
    """Base flow in units rho_b0 = c_b0 = 1; speeds are magnitudes along the incident shock direction."""
    p0, rb0, ra0, cb0 = 1.0 / gb, 1.0, R0, 1.0
    ca0 = np.sqrt(ga * p0 / ra0)
    zi = 2 * gb * (Mi**2 - 1) / (gb + 1)
    p1 = p0 * (1 + zi)
    rb1 = rb0 * (gb + 1) * Mi**2 / ((gb - 1) * Mi**2 + 2)
    cb1 = np.sqrt(gb * p1 / rb1)
    U1 = 2 / (gb + 1) * (Mi - 1 / Mi) * cb0

    def ua(p):
        return (p - p0) * np.sqrt(2 / ((ga + 1) * ra0) / (p + (ga - 1) / (ga + 1) * p0))

    def ub(p):
        if p < p1:
            return U1 + 2 * cb1 / (gb - 1) * (1 - (p / p1) ** ((gb - 1) / (2 * gb)))
        return U1 - (p - p1) * np.sqrt(2 / ((gb + 1) * rb1) / (p + (gb - 1) / (gb + 1) * p1))

    pf = brentq(lambda p: ua(p) - ub(p), p0 * (1 + 1e-14), p1 * 1e4, xtol=1e-300, rtol=1e-15, maxiter=500)
    zt = pf / p0 - 1
    Mt = np.sqrt(1 + (ga + 1) / (2 * ga) * zt)
    U = ua(pf)
    raf = ra0 * (ga + 1) * Mt**2 / ((ga - 1) * Mt**2 + 2)
    caf = np.sqrt(ga * pf / raf)
    Di, Dt = Mi * cb0, Mt * ca0
    bf = dict(ga=ga, gb=gb, R0=R0, Mi=Mi, p0=p0, ra0=ra0, rb0=rb0, ca0=ca0, cb0=cb0, zi=zi, p1=p1,
              rb1=rb1, cb1=cb1, U1=U1, pf=pf, zt=zt, Mt=Mt, U=U, raf=raf, caf=caf, Di=Di, Dt=Dt,
              bt=(Dt - U) / caf, psis=1 - U / Di, psit=1 - Dt / Di)
    if pf < p1:
        M1 = (pf / p1) ** ((gb - 1) / (2 * gb))
        cbf = M1 * cb1
        bf.update(kind="rarefaction", M1=M1, cbf=cbf, rbf=rb1 * M1 ** (2 / (gb - 1)),
                  w_head=U1 - cb1, w_tail=U - cbf)
    else:
        zr = pf / p1 - 1
        Mr = np.sqrt(1 + (gb + 1) / (2 * gb) * zr)
        rbf = rb1 * (gb + 1) * Mr**2 / ((gb - 1) * Mr**2 + 2)
        cbf = np.sqrt(gb * pf / rbf)
        Dr = U1 - Mr * cb1  # lab speed of reflected shock along the incident direction
        bf.update(kind="shock", zr=zr, Mr=Mr, rbf=rbf, cbf=cbf, Dr=Dr, br=(U - Dr) / cbf,
                  psir=1 - Dr / Di)
    bf["R"] = bf["raf"] / bf["rbf"]
    bf["N"] = caf / bf["cbf"]
    return bf


def _d1(f, x, h=1e-3):
    """Five-point central derivative."""
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def _consts(bf):
    """Shock-front constants: chi, initial tangential velocities, eta coefficients, rarefaction w0 data."""
    Mt, bt, Di = bf["Mt"], bf["bt"], bf["Di"]
    k = dict(chit=-np.arctanh(bt), vya=-(bf["U"] / Di) * bf["psit"],
             A_t=((Mt**2 + 1) / (2 * Mt**2 * bt), 1 / (2 * bt)))
    if bf["kind"] == "shock":
        Mr, br = bf["Mr"], bf["br"]
        k.update(chir=np.arctanh(br), vyb=((bf["U1"] - bf["U"]) / Di) * bf["psir"],
                 A_r=(-(Mr**2 + 1) / (2 * Mr**2 * br), -1 / (2 * br)))
        return k
    gb, M1, cb1 = bf["gb"], bf["M1"], bf["cb1"]
    psirh, psirt = 1 - bf["w_head"] / Di, 1 - bf["w_tail"] / Di
    zrh, zrt = (gb + 1 - 2 * M1) / (gb - 1), M1
    d = (psirt - psirh) / (zrh - zrt)
    a = zrh**2 / (gb + 1) * d - 2 * zrh / (gb + 1) * (d * zrt + psirt)
    b = 2 / (gb + 1) * (d * zrt + psirt)
    c = -d / (gb + 1)
    beta = (gb - 3) / (gb - 1)
    s, z0 = (gb + 1) / (gb - 1), -2 * M1 / (gb - 1)
    k.update(psirh=psirh, psirt=psirt, abc=(a, b, c), vyb=cb1 / Di * (a + b * zrt + c * zrt**2),
             xi=M1**beta, ckh=(gb + 1) / (3 - gb) * M1 ** (-beta), C=cb1 / Di * np.sqrt(M1**beta),
             al=(a + b * z0 + c * z0**2, b * s + 2 * c * z0 * s, c * s**2),
             ep=(-0.5, (gb + 1) / (2 * gb - 6), (3 * gb - 1) / (2 * gb - 6)))
    return k


def _a_side(bf):
    """Transmitted-shock functions eta_t^{+-}, eps_a1, eps_a2 [CW17 (43)-(48), CW16 (A1)-(A4)]."""
    k = _consts(bf)
    chit, vya, (a10, a11) = k["chit"], k["vya"], k["A_t"]

    def eta(x, s):
        return (a10 * sh(x) + a11 / sh(x)) / ch(x) + s

    def eps1(q):
        x = q - chit
        return sh(chit) * vya / (sh(x) * eta(x, 1))

    def eps2(q):
        x = q - chit
        return -eta(x, -1) / eta(x, 1)

    return chit, vya, eps1, eps2


def _bulk(M, w, dU, v0, F2):
    """Bulk vorticity parameter F = [1 + M^2/(M^2-1) 4w/dU]^-1 (v0 - 2F2) [CW17 (116), CW16 (32)]."""
    return (v0 - 2 * F2) / (1 + M**2 / (M**2 - 1) * 4 * w / dU)


def _rarefaction(bf, nit):
    """Reflected-rarefaction case [CW17]: F_b1 (71), functional eq. (79)-(83), (115)-(117), (A1)-(A5)."""
    Di, R, N = bf["Di"], bf["R"], bf["N"]
    chit, vya, eps1, eps2 = _a_side(bf)
    k = _consts(bf)
    psirh, psirt, (a, b, c), vyb = k["psirh"], k["psirt"], k["abc"], k["vyb"]
    xi, ckh, C, al, ep = k["xi"], k["ckh"], k["C"], k["al"], k["ep"]

    def dw0(z):
        return sum(al[j] * ep[j] * z ** (ep[j] - 1) for j in range(3))

    def Fb1(qb):
        I = quad(lambda z: dw0(z) / np.sqrt(np.exp(2 * qb) + ckh * (xi - z)), 1, xi,
                 epsabs=1e-14, epsrel=1e-12, limit=200)[0]
        return -vyb / 2 * np.exp(qb) + C * np.exp(qb) * sh(qb) * I

    def phi(q):
        qb = np.arcsinh(N * sh(q))
        D = R * ch(qb) / ch(q)
        e = eps1(q)
        return 2 / (D + 1) * Fb1(qb) - (D - 1) / (D + 1) * e, (D - 1) / (D + 1) * (-eps2(q)), e

    def F(q, n):
        p1, p2, _ = phi(q)
        return p1 / (1 - p2) if n == 0 else p1 + p2 * F(q - 2 * chit, n - 1)

    out = dict(vya=vya, vyb=vyb, chit=chit, abc=(a, b, c), psirh=psirh, psirt=psirt, xi_rt=xi,
               alpha_w0=al, Fb1_at=Fb1(np.arcsinh(N * sh(-2 * chit))), phi_at=phi(-2 * chit)[:2])
    irr = (vyb - R * vya) / (R + 1)
    rows = []
    for n in range(nit + 1):
        Fa1 = F(-2 * chit, n)
        Fa = _bulk(bf["Mt"], bf["Dt"] - bf["U"], bf["U"], vya, Fa1)

        def G(q, n=n):
            p1, p2, e1 = phi(q)
            return p1 - e1 + (p2 - eps2(q)) * F(q - 2 * chit, n)

        psi_inf = bf["psis"] - (Di / bf["caf"]) * _d1(G, 0.0)
        rows.append(dict(n=n, Fa1=Fa1, Fa=Fa, Fb=0.0, ui=irr + R * Fa / (R + 1),
                         via=-irr + Fa / (R + 1), psi_inf=psi_inf))
    return out, rows, irr


def _shock(bf, nit):
    """Reflected-shock case [CW16]: coupled functional system (A5)-(A7), (31)-(33), psi_inf (C1)-(C4)."""
    R, N, Di = bf["R"], bf["N"], bf["Di"]
    chit, vya, eps1, eps2 = _a_side(bf)
    k = _consts(bf)
    Mr, chir, vyb, (b10, b11) = bf["Mr"], k["chir"], k["vyb"], k["A_r"]

    def eta_r(y, s):
        return (b10 * sh(y) + b11 / sh(y)) / ch(y) + s

    def qb_of(qa):
        return np.arcsinh(N * sh(qa))

    def qa_of(qb):
        return np.arcsinh(sh(qb) / N)

    def phis(qa):
        qb = qb_of(qa)
        D = R * ch(qb) / ch(qa)
        y = qb + chir
        pa1 = (D + 1) / 2 * eps1(qa)
        pa2 = (D + 1) / 2 * eps2(qa)
        pb1 = (D + 1) / (2 * D) * sh(chir) * vyb / (sh(y) * eta_r(y, -1))
        pb2 = -(D + 1) / (2 * D) * eta_r(y, 1) / eta_r(y, -1)
        return pa1, pa2, (1 - D) / 2, pb1, pb2, (D - 1) / (2 * D)

    def seed(qa):
        pa1, pa2, pa3, pb1, pb2, pb3 = phis(qa)
        return (pa1 - pb1 * (pa3 - pa2)) / (1 - (pb3 - pb2) * (pa3 - pa2))

    def solve(qa, n):
        """(F_a1(qa), F_b2(qb(qa))) at iteration n; n = 0 is the A7 seed."""
        pa1, pa2, pa3, pb1, pb2, pb3 = phis(qa)
        if n == 0:
            Fb2 = seed(qa)
            return pb1 - pb3 * Fb2 + pb2 * seed(qa_of(qb_of(qa) + 2 * chir)), Fb2
        r1 = pa1 + pa2 * solve(qa - 2 * chit, n - 1)[0]
        r2 = pb1 + pb2 * solve(qa_of(qb_of(qa) + 2 * chir), n - 1)[1]
        det = pa3 * pb3 - 1
        return (r1 * pb3 - r2) / det, (pa3 * r2 - r1) / det

    out = dict(vya=vya, vyb=vyb, chit=chit, chir=chir)
    irr = (vyb - R * vya) / (R + 1)
    rows = []
    for n in range(nit + 1):
        Fa1 = solve(-2 * chit, n)[0]
        Fb2 = solve(qa_of(2 * chir), n)[1]
        Fa = _bulk(bf["Mt"], bf["Dt"] - bf["U"], bf["U"], vya, Fa1)
        Fb = _bulk(Mr, bf["U"] - bf["Dr"], bf["U1"] - bf["U"], vyb, Fb2)

        def G(q, n=n):
            return solve(q, n + 1)[0] - (eps1(q) + eps2(q) * solve(q - 2 * chit, n)[0])

        psi_inf = bf["psis"] - (Di / bf["caf"]) * _d1(G, 0.0)
        rows.append(dict(n=n, Fa1=Fa1, Fb2=Fb2, Fa=Fa, Fb=Fb, ui=irr + (R * Fa - Fb) / (R + 1),
                         via=-irr + (Fa + Fb) / (R + 1), vib=irr + R * (Fa + Fb) / (R + 1),
                         psi_inf=psi_inf))
    return out, rows, irr


def rm_linear(ga, gb, R0, Mi, nit=5):
    """Full linear solution: base flow, initial data, iterates n=0..nit, asymptotics, approximations."""
    bf = base_flow(ga, gb, R0, Mi)
    out, rows, irr = (_rarefaction if bf["kind"] == "rarefaction" else _shock)(bf, nit)
    A_post = (bf["R"] - 1) / (bf["R"] + 1)
    UD = bf["U"] / bf["Di"]
    res = dict(bf, **out)
    res.update(iterates=rows, ui=rows[-1]["ui"], psi_inf=rows[-1]["psi_inf"], u_irrot=irr,
               A_pre=(R0 - 1) / (R0 + 1), A_post=A_post,
               u_richtmyer=UD * A_post * bf["psis"],                 # k dU A+ psi0+
               u_richtmyer_pre=UD * (R0 - 1) / (R0 + 1),             # CW17 Eq.(170)
               u_meyer_blewett=UD * A_post * (1 + bf["psis"]) / 2)   # CW17 Eq.(171)
    return res


def rm_setup(gamma_inc, gamma_trans, rho_inc, rho_trans, p0, mach, k=1.0, psi0=1.0, nit=5):
    """Dimensional result for a shock (Mach in the incident gas) crossing into the transmitted gas."""
    r = rm_linear(gamma_trans, gamma_inc, rho_trans / rho_inc, mach, nit)
    cb0 = np.sqrt(gamma_inc * p0 / rho_inc)  # velocity scale: dimensionless speeds are in c_b0
    pscale = rho_inc * cb0**2                 # dimensionless pressures are in rho_b0 c_b0^2
    Di = mach * cb0
    d = dict(kind=r["kind"], R0=r["R0"], Mi=mach, c_inc0=cb0, c_trans0=r["ca0"] * cb0, D_i=Di,
             U_interface=r["U"] * cb0, U1=r["U1"] * cb0, p1=r["p1"] * pscale, p_f=r["pf"] * pscale,
             rho_inc1=r["rb1"] * rho_inc, rho_inc_f=r["rbf"] * rho_inc, rho_trans_f=r["raf"] * rho_inc,
             M_t=r["Mt"], D_t=r["Dt"] * cb0, z_t=r["zt"], R_post=r["R"], A_pre=r["A_pre"], A_post=r["A_post"],
             psi0_post=r["psis"] * psi0, ui=r["ui"], dv_inf=r["ui"] * k * psi0 * Di,
             dv_inf_per_kpsi0=r["ui"] * Di, psi_inf=r["psi_inf"] * psi0,
             dv_richtmyer=r["u_richtmyer"] * k * psi0 * Di, dv_richtmyer_pre=r["u_richtmyer_pre"] * k * psi0 * Di,
             dv_meyer_blewett=r["u_meyer_blewett"] * k * psi0 * Di, iterates=r["iterates"], raw=r,
             k=k, psi0=psi0, c_trans_f=r["caf"] * cb0)
    if r["kind"] == "shock":
        d.update(M_r=r["Mr"], z_r=r["zr"], D_r=r["Dr"] * cb0)
    else:
        d.update(M1=r["M1"], c_inc_f=r["cbf"] * cb0, head_speed=r["w_head"] * cb0,
                 tail_speed=r["w_tail"] * cb0)
    return d


# ---------------------------------------------------------------- time history psi_i(t)
# Bessel route: F's as power series in w = exp(-q) of the faster fluid ("home"), solved exactly
# order by order; H = F_a1 - F_a2 = sum omega_n w^n gives psi_i via CW17 (119)/(129).
# Laplace route: H(s) from the functional equations at complex s, Bromwich line Re s = c > 0.


def _smul(a, b):
    """Truncated power-series product."""
    return np.convolve(a, b)[:len(a)]


def _sinv(a):
    """Power-series reciprocal (a[0] != 0)."""
    K, r = len(a), np.zeros(len(a))
    r[0] = 1 / a[0]
    for k in range(1, K):
        r[k] = -np.dot(a[1:k + 1], r[k - 1::-1]) / a[0]
    return r


def _ssqrt(a):
    """Power-series square root with sqrt(a[0]) > 0."""
    K, r = len(a), np.zeros(len(a))
    r[0] = np.sqrt(a[0])
    for k in range(1, K):
        r[k] = (a[k] - np.dot(r[1:k], r[k - 1:0:-1])) / (2 * r[0])
    return r


def _spowers(g, K):
    """Rows k = [w^j] g(w)^k for k < K (g(0) = 0): the composition matrix, transposed."""
    P = np.zeros((K, len(g)))
    P[0, 0] = 1.0
    for k in range(1, K):
        P[k] = _smul(P[k - 1], g)
    return P


def _scomp(f, g):
    """f(g(w)) for g(0) = 0."""
    return f @ _spowers(g, len(f))


def _poly(c, K):
    """Polynomial coefficients padded to length K."""
    out = np.zeros(K)
    out[:len(c)] = c
    return out


def _wmap(x, N):
    """w' = exp(-q') with sinh q' = N sinh q, for w = exp(-q) given as a series x (x(0) = 0)."""
    K = len(x)
    w2 = _poly([1], K) - _smul(x, x)  # 1 - w^2
    P = N**2 * _smul(w2, w2) + 4 * _smul(x, x)
    return 2 * _smul(x, _sinv(N * w2 + _ssqrt(P)))


def _rh(A, u2, sgn):
    """A(u) + sgn (1 - u^4) for A(u) = x10 (1-u^2)^2 + 4 x11 u^2 (CW17 (43)-(48) times sinh x cosh x)."""
    x10, x11 = A
    K = len(u2)
    one_m = _poly([1], K) - u2
    return x10 * _smul(one_m, one_m) + 4 * x11 * u2 + sgn * (_poly([1], K) - _smul(u2, u2))


def _eps_series(A, u, v0sh, sgn):
    """Shock RH pair (eps1, eps2) as series in u = exp(-(q -+ chi)); sgn = +1 transmitted, -1 reflected."""
    u2 = _smul(u, u)
    den = _sinv(_rh(A, u2, sgn))
    e1 = v0sh * 2 * _smul(_smul(u, _poly([1], len(u)) + u2), den)
    e2 = -_smul(_rh(A, u2, -sgn), den)
    return e1, e2



def _toep(a):
    """Lower-triangular Toeplitz matrix of multiplication by the series a."""
    K = len(a)
    i, j = np.indices((K, K))
    return np.where(i >= j, a[np.clip(i - j, 0, K - 1)], 0.0)


def _shift_op(scale, g, K):
    """Matrix of f(w) -> f(scale w) (g None) or f(g(w))."""
    return np.diag(scale ** np.arange(K)) if g is None else _spowers(g, K).T


def _fb1_series(k, K):
    """F_b1 [CW17 (71)] as odd series in w_b: binomial expansion of the integrand, moments by Gauss-Legendre."""
    xi, ckh, (al, ep) = k["xi"], k["ckh"], (k["al"], k["ep"])
    x, wq = np.polynomial.legendre.leggauss(400)
    z = 1 + (xi - 1) * (x + 1) / 2
    dw0 = sum(al[j] * ep[j] * z ** (ep[j] - 1) for j in range(3)) * wq * (xi - 1) / 2
    M = K // 2 + 1
    binom = np.cumprod(np.r_[1.0, [(-0.5 - i) / (i + 1) for i in range(M - 1)]])
    bm = binom * ckh ** np.arange(M) * np.array([np.sum(dw0 * (xi - z) ** m) for m in range(M)])
    f = np.zeros(K)
    f[1:2 * M - 1:2] = (k["C"] / 2 * (bm[1:] - bm[:-1]))[:len(f[1:2 * M - 1:2])]
    return f, (k["C"] * bm[0] - k["vyb"]) / 2


def _series(bf, K=160):
    """Power series of F_a1, F_a2, F_b1, F_b2, H in w of the faster fluid (Bessel coefficients, CW17 (39), (65))."""
    N, R = bf["N"], bf["R"]
    k = _consts(bf)
    home = "a" if N >= 1 else "b"
    I1 = _poly([0, 1], K)
    X, Y = (I1, _wmap(I1, N)) if home == "a" else (_wmap(I1, 1 / N), I1)
    chit = k["chit"]
    lam = np.exp(2 * chit)
    St = _shift_op(lam, None, K) if home == "a" else _shift_op(0, _wmap(lam * X, N), K)
    XoY = _smul(np.r_[X[1:], 0], _sinv(np.r_[Y[1:], 0]))
    D = R * _smul(_smul(_poly([1], K) + _smul(Y, Y), XoY), _sinv(_poly([1], K) + _smul(X, X)))
    Dp, Dm = D + _poly([1], K), D - _poly([1], K)
    e1t, e2t = _eps_series(k["A_t"], np.exp(chit) * X, sh(chit) * k["vya"], 1)
    out = dict(home=home, kappa=bf["Di"] / (bf["caf"] if home == "a" else bf["cbf"]), X=X, Y=Y, D=D)
    if bf["kind"] == "rarefaction":
        fb, resid = _fb1_series(k, K)
        Fb1 = _scomp(fb, Y)
        iDp = _sinv(Dp)
        phi1 = 2 * _smul(Fb1, iDp) - _smul(_smul(Dm, iDp), e1t)
        phi2 = -_smul(_smul(Dm, iDp), e2t)
        Fa1 = np.linalg.solve(np.eye(K) - _toep(phi2) @ St, phi1)
        Fa2 = e1t + _smul(e2t, St @ Fa1)
        H = Fa1 - Fa2
        out.update(Fb1_resid=resid, Fb1=Fb1, Fb2=Fb1 - H)
    else:
        chir = k["chir"]
        mu = np.exp(-2 * chir)
        Sr = _shift_op(mu, None, K) if home == "b" else _shift_op(0, _wmap(mu * Y, 1 / N), K)
        e1r, e2r = _eps_series(k["A_r"], np.exp(-chir) * Y, sh(chir) * k["vyb"], -1)
        h, iD = 0.5 * Dp, 0.5 * _smul(Dp, _sinv(D))
        A = np.block([[_toep(-0.5 * Dm) - _toep(_smul(h, e2t)) @ St, np.eye(K)],
                      [np.eye(K), _toep(_smul(0.5 * Dm, _sinv(D))) - _toep(_smul(iD, e2r)) @ Sr]])
        sol = np.linalg.solve(A, np.r_[_smul(h, e1t), _smul(iD, e1r)])
        Fa1, Fb2 = sol[:K], sol[K:]
        Fa2 = e1t + _smul(e2t, St @ Fa1)
        Fb1 = e1r + _smul(e2r, Sr @ Fb2)
        H = Fa1 - Fa2
        out.update(Fb1=Fb1, Fb2=Fb2, Hb=Fb1 - Fb2)
    out.update(Fa1=Fa1, Fa2=Fa2, H=H)
    return out


def _inv_s(f):
    """Series in z = 1/s of f(w(z)), w = exp(-q) = (sqrt(1+z^2) - 1)/z, s = sinh q [CW17 (85), (123)]."""
    K = len(f)
    wz = (_ssqrt(_poly([1, 0, 1], K + 1)) - _poly([1], K + 1))[1:]
    return _scomp(f, wz)


def _history_coeffs(bf, tau_max):
    """Bessel coefficients omega_n (home fluid) with enough orders for tau_a <= tau_max."""
    th = tau_max * (bf["cbf"] / bf["caf"] if bf["N"] < 1 else 1.0)
    K = max(160, int(th + 12 * th ** (1 / 3) + 60)) | 1
    if K > 1201:
        raise ValueError("tau_max = %g needs %d Bessel orders; refusing (use the asymptote)" % (tau_max, K))
    S = _series(bf, K + 1)
    return dict(S, tau_scale=th / tau_max if tau_max > 0 else 1.0, K=K)


def _bessel_psi(co, psis, tau_a):
    """psi_i/psi0 and u = (d psi/dt)/(k psi0 D_i) from the Bessel series [CW17 (119), (129)], tau_a = k c_af t."""
    from scipy.special import jv
    tau = np.atleast_1d(np.asarray(tau_a, float)) * co["tau_scale"]
    K, om = co["K"], co["H"]
    J = jv(np.arange(K + 1)[:, None], tau[None, :])
    n = np.arange(1, K + 1, 2)
    cJodd = np.cumsum(J[1::2], axis=0) - J[1::2]               # sum_{j odd < n} J_j
    cJeven = np.vstack([np.zeros_like(tau), np.cumsum(J[2:K:2], axis=0)])  # sum_{k=1}^{(n-1)/2} J_2k
    # int_0^tau J0 = 2 sum_k J_2k+1 (Neumann series), not itj0y0: wrong by ~1e10 above x = 20 in scipy 1.16.3
    intJ0 = 2 * np.sum(J[1::2], axis=0)
    In = intJ0 - J[1::2] - 2 * cJodd                            # int_0^tau n J_n(x)/x dx
    intJn = 1 - J[0] - 2 * cJeven                               # int_0^tau J_n
    w = om[1:K + 1:2][:, None]
    u = -np.sum(w * In, axis=0)
    psi = psis - co["kappa"] * np.sum(w * (tau * In - n[:, None] * intJn), axis=0)
    return psi, u


def _fb1_complex(k, qb):
    """F_b1(q_b) [CW17 (71)] at complex q_b, rewritten to avoid the e^{2q} cancellation (Gauss-Legendre in z)."""
    xi, ckh, al, ep = k["xi"], k["ckh"], k["al"], k["ep"]
    x, wq = np.polynomial.legendre.leggauss(200)
    z = 1 + (xi - 1) * (x + 1) / 2
    dw0 = sum(al[j] * ep[j] * z ** (ep[j] - 1) for j in range(3)) * wq * (xi - 1) / 2
    e2 = np.exp(-2 * qb)[..., None]
    eps = ckh * (xi - z) * e2
    r = np.sqrt(1 + eps)
    integ = -eps / (r * (1 + r)) - e2 / r
    return k["C"] / 2 * np.exp(qb) * np.sum(dw0 * integ, axis=-1)


def _laplace_H(bf, s, qmax=36.0, tol=1e-14):
    """H(s_a) = F_a1 - F_a2 at complex s_a (Re s_a > 0) from the functional equations by iteration."""
    R, N = bf["R"], bf["N"]
    k = _consts(bf)
    chit, vya, eps1, eps2 = _a_side(bf)
    q = np.arcsinh(np.asarray(s, complex))
    if bf["kind"] == "rarefaction":
        def phi(q):
            qb = np.arcsinh(N * sh(q))
            D = R * ch(qb) / ch(q)
            return 2 / (D + 1) * _fb1_complex(k, qb) - (D - 1) / (D + 1) * eps1(q), -(D - 1) / (D + 1) * eps2(q)

        def Fa1(q):
            F, prod, ql = 0j, 1.0 + 0j, q
            while True:
                p1, p2 = phi(ql)
                F = F + prod * p1
                prod = prod * p2
                ql = ql - 2 * chit
                if np.max(np.abs(prod) * np.exp(-ql.real)) < tol or ql.real.min() > qmax:
                    return F

        Fs = Fa1(q - 2 * chit)
        return Fa1(q) - (eps1(q) + eps2(q) * Fs)
    chir, vyb, (b10, b11) = k["chir"], k["vyb"], k["A_r"]

    def eta_r(y, sg):
        return (b10 * sh(y) + b11 / sh(y)) / ch(y) + sg

    def node(qa, wgt):
        """(F_a1(qa), F_b2(qb(qa)), F_a1(qa - 2 chi_t)) with pruned children."""
        qb = np.arcsinh(N * sh(qa))
        D = R * ch(qb) / ch(qa)
        y = qb + chir
        pa1, pa2, pa3 = (D + 1) / 2 * eps1(qa), (D + 1) / 2 * eps2(qa), (1 - D) / 2
        pb1 = (D + 1) / (2 * D) * sh(chir) * vyb / (sh(y) * eta_r(y, -1))
        pb2, pb3 = -(D + 1) / (2 * D) * eta_r(y, 1) / eta_r(y, -1), (D - 1) / (2 * D)
        det = pa3 * pb3 - 1
        qt = qa - 2 * chit
        qr = np.arcsinh(sh(qb + 2 * chir) / N)
        wt = wgt * np.max(np.abs(pa2) * np.maximum(1, np.abs(pb3)) / np.abs(det))
        wr = wgt * np.max(np.abs(pb2) * np.maximum(1, np.abs(pa3)) / np.abs(det))
        A = node(qt, wt)[0] if wt * np.exp(-qt.real.min()) > tol and qt.real.min() < qmax else 0
        B = node(qr, wr)[1] if wr * np.exp(-qr.real.min()) > tol and qr.real.min() < qmax else 0
        r1, r2 = pa1 + pa2 * A, pb1 + pb2 * B
        return (r1 * pb3 - r2) / det, (pa3 * r2 - r1) / det, A

    F, _, A = node(q, 1.0)
    return F - (eps1(q) + eps2(q) * A)


def _gl_panels(edges, m=16):
    """Composite Gauss-Legendre nodes and weights on consecutive panels."""
    x, w = np.polynomial.legendre.leggauss(m)
    a, b = edges[:-1, None], edges[1:, None]
    return ((a + b) / 2 + (b - a) / 2 * x).ravel(), ((b - a) / 2 * w).ravel()


def _laplace_psi(bf, tau_a, c=0.1, ymax=400.0):
    """psi_i/psi0 and u by numerical Bromwich inversion on Re s_a = c of Psi = psi0*/s - (D_i/c_af) H/s^2."""
    tau = np.atleast_1d(np.asarray(tau_a, float))
    big = np.array([2000.0, 4000.0])
    h1s = big * _laplace_H(bf, big + 0j).real
    h1 = (4 * h1s[1] - h1s[0]) / 3  # lim s H(s), Richardson in 1/s^2
    d = min(0.04, c / 3)
    edges = np.unique(np.r_[np.arange(0, 12 + d / 2, d), np.arange(12, ymax + 0.25, min(0.5, 2.5 / tau.max()))])
    y, wy = _gl_panels(edges)
    s = c + 1j * y
    G = _laplace_H(bf, s) - h1 * s / (s**2 + 1)
    E = np.exp(1j * np.outer(tau, y)) * wy
    fpsi = np.exp(c * tau) / np.pi * (E @ (G / s**2)).real
    fu = np.exp(c * tau) / np.pi * (E @ (G / s)).real
    kap = bf["Di"] / bf["caf"]
    psi = bf["psis"] - kap * (fpsi + h1 * (1 - np.cos(tau)))
    return psi, -(fu + h1 * np.sin(tau)), h1


def psi_history(setup, t, dimensional=False):
    """psi_i(t)/psi0 and (d psi_i/dt)/(k psi0 D_i) by the Bessel series; t in tau_a = k c_af t unless dimensional."""
    r = setup["raw"] if "raw" in setup else setup
    t = np.atleast_1d(np.asarray(t, float))
    tau = t * setup["k"] * setup["c_trans_f"] if dimensional else t
    key = "_hist_coeffs"
    co = setup.get(key)
    if co is None or co["tau_max"] < tau.max():
        co = _history_coeffs(r, max(25.0, 1.2 * tau.max()))
        co["tau_max"] = max(25.0, 1.2 * tau.max())
        setup[key] = co
    return _bessel_psi(co, r["psis"], tau)


def _digits(ours, paper):
    """Number of leading printed digits of `paper` reproduced by `ours` rounded alike."""
    s = paper.strip()
    dec = len(s.split(".")[1]) if "." in s else 0
    ok = round(ours, dec) == round(float(s), dec)
    sig = len(s.replace("-", "").replace(".", "").lstrip("0"))
    return "all %d" % sig if ok else "|d|=%.1e" % abs(ours - float(s))


def _validate():
    """Print paper value vs ours for CW17 App.B, CW17 Table I, CW16 App.D and Table II."""
    lines = []

    def row(label, ours, paper):
        lines.append("  %-34s %16s %18.11f   %s" % (label, paper, ours, _digits(ours, paper)))

    print("=== (A) reflected rarefaction: CW17 Appendix B, HF100/10 (Mi=15.3, R0=0.0706, ga=1.45, gb=1.8)")
    r = rm_linear(1.45, 1.8, 0.0706, 15.3)
    it = r["iterates"]
    for lab, v, p in [("z_t", r["zt"], "53.09409"), ("M1", r["M1"], "0.68305"), ("M_t", r["Mt"], "6.77166"),
                      ("beta_t", r["bt"], "0.41327"), ("R", r["R"], "0.26242"), ("N", r["N"], "1.75206"),
                      ("rho_b1/rho_b0", r["rb1"], "3.46302"), ("c_b1/c_b0", r["cb1"], "9.31816"),
                      ("U1/c_b0", r["U1"], "10.88189"), ("rho_af/rho_a0", r["raf"] / r["ra0"], "4.96338"),
                      ("U/c_a0", r["U"] / r["ca0"], "5.40733"), ("psi_t0", r["psit"], "-0.49503"),
                      ("psi_rh", r["psirh"], "0.89780"), ("psi_rt", r["psirt"], "0.22218"),
                      ("psi0* (128)", r["psis"], "-0.193815267"),
                      ("v0_ya", r["vya"], "0.59097"), ("v0_yb", r["vyb"], "-0.27024"),
                      ("a (B7)", r["abc"][0], "-0.45064"), ("b (B7)", r["abc"][1], "-0.13844"),
                      ("c (B7)", r["abc"][2], "0.21751"), ("xi_rt", r["xi_rt"], "1.771421"),
                      ("F_b1(B8)", r["Fb1_at"], "0.048110033"),
                      ("phi_a1(-2chi_t)", r["phi_at"][0], "0.0409"), ("phi_a2(-2chi_t)", r["phi_at"][1], "-0.0789")]:
        row(lab, v, p)
    for n, p in enumerate(["0.037929826", "0.039750764", "0.039719406", "0.039719922", "0.039719913", "0.039719913"]):
        row("F_a1^[%d](-2chi_t)" % n, it[n]["Fa1"], p)
    for n, p in enumerate(["0.253531913", "0.251739422", "0.251770290", "0.251769783", "0.251769790", "0.251769790"]):
        row("F_a^[%d]" % n, it[n]["Fa"], p)
    for n, p in enumerate(["-0.284208640", "-0.284581247", "-0.284574830", "-0.284574936", "-0.284574934", "-0.284574934"]):
        row("dv_inf^[%d] (k psi0 Di)" % n, it[n]["ui"], p)
    for n, p in enumerate(["0.537740554", "0.536320669", "0.536345120", "0.536344718", "0.536344725", "0.536344725"]):
        row("dv_ya_inf^[%d]" % n, it[n]["via"], p)
    row("dv_inf^ws (B16)", r["u_irrot"], "-0.3369")
    row("dv_inf^[5] um/ns (B14, Di=49.79)", it[5]["ui"] * 2 * np.pi / 100 * 10 * 49.79, "-8.903")
    row("dv_inf^[0] um/ns (B14, Di=49.79)", it[0]["ui"] * 2 * np.pi / 100 * 10 * 49.79, "-8.891")
    for n, p in enumerate(["0.370315726", "0.138085814", "0.184543403", "0.182725015", "0.182761012", "0.182760406"]):
        row("psi_inf^[%d] (psi0)" % n, it[n]["psi_inf"], p)
    print("\n".join(lines))
    lines.clear()

    print("\n=== (A) CW17 Table I (lengths um; dv in um/ns with Di = Mi*3.254 um/ns, which the Table does not state)")
    cases = [("HF", 1.45, 0.0706, 15.3), ("LF", 1.45, 0.0706, 10.8), ("1.33F", 1.45, 0.0706, 1.33),
             ("5.6F", 1.45, 0.0706, 5.6), ("HP", 1.8, 0.647, 15.3), ("LP", 1.8, 0.647, 10.8)]
    table = [("HF", 100, 10, "-1.938", "1.828", "-8.904", "-8.893"), ("HF", 100, 4, "-0.775", "0.731", "-3.562", "-3.557"),
             ("HF", 50, 14, "-2.713", "2.559", "-24.932", "-24.900"),
             ("LF", 100, 10, "-1.865", "1.884", "-6.391", "-6.386"), ("LF", 100, 4, "-0.746", "0.754", "-2.557", "-2.554"),
             ("LF", 50, 14, "-2.611", "2.638", "-17.896", "-17.880"),
             ("1.33F", 100, 4, "1.995", "3.190", "-0.347", "-0.347"), ("5.6F", 100, 4, "-0.597", "0.929", "-1.429", "-1.428"),
             ("HP", 100, 14, "2.979", "4.459", "-2.750", "-2.749"), ("LP", 100, 14, "3.026", "4.506", "-1.954", "-1.953")]
    cache = {c[0]: rm_linear(c[1], 1.8, c[2], c[3]) for c in cases}
    for name, lam, p0, s_star, s_inf, v5, v0 in table:
        r = cache[name]
        Di = r["Mi"] * 3.254
        kp = 2 * np.pi / lam * p0
        tag = "%s %d/%d" % (name, lam, p0)
        row(tag + " psi0*", r["psis"] * p0, s_star)
        row(tag + " psi_inf", r["psi_inf"] * p0, s_inf)
        row(tag + " dv^[5]", r["iterates"][5]["ui"] * kp * Di, v5)
        row(tag + " dv^[0]", r["iterates"][0]["ui"] * kp * Di, v0)
    print("\n".join(lines))
    lines.clear()

    print("\n=== (B) reflected shock: CW16 Appendix D, case 5 (air->SF6, Mi=2.86, R0=5.25, ga=1.0935, gb=1.4)")
    r = rm_linear(1.0935, 1.4, 5.25, 2.86)
    it = r["iterates"]
    for lab, v, p in [("z_r", r["zr"], "0.801833"), ("z_t", r["zt"], "15.894346"), ("M_r", r["Mr"], "1.298955"),
                      ("M_t", r["Mt"], "4.026763"), ("beta_r", r["br"], "0.786488"), ("beta_t", r["bt"], "0.315299"),
                      ("rho_b1/rho_b0", r["rb1"], "3.723755"), ("c_b1/c_b0", r["cb1"], "1.586803"),
                      ("rho_bf/rho_b1", r["rbf"] / r["rb1"], "1.513875"), ("rho_af/rho_a0", r["raf"] / r["ra0"], "9.654408"),
                      ("c_bf/c_b1", r["cbf"] / r["cb1"], "1.090969"), ("c_af/c_a0", r["caf"] / r["ca0"], "1.322842"),
                      ("(U1-U)/c_b1", (r["U1"] - r["U"]) / r["cb1"], "0.4409218"), ("U/c_a0", r["U"] / r["ca0"], "3.609672"),
                      ("R", r["R"], "8.991123"), ("N", r["N"], "0.294740"), ("psi_t0", r["psit"], "0.456930"),
                      ("psi_r0", r["psir"], "0.989240"), ("v0_ya", r["vya"], "-0.222442"), ("v0_yb", r["vyb"], "0.242003"),
                      ("chi_t", r["chit"], "-0.326419"), ("chi_r", r["chir"], "1.062157"),
                      ("F_a1^[0](-2chi_t)", it[0]["Fa1"], "-0.0516249"), ("F_b2^[0](2chi_r)", it[0]["Fb2"], "-0.000622197"),
                      ("F_a^[0]", it[0]["Fa"], "-0.0798572"), ("F_b^[0]", it[0]["Fb"], "0.0120960"),
                      ("F_a^[5]", it[5]["Fa"], "-0.0865496"), ("F_b^[5] (printed 0.012538)", it[5]["Fb"], "0.0122538"),
                      ("u_irrot", r["u_irrot"], "0.224400"), ("u_i^[0]", it[0]["ui"], "0.151325"),
                      ("u_i^[5]", it[5]["ui"], "0.145287"), ("v_ia^[0]", it[0]["via"], "-0.231182"),
                      ("v_ia^[5]", it[5]["via"], "-0.231836"), ("psi0* (cm)", r["psis"] * 1.04, "0.53"),
                      ("psi_inf (cm)", r["psi_inf"] * 1.04, "-0.4")]:
        row(lab, v, p)
    print("\n".join(lines))
    lines.clear()

    print("\n=== (B) CW16 Table II u_i^[5], u_i^[0] and Table I psi_inf (cm)")
    t2 = [("1[36]", 5 / 3, 5 / 3, 1.817, 1.30, 0.29, "0.0626", "0.0626", "0.187"),
          ("2[36]", 5 / 3, 5 / 3, 1.817, 1.90, 0.26, "0.0847", "0.0847", "0.098"),
          ("5[36]", 1.0935, 1.4, 5.25, 2.86, 1.04, "0.1463", "0.1521", "-0.389"),
          ("6[36]", 1.0935, 5 / 3, 39, 1.13, 1.50, "0.0470", "0.0470", "0.777"),
          ("7[36]", 1.0935, 5 / 3, 39, 1.41, 1.38, "0.0876", "0.0882", "-0.192"),
          ("8[36]", 1.0935, 5 / 3, 39, 1.95, 1.36, "0.0976", "0.1036", "-0.771"),
          ("1[37]", 1.0935, 1.4, 5.23, 1.45, 0.306, "0.1599", "0.1619", "0.045"),
          ("2[37]", 1.0935, 1.4, 5.23, 1.45, 0.918, "0.1599", "0.1619", "0.134")]
    for name, ga, gb, R0, Mi, p0, u5, u0, pinf in t2:
        r = rm_linear(ga, gb, R0, Mi)
        row(name + " u^[5]", r["iterates"][5]["ui"], u5)
        row(name + " u^[0]", r["iterates"][0]["ui"], u0)
        row(name + " psi_inf", r["psi_inf"] * p0, pinf)
    print("\n".join(lines))
    lines.clear()

    print("\n=== Weak-shock limit: dv_inf/(k psi0 U) -> A_pre = (R0-1)/(R0+1)  [CW17 (162)]")
    for ga, gb, R0 in [(1.45, 1.8, 0.0706), (5 / 3, 5 / 3, 1 / 3), (5 / 3, 5 / 3, 3.0), (1.0935, 1.4, 5.25)]:
        for Mi in (1.1, 1.01, 1.001):
            r = rm_linear(ga, gb, R0, Mi)
            UD = r["U"] / r["Di"]
            print("  %-11s ga=%.4f gb=%.4f R0=%.4f Mi=%-6g u/(U/Di)=%+.6f  Richtmyer A+psi0+/psi0=%+.6f  A_pre=%+.6f"
                  % (r["kind"], ga, gb, R0, Mi, r["ui"] / UD, r["u_richtmyer"] / UD, r["A_pre"]))


def _a14():
    """Print the two A14 targets (gamma = 5/3, Mach 1.8439, p0 = 1e5, rho 3 and 1)."""
    for title, (ri, rt) in [("(i) heavy -> light: shock in rho=3 into rho=1", (3.0, 1.0)),
                            ("(ii) light -> heavy: shock in rho=1 into rho=3", (1.0, 3.0))]:
        d = rm_setup(5 / 3, 5 / 3, ri, rt, 1e5, 1.8439)
        print("\n=== A14 %s  (reflected %s)" % (title, d["kind"]))
        print("  c_inc0 = %.6f  c_trans0 = %.6f  D_i = %.6f  U_interface = %.6f  U1 = %.6f"
              % (d["c_inc0"], d["c_trans0"], d["D_i"], d["U_interface"], d["U1"]))
        print("  p1 = %.6e  p_f = %.6e  rho_inc1 = %.6f  rho_inc_f = %.6f  rho_trans_f = %.6f"
              % (d["p1"], d["p_f"], d["rho_inc1"], d["rho_inc_f"], d["rho_trans_f"]))
        print("  transmitted: M_t = %.6f  D_t = %.6f  z_t = %.6f" % (d["M_t"], d["D_t"], d["z_t"]))
        if d["kind"] == "shock":
            print("  reflected shock: M_r = %.6f  z_r = %.6f  lab speed along D_i = %.6f" % (d["M_r"], d["z_r"], d["D_r"]))
        else:
            print("  reflected rarefaction: M1 = c_f/c_1 = %.6f  head speed = %.6f  tail speed = %.6f (along D_i)"
                  % (d["M1"], d["head_speed"], d["tail_speed"]))
        print("  A_pre = %+.6f  A_post = %+.6f  psi0+/psi0 = %+.6f  R = rho_trans_f/rho_inc_f = %.6f"
              % (d["A_pre"], d["A_post"], d["psi0_post"], d["R_post"]))
        print("  iterates:  n   u_i=dv/(k psi0 Di)   dv/(k psi0)          psi_inf/psi0")
        for t in d["iterates"]:
            print("            %d   %+.10f     %+.8f     %+.10f" % (t["n"], t["ui"], t["ui"] * d["D_i"], t["psi_inf"]))
        r = d["raw"]
        print("  u_inf/(k psi0 Di) = %+.8f   u_inf/(k psi0 U) = %+.8f   u_inf/(k psi0) = %+.6f"
              % (d["ui"], d["ui"] * d["D_i"] / d["U_interface"], d["dv_inf_per_kpsi0"]))
        print("  psi_inf/psi0 = %+.8f   (psi0+/psi0 = %+.8f)" % (d["psi_inf"], d["psi0_post"]))
        print("  compare dv/(k psi0): Richtmyer k U A+ psi0+ = %+.6f   Richtmyer pre (A_pre psi0) = %+.6f   "
              "Meyer-Blewett = %+.6f   irrotational (ws) = %+.6f   uncorrected u^[0] = %+.6f"
              % (d["dv_richtmyer"], d["dv_richtmyer_pre"], d["dv_meyer_blewett"], r["u_irrot"] * d["D_i"],
                 r["iterates"][0]["ui"] * d["D_i"]))



def _affine_PH(bf, K=40):
    """Constant and f_a1^(j) coefficients of P_ai, H_ai in powers of 1/s_a [CW17 (B18a), (B19a)]."""
    k = _consts(bf)
    X, chit = _poly([0, 1], K), k["chit"]
    e1, e2 = _eps_series(k["A_t"], np.exp(chit) * X, sh(chit) * k["vya"], 1)
    zw = 2 * _smul(X, _sinv(_poly([1, 0, -1], K)))
    ic = 2 * _smul(X, _sinv(_poly([1, 0, 1], K)))

    def PH(g):
        F = _scomp(g, zw)
        F2 = e1 + _smul(e2, np.exp(2 * chit) ** np.arange(K) * F)
        return np.r_[_inv_s(_smul(F + F2, ic)), _inv_s(F - F2)]

    base = PH(np.zeros(K))
    cols = [base] + [PH(np.eye(K)[j]) - base for j in (1, 3, 5, 7)]
    return np.array(cols).T[:K], np.array(cols).T[K:]


def _check_routes(bf, tau, lines, label):
    """Bessel vs Laplace psi_i/psi0 and u over tau_a; returns both."""
    import time
    co = _history_coeffs(bf, max(25.0, tau.max()))
    pb, ub = _bessel_psi(co, bf["psis"], tau)
    t0 = time.time()
    pl, ul, h1 = _laplace_psi(bf, tau)
    big = np.abs(pb) > 0.05 * np.abs(pb).max()
    lines.append("  %-26s home=%s K=%3d  max|dpsi|/max|psi| = %.1e  (pointwise rel. where |psi|>5%%max: %.1e)"
                 "  max|du|/max|u| = %.1e   [Laplace %.0f s]"
                 % (label, co["home"], co["K"], np.abs(pb - pl).max() / np.abs(pb).max(),
                    np.max(np.abs(pb - pl)[big] / np.abs(pb)[big]), np.abs(ub - ul).max() / np.abs(ub).max(),
                    time.time() - t0))
    return co, pb, ub, pl, ul


def _validate_history():
    """Time history psi_i(t): printed CW17 App. B / CW16 App. D numbers, Bessel vs Laplace, limits."""
    lines = []

    def row(label, ours, paper, note=""):
        lines.append("  %-34s %16s %18.11f   %s%s" % (label, paper, ours, _digits(ours, paper), note))

    print("\n=== psi_i(t) (A) CW17 App. B secs 6-8, HF100/10: series in 1/s_a and Bessel coefficients")
    bf = base_flow(1.45, 1.8, 0.0706, 15.3)
    S = _series(bf, 160)
    Fa1, Fa2, H = S["Fa1"], S["Fa2"], S["H"]
    pi, om = Fa1 + Fa2, Fa1 - Fa2
    n = np.arange(160)
    ic = 2 * _smul(_poly([0, 1], 160), _sinv(_poly([1, 0, 1], 160)))
    fa1, fa2, Hs = _inv_s(Fa1), _inv_s(Fa2), _inv_s(H)
    Pai = _inv_s(_smul(pi, ic))
    chit = _consts(bf)["chit"]
    pt = _inv_s(_smul(Fa1 * np.exp(n * chit) + Fa2 * np.exp(-n * chit), ic))
    for j, p in enumerate(["0.04359", "-0.00553", "0.00298", "-0.00290", "0.00293"]):
        row("f_a1^%d (B17)" % (2 * j + 1), fa1[2 * j + 1], p)
    PA, HA = _affine_PH(bf)
    b18 = [["-0.0773", "0.9561"], ["0.0584", "-0.5098", "0.9924"], ["-0.0488", "0.3992", "-0.4986", "0.9987"],
           ["0.0428", "-0.3415", "0.3781", "-0.4992", "0.9998"]]
    b19 = [["0.0773", "1.0437"], ["-0.0198", "0.0317", "1.0076"], ["0.0010", "-0.0247", "0.0023", "1.0013"],
           ["-0.0063", "0.0184", "-0.0048", "-0.0001", "1.0002"]]
    for j in range(4):
        for i, p in enumerate(b18[j]):
            row("P_ai s^-%d %s (B18a)" % (2 * j + 2, "const" if i == 0 else "f%d" % (2 * i - 1)), PA[2 * j + 2, i], p)
    notes = {(0, 1): "  <- misprint? 1 - lambda_t eps2(0) = 1.04386", (2, 0): "  <- TYPO 0.0100"}
    for j in range(4):
        for i, p in enumerate(b19[j]):
            row("H_ai s^-%d %s (B19a)" % (2 * j + 1, "const" if i == 0 else "f%d" % (2 * i - 1)), HA[2 * j + 1, i], p,
                notes.get((j, i), ""))
    for j, (pp, po) in enumerate([("-0.0712", "0.2456"), ("0.0321", "0.0537"), ("0.0497", "0.0498"),
                                  ("-0.0737", "-0.0726")]):
        row("pi_%d^a (B20)" % (2 * j + 1), pi[2 * j + 1], pp)
        row("omega_%d^a (B20)" % (2 * j + 1), om[2 * j + 1], po)
    for j, p in enumerate(["-0.094812947", "0.067499167", "-0.054579150", "0.046698056"]):
        row("p_t0^(%d) (B21)" % (2 * j + 1), pt[2 * j + 2], p)
    row("10 D_i/c_af um (B22)", 10 * bf["Di"] / bf["caf"], "13.7201801")
    row("k c_af 1/ns (B22, D_i=49.79um/ns)", 2 * np.pi / 100 * 49.79 / 15.3 * bf["caf"], "2.28014351")
    for j, p in enumerate(["-0.07919", "0.01845", "-0.00889", "0.00542", "-0.00373"]):
        row("f_a2^%d (B27)" % (2 * j + 1), fa2[2 * j + 1], p)
    for j, p in enumerate(["0.12278", "-0.07193", "0.05932", "-0.05827", "0.05993"]):
        row("(2n+1) h_i0^(%d) (B28 as printed)" % (2 * j + 1), (2 * j + 1) * Hs[2 * j + 1], p)
    for j, p in enumerate(["-0.03560", "0.03071", "-0.02572", "0.02144", "-0.01805"]):
        row("p_i0^(%d) (B29)" % (2 * j + 1), Pai[2 * j + 2], p)
    fact = [1.0, 6.0, 120.0, 5040.0, 362880.0]
    for j, p in enumerate(["-0.122783515", "0.003996278", "-0.000098873", "0.000000165", "-0.000000018"]):
        row("u_ai tau^%d (B30a)" % (2 * j + 1), -Hs[2 * j + 1] / fact[j], p, "  <- TYPO 0.00000165" if j == 3 else "")
    row("v_ai tau^0 (B30b)", _consts(bf)["vya"], "0.590970559")
    for j, p in enumerate(["-0.017797562", "0.001279777", "-0.000035717", "0.000000532", "-0.000000005"]):
        row("v_ai tau^%d (B30b)" % (2 * j + 2), Pai[2 * j + 2] / np.prod(np.arange(1, 2 * j + 3)), p)
    print("\n".join(lines))
    lines.clear()

    print("\n=== psi_i(t) (B) CW16 App. D, case 5 (air->SF6): first Bessel coefficients and initial derivatives")
    bf5 = base_flow(1.0935, 1.4, 5.25, 2.86)
    S5 = _series(bf5, 160)
    k5 = _consts(bf5)
    X1, Y1 = S5["X"][1], S5["Y"][1]
    pa, oa = (S5["Fa1"][1] + S5["Fa2"][1]) / X1, (S5["Fa1"][1] - S5["Fa2"][1]) / X1
    pb, ob = (S5["Fb1"][1] + S5["Fb2"][1]) / Y1, (S5["Fb1"][1] - S5["Fb2"][1]) / Y1
    row("pi_1^a (D7)", pa, "-0.1518432", "  <- TYPO -0.1584321 (p_t1 below needs it)")
    row("omega_1^a (D7)", oa, "-0.298504")
    row("pi_1^b (D7)", pb, "-0.123747")
    row("omega_1^b (D7)", ob, "-0.087981")
    row("p_t1 (D8)", 0.5 * (pa * ch(k5["chit"]) + oa * sh(k5["chit"])), "-0.033885")
    row("p_r1 (D8)", 0.5 * (pb * ch(k5["chir"]) + ob * sh(k5["chir"])), "-0.156202")
    print("\n".join(lines))
    lines.clear()

    print("\n=== psi_i(t): Bessel series vs numerical Bromwich inversion (Re s_a = 0.1), 0 <= tau_a = k c_af t <= 20")
    tau = np.linspace(0.0, 20.0, 401)
    cases = [("CW17 HF100/10 (rarefaction)", bf), ("CW16 case 5 (shock)", bf5),
             ("A14 (i) heavy->light", base_flow(5 / 3, 5 / 3, 1 / 3, 1.8439)),
             ("A14 (ii) light->heavy", base_flow(5 / 3, 5 / 3, 3.0, 1.8439))]
    res = {}
    for label, b in cases:
        res[label] = _check_routes(b, tau, lines, label)
        print(lines[-1])
    lines.clear()

    print("\n=== psi_i(t): t -> 0+ (psi0*, zero normal velocity, Taylor series CW17 (136)) and t -> infinity")
    for label, b in cases:
        co, pb_, ub_, pl, ul = res[label]
        Hs_ = _inv_s(co["H"])
        m = tau * co["tau_scale"] <= 1.5  # home-fluid time <= 1.5
        th = tau[m] * co["tau_scale"]
        tay = b["psis"] - co["kappa"] * sum(Hs_[2 * j + 1] * th ** (2 * j + 2) / np.prod(np.arange(1, 2 * j + 3))
                                            for j in range(20))
        r = rm_linear(b["ga"], b["gb"], b["R0"], b["Mi"], nit=12)
        lines.append("  %-26s psi(0+): Bessel %+.12f Laplace %+.12f psi0* %+.12f | u(0+): %.1e %.1e | "
                     "Taylor vs Bessel, k c_f t<=1.5: %.1e" % (label, pb_[0], pl[0], b["psis"], ub_[0], ul[0],
                                                              np.abs(tay - pb_[m]).max()))
        kap = b["Di"] / b["caf"]
        tl = np.array([20.0, 50.0, 100.0, 200.0])
        co2 = _history_coeffs(b, 205.0)
        p2, u2 = _bessel_psi(co2, b["psis"], tl)
        pl2, ul2, _ = _laplace_psi(b, tl[:2], c=0.03, ymax=100.0)
        lines.append("  %-26s psi_inf=%+.10f u_inf=%+.10f; psi-(psi_inf+u_inf D_i/c_af tau) at tau_a=20,50,100,200: "
                     "%s (Laplace 20,50: %s); u-u_inf: %s"
                     % ("", r["psi_inf"], r["ui"], " ".join("%+.1e" % x for x in p2 - r["psi_inf"] - kap * r["ui"] * tl),
                        " ".join("%+.1e" % x for x in pl2 - r["psi_inf"] - kap * r["ui"] * tl[:2]),
                        " ".join("%+.1e" % x for x in u2 - r["ui"])))
        print("\n".join(lines))
        lines.clear()


def _a14_history():
    """Print psi_i(t)/psi0 for the two A14 cases at a few times, Bessel and Laplace, with the asymptote."""
    import time
    for title, (ri, rt) in [("(i) heavy -> light", (3.0, 1.0)), ("(ii) light -> heavy", (1.0, 3.0))]:
        d = rm_setup(5 / 3, 5 / 3, ri, rt, 1e5, 1.8439)
        r = d["raw"]
        tau = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        t0 = time.time()
        psi, u = psi_history(d, tau)
        t1 = time.time()
        many = psi_history(d, np.linspace(0, 20, 1000))
        t2 = time.time()
        pl, ul, _ = _laplace_psi(r, tau)
        kap = r["Di"] / r["caf"]
        print("\n=== A14 %s psi_i(t)/psi0, tau_a = k c_af t, c_af = %.6f (k=1: t in s for lengths in m)"
              % (title, d["c_trans_f"]))
        print("  psi_history: first call (coefficients) %.2f s, then 1000 times %.3f s" % (t1 - t0, t2 - t1))
        print("  tau_a    psi/psi0 (Bessel)   psi/psi0 (Laplace)   psi_inf+u_inf*D_i/c_af*tau   u=dpsi/dt/(k psi0 D_i)")
        for i, ta in enumerate(tau):
            print("  %5.1f   %+.10f      %+.10f        %+.10f              %+.10f"
                  % (ta, psi[i], pl[i], r["psi_inf"] + kap * d["ui"] * ta, u[i]))
        print("  asymptote: psi_inf/psi0 = %+.10f, u_inf = %+.10f (k psi0 D_i), D_i/c_af = %.8f; len(many) = %d"
              % (r["psi_inf"], d["ui"], kap, len(many[0])))

if __name__ == "__main__":
    _validate()
    _a14()
    _validate_history()
    _a14_history()
