"""Network 5 (rates_net5.rhs, fixes="chemcool", eos="table") as SymPy expressions of y and the parameters.
Table lookups and the dust solve are opaque functions whose derivatives are slope functions (lookups.py).
"""
import sympy as sp

from luna.chemistry import tables as tb

S0 = sp.S.Zero
EV, KB, MH, PI, GRAV, ABHE = tb.EV, tb.KBOLTZ, tb.MH, tb.PI, tb.GRAVITY, tb.ABHE
TMIN = 1.0


def R(name):
    """A real symbol."""
    return sp.Symbol(name, real=True)


def F(x):
    """An exact double as a SymPy Float."""
    return sp.Float(float(x), 53)


class RealFn(sp.Function):
    """A real-valued function (so relations on it stay relations)."""

    def _eval_is_extended_real(self):
        return True


class jmax(RealFn):
    """jnp.maximum, with JAX's derivative (1/2 to each side on a tie)."""
    nargs = 2

    def fdiff(self, i):
        a, b = self.args
        return gtw(a, b) if i == 1 else gtw(b, a)


class jmin(RealFn):
    """jnp.minimum, with JAX's derivative."""
    nargs = 2

    def fdiff(self, i):
        a, b = self.args
        return gtw(b, a) if i == 1 else gtw(a, b)


class gtw(RealFn):
    """1 if a > b, 1/2 if a == b, else 0 (piecewise constant)."""
    nargs = 2

    def fdiff(self, i):
        return S0


class Opaque(RealFn):
    """Base of a table lookup output: group, key and the per-argument slope classes."""
    group = None
    key = None
    slopes = ()

    def fdiff(self, i):
        return self.slopes[i - 1](*self.args)


class Slope(RealFn):
    """Base of a lookup slope: constant within a table cell."""
    group = None
    key = None

    def fdiff(self, i):
        return S0


_CLASSES = {}


def opaque(group, key, nargs):
    """The (cached) opaque function class of one lookup output."""
    name = f"{group}_{key}"
    if name not in _CLASSES:
        sl = tuple(type(f"{name}_d{k}", (Slope,), {"nargs": nargs, "group": group, "key": (key, k)})
                   for k in range(nargs))
        _CLASSES[name] = type(name, (Opaque,), {"nargs": nargs, "group": group, "key": key,
                                                "slopes": sl})
    return _CLASSES[name]


class Where(RealFn):
    """jnp.where(c, a, b) as an opaque function (a Piecewise would be folded into its arguments' callers)."""
    nargs = 3

    @classmethod
    def eval(cls, c, a, b):
        if c == sp.true or a == b:
            return a
        if c == sp.false:
            return b

    def _eval_derivative(self, s):
        c, a, b = self.args
        return Where(c, a.diff(s), b.diff(s))


def where(c, a, b):
    """jnp.where(c, a, b)."""
    return Where(c, sp.sympify(a), sp.sympify(b))


def safe(d):
    """jnp.where(d == 0, 1, d)."""
    return where(sp.Eq(d, 0), 1, d)


def stim(a10, e10, t_rad):
    """cooling._stim at a constant radiation temperature."""
    import math
    x = e10 / (KB * t_rad)
    return a10 / (math.exp(min(x, 5.0)) - 1.0) if x < 5.0 else 0.0


class Model:
    """Symbols and expressions of one network-5 RHS; dust is 'implicit' or 'frozen'; lvg=False: divv = 0."""

    PARAMS = ("n", "g0_dust", "zeta", "dgr", "abundc", "abundo", "abundsi", "dl", "divv",
              "tdust_in", "h2_form_ex", "h2_form_kin")
    PHOTO = ("k_h2_dust", "k_co", "g_dust", "chi")

    def __init__(self, npix, dust, lvg=True):
        self.npix, self.dust, self.lvg = npix, dust, lvg
        self.y = [R(f"y{i}") for i in range(4)]
        self.p = {k: R(k) for k in self.PARAMS}
        if not lvg:
            self.p["divv"] = S0
        self.ph = {k: R(k) for k in self.PHOTO}
        self.cols = [R(f"nh2c{j}") for j in range(npix)]
        self.build()

    # ---- lookups ------------------------------------------------------------
    def chem(self, k, T):
        return opaque("chem_t", k, 1)(T)

    def cool(self, k, T):
        return opaque("cool_f", k, 1)(T)

    def dust_residual(self, td, T, x_hi):
        """f_nr of cooling.dust_temperature's Newton step: ISRF - emission + gas-grain + H2 heating."""
        p = self.p
        n, dgr = p["n"], p["dgr"]
        ch7 = self.chem(7, T) * dgr
        cl14 = opaque("cool_t", 14, 1)(T)
        isrf = F(5.8e-24) * n * (p["g0_dust"] * self.ph["chi"] / F(tb.f32(1.7))) * dgr
        rdust = self.rdust(td, T, ch7)
        dust_cool = F(4.68e-31) * td**6 * n * dgr
        gas_dust = cl14 * (T - td) * n**2 * dgr
        h2_heat = rdust * x_hi * n**2 * (1.0 - p["h2_form_ex"] - p["h2_form_kin"]) * F(4.48 * EV)
        return isrf - dust_cool + gas_dust + where(sp.Eq(rdust, 0), 0, h2_heat)

    @staticmethod
    def rdust(td, T, ch7):
        temp2 = T * F(1e-2)
        fa = 1 / (1 + F(1e4) * sp.exp(F(-6e2) / td))
        stick = 1 / (1 + F(0.4) * (temp2 + td * F(1e-2)) ** F(0.5) + F(0.2) * temp2
                     + F(0.08) * temp2 ** F(2.0))
        return ch7 * fa * stick

    def dust_temperature(self, T, x_hi):
        if self.dust == "frozen":
            return self.p["tdust_in"]
        model = self
        _td, _T, _x = R("_td"), R("_T"), R("_x")
        res = self.dust_residual(_td, _T, _x)
        dres = (res.diff(_T), res.diff(_x), res.diff(_td))

        class DUST(Opaque):
            nargs = 2
            group = "dust"
            key = "td"

            def fdiff(self, i):
                sub = {_td: self, _T: self.args[0], _x: self.args[1]}
                clipped = where(sp.Or(self >= 1500.0, self <= TMIN), 0, 1)
                return (-dres[i - 1] / dres[2]).xreplace(sub) * clipped
        model._dust_cls = DUST
        return DUST(T, x_hi)

    # ---- the RHS ------------------------------------------------------------
    def build(self):
        y, p, ph = self.y, self.p, self.ph
        n, dgr = p["n"], p["dgr"]
        x_h2 = jmin(F(0.5), jmax(S0, y[0]))
        x_hp = jmin(F(1.0), jmax(S0, y[1]))
        x_co = jmin(F(1.0), jmax(S0, y[2]))
        energy = jmax(y[3], S0)
        x_hi = jmax(1 - 2 * x_h2 - x_hp, S0)
        x_cp = jmax(S0, p["abundc"] - x_co)
        x_e = x_hp + x_cp + p["abundsi"]
        ekn = energy / ((1 + F(ABHE) - x_h2 + x_e) * n * F(KB))
        t_raw = opaque("tg", "T", 2)(ekn, x_h2)
        gam = opaque("tg", "G", 2)(ekn, x_h2)
        T = where(t_raw <= 0, F(TMIN), t_raw)
        self.t_raw = t_raw
        ch = lambda k: self.chem(k, T)  # noqa: E731

        ncrinv = 2 * x_h2 * (ch(6) - ch(5)) + ch(5)
        h2var0 = 1 / (1 + n * ncrinv)
        k_dh = ch(1) * ch(3) ** h2var0
        k_dh2 = ch(2) * ch(4) ** h2var0

        b5 = sp.sqrt(F(KB) * T / F(MH)) / F(1e5)
        shield = S0
        for c in self.cols:
            xx = (x_h2 * (n * p["dl"]) + c) / F(5e14)
            sq = sp.sqrt(1 + xx)
            fj = F(0.965) / ((1 + xx / b5) * (1 + xx / b5)) + F(0.035) * sp.exp(F(-8.5e-4) * sq) / sq
            shield = shield + fj * F(1.0 / self.npix)
        k_dis = ph["k_h2_dust"] * shield
        g_dust = ph["g_dust"]
        gamma_chx = F(2.94e-10) * g_dust

        phi = where(sp.Eq(x_e, 0), F(1e20), g_dust * sp.sqrt(T) / (n * safe(x_e)))
        phi_s = jmax(phi, F(1e-6))
        h_gr = where(phi < F(1e-6), F(1.225e-13) * dgr,
                     F(1.225e-13) * dgr / (1 + F(8.074e-6) * phi_s ** F(1.378)
                                           * (1 + ch(34) * phi_s ** ch(35))))
        h_gr = h_gr * sp.exp(-T / F(3e4))

        active = sp.And(dgr > F(1e-8), T < F(1e5))
        td = self.dust_temperature(T, x_hi)
        rdust_a = self.rdust(td, T, ch(7) * dgr)
        rdust = where(active, rdust_a, 0)
        rh2 = where(active, rdust_a * x_hi * n**2, 0)
        tdust = where(active, td, jmin(T, F(1.5e3)))

        z = p["zeta"]
        cr1, cr2, cr10, cr11 = z, 2.0 * z, F(3.7e-2) * z, F(2.2e-1) * z
        dx_hp = ((ch(11) * x_e * x_hi - ch(13) * x_hp * x_e - h_gr * x_hp) * n
                 + cr1 * x_hi + cr10 * x_h2)
        dx_h2 = ((-ch(25) * x_e * x_h2 - k_dh * x_hi * x_h2 - k_dh2 * x_h2 * x_h2
                  + rdust * x_hi) * n
                 - k_dis * x_h2 - F(0.5) * cr2 * x_h2 - cr10 * x_h2 - cr11 * x_h2)
        x_o = jmax(S0, p["abundo"] - x_co)
        den = F(5e-10) * x_o + gamma_chx / n
        beta = where(gamma_chx < F(1e-30), 1, F(5e-10) * x_o / safe(den))
        beta = where(sp.Eq(x_o, 0), 0, beta)
        dx_co = F(5e-16) * x_cp * x_h2 * beta * n - ph["k_co"] * x_co

        chem = self.chem_terms(T, n, x_h2, x_hi, x_hp, x_e, k_dh, k_dh2, k_dis, h_gr, rh2)
        rho = F((1.0 + 4.0 * ABHE) * MH) * n
        cs2 = gam * (gam - 1) * energy / rho
        l_j = sp.sqrt(F(PI) * cs2 / (F(GRAV) * rho))
        ab = dict(h2=x_h2, o=x_o, co=x_co, cp=x_cp, e=x_e, hp=x_hp, hi=x_hi)
        cool, tdust = self.cool_func(T, l_j, ab, tdust)
        de = -(sp.Add(*cool) + sp.Add(*chem))
        self.f = [dx_h2, dx_hp, dx_co, de]
        self.aux = {"T": T, "gamma": gam, "tdust": tdust}

    def chem_terms(self, T, n, x_h2, x_hi, x_hp, x_e, k_dh, k_dh2, k_dis, h_gr, rh2):
        ch = lambda k: self.chem(k, T)  # noqa: E731
        l1 = F(4.48 * EV) * (k_dh * x_hi * x_h2 + k_dh2 * x_h2 * x_h2 + ch(25) * x_e * x_h2) * n**2
        l2 = F(-4e-1 * EV) * x_h2 * n * k_dis
        cdex = ((F(tb.f32(1.4)) * sp.exp(F(-1.81e4) / (T + 1200)) * x_h2
                 + sp.exp(F(-1e3) / T) * x_hi) * F(1e-12) * sp.sqrt(T) * n)
        l3 = F(-2.0 * EV) * x_h2 * n * k_dis * F(8.5) * (cdex / (cdex + F(2e-7)))
        l4 = F(-4.48 * EV) * self.p["h2_form_kin"] * rh2
        l5 = F(13.6 * EV) * ch(11) * x_e * x_hi * n**2
        l6 = where(T < 100, 0, F(KB) * T * x_hp * (ch(13) * x_e + h_gr) * n**2)
        return [l1, l2, l3, l4, l5, l6]

    def cool_func(self, T, dl, ab, tdust):
        p = self.p
        n, dgr, divv = p["n"], p["dgr"], p["divv"]
        c = lambda k: self.cool(k, T)  # noqa: E731
        t_cmb = 2.726
        ynh2, ynh, yne, ynhp = ab["h2"] * n, ab["hi"] * n, ab["e"] * n, ab["hp"] * n
        r = []
        dense = n * dgr > F(1e10)
        tdust_hi = jmin(T, F(1.5e3))
        r.append(where(dense, F(4.68e-31) * tdust_hi**6 * n * dgr, c(14) * (T - tdust) * n**2 * dgr))
        tdust = where(dense, tdust_hi, tdust)

        h2var0 = ynh * c(4) + ynh2 * c(5) + ynhp * c(65) + yne * c(66)
        lte_lim = ynh2 * c(1) / (1 + c(1) / safe(h2var0))
        r2 = where(h2var0 < F(1e-4) * c(1), h2var0 * ynh2, lte_lim)
        vth = sp.sqrt(F(KB) * T / F(MH))
        n_jeans = ynh2 / (vth / dl)
        n_lvg = where(sp.Eq(divv, 0), sp.oo, ynh2 / safe(sp.Abs(divv)))
        opac = opaque("opac", "v", 2)(T, jmin(n_lvg, n_jeans))
        r2 = where(n < F(1e9), r2, r2 * opac)
        r.append(where(sp.Eq(ynh2, 0), 0, r2))
        r.append(c(52) * ynh * yne)
        r += self.co_terms(T, ab, ynh, ynh2)
        r.append(F(-3.2e-11) * (ynh2 + ynh) * p["zeta"])

        g = self.ph["g_dust"]
        pe_on = sp.And(sp.Ne(g, 0), dgr >= F(1e-8), yne >= F(1e-20) * g * c(16))
        pevar0 = where(pe_on, g * c(16) / safe(where(pe_on, yne, 1)), 1)
        eps = F(4.9e-2) / (1 + F(4e-3) * pevar0 ** F(tb.f32(0.73))) + c(17) / (1 + F(2e-4) * pevar0)
        r.append(where(pe_on, F(-1.3e-24) * eps * g * n * dgr, 0))
        r.append(self.oi(c, T, t_cmb, ynh, ynh2, yne, ynhp, ab["o"]))
        r.append(self.cii(c, T, t_cmb, ynh, ynh2, yne, ab["cp"], dl))

        si = p["abundsi"]
        c10 = c(49) * ynh + c(50) * yne
        b10 = F(stim(tb.SIII_A10, tb.SIII_E10, t_cmb))
        var0, var1 = c10 + F(tb.SIII_A10) + b10, c(51) * c10 + 2 * b10
        n0, n1 = var0 / (var0 + var1), var1 / (var0 + var1)
        si_on = sp.And(sp.Ne(c10, 0), si > F(1e-5) * si)
        r.append(where(si_on, ((F(tb.SIII_A10) + b10) * n1 - 2 * b10 * n0) * F(tb.SIII_E10) * si * n, 0))
        r.append(F(1.017e-37 * t_cmb**4) * (T - F(t_cmb)) * yne)
        tau = jmax((n / F(1.4e16)) ** F(2.8), F(1e-5))
        r.append(c(58) * n * (2 * ynh2) * jmin((1 - sp.exp(-tau)) / tau, F(1.0)))
        return r, tdust

    def co_terms(self, T, ab, ynh, ynh2):
        p = self.p
        n, divv = p["n"], p["divv"]
        co_on = ab["co"] > F(1e-5) * p["abundo"]
        dv = F(1e-5) * sp.Abs(where(sp.Eq(divv, 0), 1, divv))
        n_co = sp.log(where(co_on, ab["co"], 1) * n / dv) / sp.log(10)
        n13, n18 = n_co - 2, n_co - F(2.69897)
        lvg = sp.Ne(divv, 0)
        m = F(19.0)
        a2 = where(sp.Or(n18 > m, n13 > m, n_co > m), m, n_co)
        b2 = where(sp.Or(n18 > m, n13 > m), m, n13)
        c2 = where(n18 > m, m, n18)
        rot = [where(lvg, v, m) for v in (a2, b2, c2)]
        t_co = jmax(T, F(5.0))
        thin = rot[0] <= F(14.5)
        p_main = self.co_cool(t_co, rot[0])
        p13 = [where(thin, a, b) for a, b in zip(p_main, self.co_cool(t_co, rot[1]))]
        p18 = [where(thin, a, b) for a, b in zip(p_main, self.co_cool(t_co, rot[2]))]
        neff = ynh2 + F(2.0 ** 0.5) * ynh
        rot_on = sp.And(co_on, sp.Ne(neff, 0))
        low_t = where(T <= 5, sp.exp(F(-1e1) / T) / F(__import__("math").exp(-2.0)), 1)
        out = {}
        for k, frac, pp in ((7, 1.0, p_main), (21, 1e-2, p13), (22, 2e-3, p18)):
            l0, lte, alp, n05, _ = pp
            l0s, ltes = where(rot_on, l0, 1), where(rot_on, lte, 1)
            inv = 1 / l0s + neff / ltes + (1 / l0s) * (1 - n05 * l0s / ltes) * (neff / n05) ** alp
            out[k] = where(rot_on, F(frac) * ab["co"] * neff * n / inv * low_t, 0)
        neff = ynh2 + ynh
        vib_on = sp.And(co_on, sp.Ne(neff, 0), T >= 100)
        l0_vib = self.cool(12, T) * ab["h2"] + self.cool(13, T) * ab["hi"]
        for k, frac, pp in ((8, 1.0, p_main), (9, 1e-2, p13), (24, 2e-3, p18)):
            inv = 1 / where(vib_on, l0_vib, 1) + n / where(vib_on, pp[4], 1)
            out[k] = where(vib_on, F(frac) * ab["co"] * n * neff * (1 / inv), 0)
        return [out[k] for k in (7, 21, 22, 8, 9, 24)]

    def co_cool(self, t, ncol):
        """cooling.co_cool from the raw table values: (L0, LTE, alpha, n05, vib)."""
        g = lambda k: opaque("co", k, 2)(t, ncol)  # noqa: E731
        below = t < 5
        l0 = where(below, 0, F(10.0) ** (-opaque("co", "l0", 2)(t, ncol)))
        lte = where(below, 0, F(10.0) ** (-g("lte")))
        n05 = where(below, 1, F(10.0) ** g("n05"))
        alp = where(below, 1, g("alp"))
        vib = where(below, 0, F(10.0) ** (-g("vib")) * sp.exp(F(-3.08e3) / t))
        return l0, lte, alp, n05, vib

    def oi(self, c, T, t_cmb, ynh, ynh2, yne, ynhp, x_o):
        n = self.p["n"]
        c10 = c(18) * ynh + c(21) * ynh2 + c(24) * yne + c(27) * ynhp
        c20 = c(19) * ynh + c(22) * ynh2 + c(25) * yne + c(28) * ynhp
        c21 = c(20) * ynh + c(23) * ynh2 + c(26) * yne + c(29) * ynhp
        e10, e20, e21 = tb.OX_E10, tb.OX_E20, tb.OX_E21
        a10, a20, a21 = tb.OX_A10, tb.OX_A20, tb.OX_A21
        oa, ob, oc = (sp.exp(F(-e) / (F(KB) * T)) for e in (e10, e20, e21))
        c01, c02, c12 = F(0.6) * c10 * oa, F(0.2) * c20 * ob, F(1.0 / 3.0) * c21 * oc
        b10, b20, b21 = (stim(a, e, t_cmb) for a, e in ((a10, e10), (a20, e20), (a21, e21)))
        b01, b02, b12 = 0.6 * b10, 0.2 * b20, (1.0 / 3.0) * b21
        r01, r02, r12 = c01 + F(b01), c02 + F(b02), c12 + F(b12)
        r10, r20, r21 = c10 + F(a10 + b10), c20 + F(a20 + b20), c21 + F(a21 + b21)
        A1, A2, A3 = r01 + r02, -r10, -r20
        B1, B2, B3 = r01, -(r10 + r12), r21
        den = (A1 - A2) * (A1 * B3 - B1 * A3) - (A1 - A3) * (A1 * B2 - B1 * A2)
        zero = sp.And(sp.Eq(r01, 0), sp.Eq(r02, 0))
        den = where(zero, 1, den)
        n2 = -A1 * (A1 * B2 - B1 * A2) / den
        a12 = where(zero, 1, A1 - A2)
        n1 = (A1 / a12) - ((A1 - A3) / a12) * n2
        n0 = 1 - n1 - n2
        n0, n1, n2 = where(zero, 1, n0), where(zero, 0, n1), where(zero, 0, n2)
        tiny = r01 < F(1e-40)
        n0, n1, n2 = where(tiny, 1, n0), where(tiny, 0, n1), where(tiny, 0, n2)
        lam1 = F((a10 + b10) * e10) * n1 + F((a20 + b20) * e20 + (a21 + b21) * e21) * n2
        lam2 = F(b01 * e10 + b02 * e20) * n0 + F(b12 * e21) * n1
        on = sp.And(x_o > F(1e-5) * self.p["abundo"],
                    sp.Not(sp.And(sp.Eq(c10, 0), sp.Eq(c20, 0), sp.Eq(c21, 0))))
        return where(on, (lam1 - lam2) * x_o * n, 0)

    def cii(self, c, T, t_cmb, ynh, ynh2, yne, x_cp, dl):
        p = self.p
        n, divv = p["n"], p["divv"]
        c10 = c(45) * ynh + c(46) * ynh2 + c(47) * yne
        c01 = c(48) * c10
        a10, e10 = tb.CII_A10, tb.CII_E10
        b10 = stim(a10, e10, t_cmb)
        b01 = 2.0 * b10
        vth = sp.sqrt(F(KB) * T / F(6.0 * MH))
        col_jeans = (n * x_cp) / (vth / dl)
        col_lvg = where(sp.Eq(divv, 0), sp.oo, (n * x_cp) / sp.Abs(safe(divv)))
        tau = F(7.2e-13) * jmin(col_jeans, col_lvg)
        a10_eff = F(a10) * (1 / (1 + 3 * tau))
        var0, var1 = c10 + a10_eff + F(b10), c01 + F(b01)
        n0, n1 = var0 / (var0 + var1), var1 / (var0 + var1)
        on = sp.And(sp.Ne(c10, 0), x_cp > F(1e-5) * p["abundc"])
        return where(on, ((a10_eff + F(b10)) * F(e10) * n1 - F(b01 * e10) * n0) * x_cp * n, 0)
