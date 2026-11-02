"""Compile the external aprox13 channel assembly and RHS into numerical fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from luna.reference_data import reference_path

from nuclear_rate_reference import (
    COMMIT,
    FUNCTIONS,
    ALPHA_TARGETS,
    PROTON_PARENTS,
    INTERMEDIATES,
    routine,
)


def block(source, declaration):
    start = source.index(declaration)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def function(source, name):
    match = re.search(r"(?m)^[^\n;{}]*\b" + name + r"\s*\([^;{}]*\)\s*\{", source)
    if match is None:
        raise ValueError("missing function definition: " + name)
    start = source.rfind("template", 0, match.start())
    return block(source, source[start : match.end() - 1])


def build(source):
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if revision != COMMIT:
        raise ValueError("unexpected Microphysics revision")
    paths = [
        "constants/fundamental_constants.H",
        "interfaces/tfactors.H",
        "interfaces/rhs_type.H",
        "networks/aprox13/actual_network.H",
        "networks/aprox13/aprox13.net",
        "networks/rhs.H",
        "rates/aprox_rates.H",
        "screening/screen.H",
        "screening/screen_data.H",
    ]
    original = {name: (source / name).read_text() for name in paths}
    headers = {
        name: re.sub(r"^#include.*$", "", value, flags=re.M) for name, value in original.items()
    }
    rows = [
        line.split()
        for line in original[paths[4]].splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    species = [row for row in rows if not row[0].startswith("__extra")] + [
        row for row in rows if row[0].startswith("__extra")
    ]
    names = ",".join(row[1] + "=" + str(i + 1) for i, row in enumerate(species))
    masses = ",".join(row[2] for row in species)
    charges = ",".join(row[3] for row in species)
    prefix = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>
#define AMREX_GPU_HOST_DEVICE
#define AMREX_INLINE
#define AMREX_FORCE_INLINE
#define AMREX_FLATTEN
#define AMREX_GPU_MANAGED
#define SCREENING
#define SCREEN_METHOD 1
#define SCREEN_METHOD_screen5 1
#define SCREEN_METHOD_chugunov2007 2
#define SCREEN_METHOD_chugunov2009 3
#define SCREEN_METHOD_chabrier1998 4
namespace amrex {
using Real=double; using std::min; using std::max;
template<class T, std::size_t N> using Array=std::array<T,N>;
template<class T,int A,int B> struct Array1D {
 std::array<T,B-A+1> data{};
 constexpr T& operator()(int n) {return data[n-A];}
 constexpr const T& operator()(int n) const {return data[n-A];}
};
template<int A,int B,class F> constexpr void constexpr_for(F&& f) {
 if constexpr(A<B) {f(std::integral_constant<int,A>{}); constexpr_for<A+1,B>(f);}
}
namespace literals {
constexpr double operator""_rt(long double x) {return double(x);}
constexpr double operator""_rt(unsigned long long x) {return double(x);}
}
}
using namespace amrex::literals;
namespace admath {using std::exp; using std::sqrt; using std::cbrt; using std::log; using std::min; using std::max; using std::pow;}
namespace gcem {using std::pow; using std::log;}
namespace autodiff {
template<class T> T eval(T x) {return x;}
template<class T> double derivative(T) {return 0;}
namespace detail {template<class T> constexpr bool isDual=false;}
}
namespace network_rp {constexpr bool use_c12ag_deboer17=false, use_tables=false;}
using namespace network_rp;
constexpr int NumSpec=13, neqs=14, net_ienuc=14;
struct burn_t {double rho,T,eta=0,y_e=.5; std::array<double,13> xn;};
"""
    properties = f"""
namespace Species {{enum {{{names}}};}}
constexpr std::array<double,{len(species)}> aion={{{masses}}},zion={{{charges}}};
constexpr auto aion_inv=[]{{std::array<double,13> a{{}};for(int i=0;i<13;i++) a[i]=1/aion[i];return a;}}();
namespace NetworkProperties {{constexpr double aion(int s){{return ::aion[s-1];}} constexpr double zion(int s){{return ::zion[s-1];}}}}
"""
    screen = original["screening/screen.H"]
    start = screen.index("template <typename number_t =")
    screening = headers["screening/screen_data.H"].replace("constexpr", "")
    screening += screen[start : screen.index("};", start) + 2]
    screening += function(screen, "fill_plasma_state").replace("constexpr", "")
    screening += function(screen, "actual_screen5")
    screening += r"""
bool screening_enabled=true;
template<class T> T actual_log_screen(const plasma_state_t<T>& s,const scrn::screen_factors_t& p) {return screening_enabled ? actual_screen5(s,p) : 0.;}
"""
    screening += function(screen, "actual_screen")
    fits = original["rates/aprox_rates.H"]
    corrections = []
    for name in ("rate_cr48ap", "rate_fe52ap"):
        old = routine(fits, name)
        new = old.replace("rr    = den * rev * term;", "fr = den * term; rr = den * rev * term;")
        if new == old:
            raise ValueError("expected missing forward assignment: " + name)
        fits = fits.replace(old, new)
        corrections.append(name + ": restored forward = density * term")
    functions = (
        *FUNCTIONS,
        *("rate_" + n + "ag" for n in ALPHA_TARGETS),
        *("rate_" + n + "ap" for n in PROTON_PARENTS),
        *("rate_" + n + "pg" for n in INTERMEDIATES),
    )
    rhs = original["networks/rhs.H"]
    support = block(rhs, "constexpr int factorial")
    support += function(rhs, "is_rate_used")
    support += block(rhs, "constexpr int num_intermediate_reactions")
    support += block(rhs, "constexpr int\nlocate_intermediate_rate_index")
    for name in (
        "density_exponent_forward",
        "density_exponent_reverse",
        "apply_density_scaling",
        "apply_screening",
        "apply_branching",
        "rhs_term",
        "construct_rate",
        "fill_additional_rates",
    ):
        value = function(rhs, name)
        if name == "apply_screening":
            value = value.replace("constexpr auto scn_fac", "const auto scn_fac")
            value = re.sub(r"static_assert\(scn_fac.*?;", "", value)
        support += value
    # Tabulation is disabled; declaration satisfies the unused upstream branch.
    support = (
        "template<int R,class T> void evaluate_tabulated_rate(const rhs_state_t<T>&,rate_t<T>&);\n"
        + support
    )
    support += function(rhs, "rhs")
    main = r"""
int main() {
 std::cout<<std::setprecision(17); burn_t state; int screened;
 while(std::cin>>state.rho>>state.T>>screened) {
  screening_enabled=screened;
  for(auto& x:state.xn) std::cin>>x;
  amrex::Array1D<double,1,14> derivative;
  RHS::rhs<14>(state,derivative);
  for(int i=1;i<=13;i++) std::cout<<derivative(i)*aion[i-1]<<" ";
  std::cout<<derivative(14)<<"\n";
 }
}
"""
    cpp = (
        prefix
        + headers["constants/fundamental_constants.H"]
        + properties
        + headers["interfaces/tfactors.H"]
        + screening
        + "\n"
        + headers["interfaces/rhs_type.H"]
        + "\n".join(routine(fits, n) for n in functions)
        + "\n"
        + headers["networks/aprox13/actual_network.H"]
        + "\nnamespace RHS {\n"
        + support
        + "\n}\n"
        + main
    )
    return cpp, dict(
        source="AMReX-Astro/Microphysics@" + COMMIT,
        corrections=corrections,
        source_sha256={
            name: hashlib.sha256(value.encode()).hexdigest() for name, value in original.items()
        },
        settings=dict(rate_tables=False, debye_huckel_skip=False, neutrinos=False),
        harness_adjustments=[
            "runtime screening pair calculation replaces compile-time evaluation; same formulas"
        ],
    )


def generate(source, output, compiler):
    cpp, metadata = build(source)
    rows = []
    mixtures = [np.eye(13)[0], (np.eye(13)[1] + np.eye(13)[2]) / 2, np.ones(13) / 13]
    saved = Path(reference_path("luna/wd_validation/typeia_burns_cpu.json"))
    for case in json.loads(saved.read_text())["cases"]:
        for temp, state in zip(case["reference_temperature"], case["reference"]):
            rows.extend([case["density"], temp, screened, *state[:13]] for screened in (0, 1))
    for density in (1e4, 1e6, 1e9):
        for temperature in (1e8, 1e9, 2e9, 5e9, 8e9):
            rows.extend(
                [density, temperature, screened, *x] for x in mixtures for screened in (0, 1)
            )
    with tempfile.TemporaryDirectory() as directory:
        path, binary = Path(directory) / "rhs.cpp", Path(directory) / "rhs"
        path.write_text(cpp)
        subprocess.run([compiler, "-std=c++20", "-O2", str(path), "-o", str(binary)], check=True)
        result = subprocess.run(
            [str(binary)],
            input="\n".join(" ".join(map(str, row)) for row in rows),
            text=True,
            capture_output=True,
            check=True,
        )
    values = np.loadtxt(result.stdout.splitlines())
    if not np.isfinite(values).all():
        raise ValueError("nonfinite reference")
    output.write_text(
        json.dumps(dict(**metadata, inputs=rows, derivatives=values.tolist()), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    generate(args.source, args.output, args.compiler)
