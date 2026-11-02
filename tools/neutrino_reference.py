"""Compile four default sneut5 components into references, without copying upstream source into LUNA."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from screening_reference import COMMIT, function


def generate(source, output, compiler):
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if revision != COMMIT:
        raise ValueError("unexpected Microphysics revision")
    header = (source / "neutrinos/sneut5.H").read_text()
    constants = header[
        header.index("namespace nu_constants") : header.index("template <typename number_t>")
    ]
    start = header.index("struct sneutf_t")
    factors = "template <typename number_t>\n" + header[start : header.index("};", start) + 2]
    routines = "\n".join(
        function(header, name)
        for name in ("get_sneut_factors", "nu_pair", "nu_plasma", "nu_photo", "nu_brem")
    )
    # double needs none of the AD dimension-narrowing machinery.
    prefix = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>
#define AMREX_GPU_HOST_DEVICE
#define AMREX_INLINE
namespace amrex {
using Real=double;
namespace literals {
constexpr double operator""_rt(long double x) {return double(x);}
constexpr double operator""_rt(unsigned long long x) {return double(x);}
}
namespace Math {auto sincos(double x) {return std::pair{std::sin(x),std::cos(x)};}}
}
namespace admath {
using std::sqrt; using std::exp; using std::pow; using std::log; using std::log10;
using std::abs; using std::sin; using std::cos; using std::min; using std::max;
template<int N> double powi(double x) {return std::pow(x,N);}
auto sincos(double x) {return std::pair{std::sin(x),std::cos(x)};}
}
namespace autodiff {template<class T> T eval(T x) {return x;} template<class T> T val(T x) {return x;}}
namespace microphysics_autodiff {template<class T,class U> T narrow_array(U x) {return x;}}
template<class T> using dt_t=T;
template<class T> using dcomp_t=T;
using namespace amrex::literals;
"""
    main = r"""
int main() {
 double rho,T,A,Z; std::cout<<std::setprecision(17);
 while(std::cin>>rho>>T>>A>>Z) {
  if(T<1e7) {std::cout<<"0 0 0 0\n"; continue;}
  auto state=get_sneut_factors(rho,T,A,Z);
  std::cout<<nu_pair(state)/rho<<" "<<nu_plasma(state)/rho<<" "
   <<nu_photo(state)/rho<<" "<<nu_brem(state)/rho<<"\n";
 }
}
"""
    rows = []
    temperatures = np.unique(
        np.r_[
            np.logspace(6, 10, 13),
            (np.array([1e7, 1e8, 1e9, 1e10])[:, None] * np.array([1 - 1e-8, 1, 1 + 1e-8])).ravel(),
        ]
    )
    for mass in (4.0, 12.0, 1 / (0.5 / 12 + 0.5 / 16)):
        for density in (10.0, 1e4, 1e6, 1e9, 5e9):
            rows.extend([density, temp, mass, mass / 2] for temp in temperatures)
    with tempfile.TemporaryDirectory() as directory:
        cpp, binary = Path(directory) / "reference.cpp", Path(directory) / "reference"
        cpp.write_text(prefix + constants + factors + routines + main)
        subprocess.run([compiler, "-std=c++20", "-O2", str(cpp), "-o", str(binary)], check=True)
        result = subprocess.run(
            [str(binary)],
            input="\n".join(" ".join(map(str, row)) for row in rows),
            check=True,
            capture_output=True,
            text=True,
        )
    values = np.loadtxt(result.stdout.splitlines())
    if not np.isfinite(values).all():
        raise ValueError("nonfinite reference")
    output.write_text(
        json.dumps(
            dict(
                source="AMReX-Astro/Microphysics@" + COMMIT,
                source_sha256=hashlib.sha256(header.encode()).hexdigest(),
                include_recombination=False,
                columns=["pair", "plasma", "photo", "bremsstrahlung"],
                inputs=rows,
                values=values.tolist(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    generate(args.source, args.output, args.compiler)
