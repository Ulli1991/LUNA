"""Compile the pinned screen5 routines into numerical references; source files stay external."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

COMMIT = "99b9b47cdb6d0d459deefeacdd4d78639bbd5f93"


def function(source, name):
    position = source.index(name + " (") if name + " (" in source else source.index(name + "(")
    start = source.rfind("template <", 0, position)
    end = source.index("{", position) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def generate(source, output, compiler):
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != COMMIT:
        raise ValueError("unexpected Microphysics revision")
    screen = (source / "screening/screen.H").read_text()
    factors = (source / "screening/screen_data.H").read_text()
    factors = re.sub(r"^#include.*$", "", factors, flags=re.M).replace("constexpr", "")
    start = screen.index("template <typename number_t =")
    plasma = screen[start : screen.index("};", start) + 2]
    fill = function(screen, "fill_plasma_state").replace("constexpr", "")
    evaluate = function(screen, "actual_screen5")
    prefix = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#define AMREX_GPU_HOST_DEVICE
#define AMREX_FORCE_INLINE
#define AMREX_INLINE
#define SCREEN_METHOD 1
#define SCREEN_METHOD_screen5 1
#define SCREEN_METHOD_chugunov2007 2
#define SCREEN_METHOD_chugunov2009 3
#define SCREEN_METHOD_chabrier1998 4
namespace amrex {
using Real=double;
namespace literals {
constexpr double operator""_rt(long double x) {return double(x);}
constexpr double operator""_rt(unsigned long long x) {return double(x);}
}
template<class T,int A,int B> struct Array1D {
 std::array<T,B-A+1> data;
 T& operator()(int n) {return data[n-A];}
 const T& operator()(int n) const {return data[n-A];}
};
}
namespace admath {using std::exp; using std::sqrt; using std::cbrt; using std::log; using std::min; using std::max;}
namespace gcem {using std::pow; using std::log;}
namespace C {constexpr double n_A=6.02214076e23,q_e=4.803204712570263e-10,k_B=1.380649e-16;}
constexpr int NumSpec=13;
double zion[13]={2,6,8,10,12,14,16,18,20,22,24,26,28};
double mass[13]={4,12,16,20,24,28,32,36,40,44,48,52,56};
using namespace amrex::literals;
"""
    main = r"""
int main() {
 double rho,T,z1,a1,z2,a2;
 amrex::Array1D<double,1,NumSpec> molar;
 std::cout<<std::setprecision(17);
 while(std::cin>>rho>>T>>z1>>a1>>z2>>a2) {
  for(int i=1;i<=NumSpec;i++) {double X; std::cin>>X; molar(i)=X/mass[i-1];}
  plasma_state_t<double> plasma;
  fill_plasma_state(plasma,T,rho,molar);
  auto pair=scrn::calculate_screen_factor(z1,a1,z2,a2);
  std::cout<<actual_screen5(plasma,pair)<<"\n";
 }
}
"""
    helium = np.eye(13)[0]
    mixture = (np.eye(13)[1] + np.eye(13)[2]) / 2
    pairs = [
        (2, 4, 2, 4),
        (2, 4, 4, 8),
        (6, 12, 6, 12),
        (6, 12, 8, 16),
        (8, 16, 8, 16),
        (2, 4, 26, 52),
        (1, 1, 27, 55),
    ]
    rows = []
    for fractions in (helium, mixture):
        for density in (1e2, 1e6, 1e9):
            for temperature in (1e7, 1e8, 1e9, 5e9):
                rows.extend([density, temperature, *pair, *fractions] for pair in pairs)
    pair_scale = 2 ** (1 / 3) * 36 / 12 ** (1 / 3)
    for coupling in (0.3 * (1 - 1e-8), 0.3 * (1 + 1e-8), 0.8 * (1 - 1e-8), 0.8 * (1 + 1e-8)):
        density = (coupling * 1e9 / (pair_scale * 2.27493e5)) ** 3 / 0.5
        rows.append([density, 1e9, 6, 12, 6, 12, *mixture])
    with tempfile.TemporaryDirectory() as directory:
        cpp, binary = Path(directory) / "reference.cpp", Path(directory) / "reference"
        cpp.write_text(prefix + factors + plasma + fill + evaluate + main)
        subprocess.run([compiler, "-std=c++20", "-O2", str(cpp), "-o", str(binary)], check=True)
        result = subprocess.run(
            [str(binary)],
            input="\n".join(" ".join(map(str, row)) for row in rows),
            check=True,
            capture_output=True,
            text=True,
        )
    values = np.loadtxt(result.stdout.splitlines())
    output.write_text(
        json.dumps(
            dict(
                source="AMReX-Astro/Microphysics@" + COMMIT,
                source_sha256=hashlib.sha256(screen.encode()).hexdigest(),
                inputs=rows,
                log_enhancement=values.tolist(),
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
