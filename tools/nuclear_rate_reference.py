"""Compile five upstream rate routines into numerical references; upstream sources remain external."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

COMMIT = "99b9b47cdb6d0d459deefeacdd4d78639bbd5f93"
FUNCTIONS = ("rate_triplealf", "rate_c12ag", "rate_c12c12", "rate_c12o16", "rate_o16o16")
NAMES = ("triple_alpha", "carbon_alpha", "carbon_fusion", "carbon_oxygen_fusion", "oxygen_fusion")

ALPHA_TARGETS = ("o16", "ne20", "mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48", "fe52")
PROTON_PARENTS = ("mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48", "fe52")
INTERMEDIATES = ("al27", "p31", "cl35", "k39", "sc43", "v47", "mn51", "co55")


def routine(source, name):
    position = source.index("void " + name + "(")
    start = source.rfind("template <", 0, position)
    end = source.index("{", position) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def generate(source, output, compiler, full=False):
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if revision != COMMIT:
        raise ValueError(f"expected Microphysics {COMMIT}")
    fits = (source / "rates/aprox_rates.H").read_text()
    corrections = []
    if full:
        for name in ("rate_cr48ap", "rate_fe52ap"):
            original = routine(fits, name)
            corrected = original.replace(
                "rr    = den * rev * term;", "fr = den * term; rr = den * rev * term;"
            )
            if original == corrected:
                raise ValueError(f"expected missing forward assignment in {name}")
            fits = fits.replace(original, corrected)
            corrections.append(name + ": forward = density * term (matches current upstream)")
    functions = FUNCTIONS
    names = NAMES
    if full:
        functions += tuple("rate_" + name + "ag" for name in ALPHA_TARGETS)
        functions += tuple("rate_" + name + "ap" for name in PROTON_PARENTS)
        functions += tuple("rate_" + name + "pg" for name in INTERMEDIATES)
        names += tuple("alpha:" + str(i) for i in range(1, 11))
        names += tuple("alpha_proton:" + str(i) for i in range(8))
        names += tuple("proton_capture:" + str(i) for i in range(8))
    factors = (source / "interfaces/tfactors.H").read_text()
    factors = re.sub(r"^#.*$", "", factors, flags=re.M)
    prefix = r"""
#include <cmath>
#include <iomanip>
#include <iostream>
#define AMREX_GPU_HOST_DEVICE
#define AMREX_FLATTEN
namespace amrex {
using Real=double;
namespace literals {
constexpr double operator""_rt(long double x) {return double(x);}
constexpr double operator""_rt(unsigned long long x) {return double(x);}
}
}
namespace admath { using std::sqrt; using std::exp; using std::pow; using std::cbrt; using std::log; using std::min; }
namespace autodiff {template<class T> T eval(T x) {return x;}}
bool use_c12ag_deboer17=false;
using namespace amrex::literals;
"""
    calls = "".join(
        f'{name}(tf,1.,forward,reverse); std::cout<<forward<<" "<<reverse<<" ";'
        for name in functions
    )
    main = (
        r"int main() {std::cout<<std::setprecision(17); double T,forward=0,reverse=0;"
        r"while(std::cin>>T) {auto tf=get_tfactors(T);" + calls + r'std::cout<<"\n";}}'
    )
    temperatures = np.unique(
        np.r_[
            np.logspace(6, 10, 33),
            8e7 * np.array([1 - 1e-9, 1, 1 + 1e-9]),
            5e8 * np.array([1 - 1e-9, 1, 1 + 1e-9]),
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        cpp, binary = Path(directory) / "reference.cpp", Path(directory) / "reference"
        cpp.write_text(prefix + factors + "\n".join(routine(fits, n) for n in functions) + main)
        subprocess.run([compiler, "-std=c++20", "-O2", str(cpp), "-o", str(binary)], check=True)
        result = subprocess.run(
            [str(binary)],
            input="\n".join(map(str, temperatures)),
            check=True,
            text=True,
            capture_output=True,
        )
    values = np.loadtxt(result.stdout.splitlines()).reshape(-1, len(functions), 2)
    output.write_text(
        json.dumps(
            dict(
                source="AMReX-Astro/Microphysics@" + COMMIT,
                source_sha256=hashlib.sha256(fits.encode()).hexdigest(),
                rates=names,
                corrections=corrections,
                temperature=temperatures.tolist(),
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
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    generate(args.source, args.output, args.compiler, args.full)
