"""Compile external Helmholtz and prescribed-ash CJ routines into numerical reference fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from helmholtz_reference import COMMIT, build


def generate(source, output, compiler="c++"):
    program = build(source)
    prefix, original_main = program.split("int main(int argc,char** argv)", 1)
    prefix = prefix.replace("double rho,T,abar,zbar,y_e;", "double rho,T,abar,zbar,y_e,gam1;")
    setup = original_main[
        original_main.index("{") + 1 : original_main.index(" State q; int coulomb;")
    ]
    cj = (source / "util/cj_detonation/cj_det.H").read_text()
    bridge = r"""
#include <stdexcept>
using Real=double;
using eos_t=State;
constexpr int eos_input_rt=0;
bool use_coulomb=true;
namespace amrex {void Error(const char* x) {throw std::runtime_error(x);}}
void eos(int,eos_t& q) {
 q.y_e=q.zbar/q.abar;
 apply_radiation(q);apply_ions(q);apply_electrons(q);
 if(use_coulomb) apply_coulomb_corrections(q);
 double cs2=q.dpdr+q.T*q.dpdT*q.dpdT/(q.rho*q.rho*q.dedT);
 q.gam1=q.rho*cs2/q.p;
}
"""
    main = r"""
 State fuel; double heat; int coulomb;
 std::cout<<std::setprecision(17);
 while(std::cin>>fuel.rho>>fuel.T>>fuel.abar>>fuel.zbar>>heat>>coulomb) {
  use_coulomb=coulomb; eos(0,fuel);
  State ash=fuel; ash.abar=56; ash.zbar=28;
  cj_cond(fuel,ash,heat); eos(0,ash);
  double speed=std::sqrt((ash.p-fuel.p)/(1/fuel.rho-1/ash.rho))/fuel.rho;
  std::cout<<ash.rho<<" "<<ash.T<<" "<<ash.p<<" "<<speed<<"\n";
 }
}
"""
    network = (source / "networks/aprox13/actual_network.H").read_text()
    binding = [float(value) for value in re.findall(r"return\s+([0-9.]+)e0_rt;", network)[:13]]
    if len(binding) != 13:
        raise ValueError("unexpected upstream binding-energy constants")
    masses = np.array([4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56])
    per_mass = np.array(binding) / masses * 6.0221417930e23 * 1.60217648740e-6
    rows, inputs = [], []
    for case, density, temperature in (
        ("he", 1e6, 1e8),
        ("he", 1e7, 1e8),
        ("co", 1e6, 1e8),
        ("co", 1e7, 1e8),
        ("co", 3e8, 2e8),
    ):
        fractions = np.zeros(13)
        if case == "he":
            fractions[0] = 1
        else:
            fractions[1:3] = 0.5
        mass = 1 / np.dot(fractions, 1 / masses)
        heat = float(per_mass[-1] - np.dot(fractions, per_mass))
        for coulomb in (False, True):
            rows.append([density, temperature, mass, mass / 2, heat, int(coulomb)])
            inputs.append(
                dict(
                    case=case,
                    density=density,
                    temperature=temperature,
                    fuel=fractions.tolist(),
                    ash=[0.0] * 12 + [1.0],
                    coulomb=coulomb,
                    heat=heat,
                )
            )
    with tempfile.TemporaryDirectory(prefix="luna-cj-reference-") as temporary:
        cpp, executable = Path(temporary) / "reference.cpp", Path(temporary) / "reference"
        cpp.write_text(prefix + bridge + cj + "\nint main(int argc,char** argv) {" + setup + main)
        subprocess.run([compiler, "-std=c++20", "-O2", str(cpp), "-o", str(executable)], check=True)
        result = subprocess.run(
            [str(executable), str(source / "EOS/helmholtz/helm_table.dat")],
            input="\n".join(" ".join(map(str, row)) for row in rows),
            text=True,
            capture_output=True,
            check=True,
        )
    values = np.loadtxt(result.stdout.splitlines())
    if values.shape != (len(inputs), 4) or not np.isfinite(values).all():
        raise ValueError("invalid CJ reference output")
    report = dict(
        source=f"AMReX-Astro/Microphysics@{COMMIT}",
        scope="prescribed pure Ni56 ash and frozen-sonic CJ; not NSE or self-sustaining network eigenvalues",
        cj_sha256=hashlib.sha256(cj.encode()).hexdigest(),
        eos_sha256=hashlib.sha256((source / "EOS/helmholtz/actual_eos.H").read_bytes()).hexdigest(),
        table_sha256=hashlib.sha256(
            (source / "EOS/helmholtz/helm_table.dat").read_bytes()
        ).hexdigest(),
        columns=["density", "temperature", "pressure", "speed"],
        inputs=inputs,
        values=values.tolist(),
    )
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    generate(args.source, args.output, args.compiler)
