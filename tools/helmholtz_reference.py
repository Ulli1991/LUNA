"""Generate forward-EOS reference states by compiling pinned Microphysics routines, without AMReX."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

COMMIT = "99b9b47cdb6d0d459deefeacdd4d78639bbd5f93"


def build(source):
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != COMMIT:
        raise ValueError(f"expected Microphysics {COMMIT}, got {revision}")
    eos = source / "EOS/helmholtz"
    header = (eos / "actual_eos.H").read_text()
    routines = header[header.index("constexpr std::string_view eos_name") :]
    routines = routines[: routines.index("template <typename I, typename T>")]
    data = (eos / "actual_eos_data.H").read_text()
    data = re.sub(r"^#.*$", "", data, flags=re.M).replace("extern ", "")
    traits = sorted(set(re.findall(r"has_\w+", routines)))
    prefix = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <string_view>
#include <type_traits>
#define AMREX_GPU_HOST_DEVICE
#define AMREX_GPU_MANAGED
#define AMREX_FORCE_INLINE inline
#define AMREX_INLINE inline
#define AMREX_RESTRICT
namespace amrex {
using Real = double;
namespace literals {
constexpr double operator""_rt(long double x) { return double(x); }
constexpr double operator""_rt(unsigned long long x) { return double(x); }
}
template<class T> T Clamp(T x,T a,T b) {return std::clamp(x,a,b);}
using std::max;
template<int A,int B,class F> void constexpr_for(F f) {
 if constexpr(A<B) { f(std::integral_constant<int,A>{}); constexpr_for<A+1,B>(f); }
}
}
namespace eos_rp { double prad_limiter_rho_c=0, prad_limiter_delta_rho=0; }
"""
    for trait in traits:
        enabled = trait in ("has_pressure", "has_energy", "has_entropy")
        prefix += f"template<class T> constexpr bool {trait} = {'true' if enabled else 'false'};\n"
    main = r"""
struct State {
 double rho,T,abar,zbar,y_e;
 double p=0,e=0,s=0,dpdr=0,dpdT=0,dedr=0,dedT=0,dsdr=0,dsdT=0;
};
int main(int argc,char** argv) {
 using namespace helmholtz;
 std::ifstream table(argv[1]);
 int order[9]={0,3,1,4,2,5,6,7,8};
 for(int j=0;j<jmax;j++) for(int i=0;i<imax;i++)
  for(int k=0;k<9;k++) table >> f[j][i][order[k]];
 for(int j=0;j<jmax;j++) for(int i=0;i<imax;i++)
  for(int k=0;k<4;k++) { int dp_order[4]={0,2,1,3}; table >> dpdf[j][i][dp_order[k]]; }
 if(!table) return 2;
 for(int i=0;i<imax;i++) d[i]=std::pow(10.,dlo+i*dstp);
 for(int j=0;j<jmax;j++) t[j]=std::pow(10.,tlo+j*tstp);
 for(int i=0;i<imax-1;i++) {
  dd_sav[i]=d[i+1]-d[i]; dd2_sav[i]=dd_sav[i]*dd_sav[i];
  ddi_sav[i]=1/dd_sav[i]; dd2i_sav[i]=1/dd2_sav[i];
 }
 for(int j=0;j<jmax-1;j++) {
  dt_sav[j]=t[j+1]-t[j]; dt2_sav[j]=dt_sav[j]*dt_sav[j];
  dti_sav[j]=1/dt_sav[j]; dt2i_sav[j]=1/dt2_sav[j];
 }
 State q; int coulomb;
 std::cout << std::setprecision(17);
 while(std::cin>>q.rho>>q.T>>q.abar>>q.zbar>>coulomb) {
  q.y_e=q.zbar/q.abar;
  apply_radiation(q); apply_ions(q); apply_electrons(q);
  if(coulomb) apply_coulomb_corrections(q);
  double cs2=q.dpdr+q.T*q.dpdT*q.dpdT/(q.rho*q.rho*q.dedT);
  std::cout<<q.p<<" "<<q.e<<" "<<q.s<<" "<<q.dpdr<<" "<<q.dpdT<<" "
   <<q.dedr<<" "<<q.dedT<<" "<<q.rho*cs2/q.p<<" "<<std::sqrt(cs2)<<"\n";
 }
}
"""
    return prefix + data + "using namespace amrex::literals;" + routines + main


def generate(source, output, compiler):
    program = build(source)
    eos = source / "EOS/helmholtz"
    header = (eos / "actual_eos.H").read_text()
    rng = np.random.default_rng(41)
    states = [[1e6, 1e9, 4.0, 2.0]]
    for i in range(24):
        abar = (4.0, 12.0, 1 / (0.5 / 12 + 0.5 / 16))[i % 3]
        states.append([10 ** rng.uniform(-4, 10), 10 ** rng.uniform(5, 10), abar, abar / 2])
    inputs = [state + [c] for state in states for c in (0, 1)]
    with tempfile.TemporaryDirectory() as scratch:
        cpp, exe = Path(scratch) / "reference.cpp", Path(scratch) / "reference"
        cpp.write_text(program)
        subprocess.run([compiler, "-std=c++20", "-O2", str(cpp), "-o", str(exe)], check=True)
        run = subprocess.run(
            [str(exe), str(eos / "helm_table.dat")],
            check=True,
            text=True,
            input="\n".join(" ".join(map(str, row)) for row in inputs),
            capture_output=True,
        )
    values = np.loadtxt(run.stdout.splitlines())
    if not np.isfinite(values).all():
        raise ValueError("reference produced invalid states")
    result = dict(
        source=f"AMReX-Astro/Microphysics@{COMMIT}",
        source_sha256=hashlib.sha256(header.encode()).hexdigest(),
        table_sha256=hashlib.sha256((eos / "helm_table.dat").read_bytes()).hexdigest(),
        columns=[
            "pressure",
            "energy",
            "entropy",
            "dpdr",
            "dpdt",
            "dedr",
            "cv",
            "gamma1",
            "sound_speed",
        ],
        inputs=inputs,
        values=values.tolist(),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help=f"Microphysics checkout at {COMMIT}")
    parser.add_argument("output", type=Path)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    generate(args.source, args.output, args.compiler)
