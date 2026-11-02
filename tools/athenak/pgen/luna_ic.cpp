//========================================================================================
// AthenaXXX astrophysical plasma code -- luna_ic pgen
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file luna_ic.cpp
//! \brief Hydro initial data from a LUNA IC directory (U.npy + meta.json), and
//! double-precision dumps in the same layout, so both codes start from the same cells
//! and are measured on the same cells.
//!
//! LUNA IC (src/luna/ic_generation/ic_file.py): U.npy, little-endian float64, C order,
//! shape (5, Nx, Ny, Nz), variables rho, mx, my, mz, E (= AthenaK IDN, IM1, IM2, IM3,
//! IEN), cell i at x1min + (i + 1/2) dx. 1D and 2D runs use Ny = Nz = 1 / Nz = 1.
//! Each rank reads only its own MeshBlocks' cells (pread); uniform meshes only.
//!
//! <problem>
//!   ic_dir     = DIR    (required) holds U.npy and meta.json
//!   dump       = false  at every history output write <basename>.luna.NNNNN.npy
//!                       (u0, same layout as U.npy) and append "index time ncycle" to
//!                       <basename>.luna_dumps.txt; needs user_hist = true and an hst
//!                       output block (its dt is the dump cadence)
//!   dump_vars  = 5      how many leading variables the history-cadence dumps hold
//!                       (1: rho only, to keep long series small)
//!   final_dump = true   write <basename>.luna.final.npy (all 5 variables) after the
//!                       last step
//!   errors     = false  after the last step, AthenaK's OutputErrors (u0 against the IC,
//!                       re-read into u1): <basename>-errs.dat, RMS-L1 over variables
//!   potential  = none   a static external potential as a user source term (needs
//!                       user_srcs = true): "sin", phi = phi0 sin(2 pi x_axis / wavelength),
//!                       g = -dphi/dx at the cell centres x = xmin + (i + 1/2) dx, i the
//!                       global cell index (LUNA's external_potential = "sin"); each RK
//!                       stage adds bdt rho g to the momentum and bdt rho v g to the energy
//!                       from the stage's primitives, as AthenaK's const_accel
//!   phi0 = 1.0, wavelength = 1.0, axis = 1 (1..3)
//! Uniform gravity through AthenaK's own const_accel in <hydro_srcterms> (not <hydro>).
//! A meta.json whose Lx, Ly, Lz or gamma disagrees with the input is refused.

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>    // NOLINT
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "outputs/outputs.hpp"
#include "pgen.hpp"

namespace {

std::string ic_dir_;
std::string basename_;
bool dump_ = false;
bool final_dump_ = false;
bool errors_ = false;
int ndump_ = 0;
int dump_vars_ = 5;
Real phi0_ = 1.0;
Real wavelength_ = 1.0;
int axis_ = 1;

[[noreturn]] void Fail(const std::string &msg) {
  std::cout << "### FATAL ERROR in luna_ic pgen: " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

// global shape (Nx, Ny, Nz) of the mesh, as LUNA writes it
void MeshShape(Mesh *pm, std::int64_t shape[3]) {
  shape[0] = pm->mesh_indcs.nx1;
  shape[1] = pm->mesh_indcs.nx2;
  shape[2] = pm->mesh_indcs.nx3;
}

// npy v1.0 header for float64 (nvar, Nx, Ny, Nz), padded to a multiple of 64 bytes
std::string NpyHeader(int nvar, const std::int64_t shape[3]) {
  std::ostringstream d;
  d << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << nvar << ", " << shape[0] << ", "
    << shape[1] << ", " << shape[2] << "), }";
  std::string dict = d.str();
  std::size_t total = 10 + dict.size() + 1;
  std::size_t pad = (64 - total % 64) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');
  std::string h("\x93NUMPY\x01\x00", 8);
  std::uint16_t hl = static_cast<std::uint16_t>(dict.size());
  h.push_back(static_cast<char>(hl & 0xff));
  h.push_back(static_cast<char>(hl >> 8));
  return h + dict;
}

// byte offset of the data in an npy file whose shape must be (5, Nx, Ny, Nz) float64 C
std::int64_t NpyDataOffset(const std::string &fname, const std::int64_t want[3]) {
  std::ifstream f(fname, std::ios::binary);
  if (!f) Fail("cannot open " + fname);
  char magic[10];
  f.read(magic, 10);
  if (!f || std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
    Fail(fname + " is not an npy file");
  }
  std::int64_t hlen, start;
  if (magic[6] == 1) {
    hlen = static_cast<unsigned char>(magic[8]) | (static_cast<unsigned char>(magic[9]) << 8);
    start = 10;
  } else {
    char b[2];
    f.read(b, 2);
    hlen = static_cast<unsigned char>(magic[8]) |
           (static_cast<unsigned char>(magic[9]) << 8) |
           (static_cast<std::int64_t>(static_cast<unsigned char>(b[0])) << 16) |
           (static_cast<std::int64_t>(static_cast<unsigned char>(b[1])) << 24);
    start = 12;
  }
  std::string hdr(hlen, ' ');
  f.read(&hdr[0], hlen);
  if (hdr.find("'descr': '<f8'") == std::string::npos) Fail(fname + ": dtype not <f8");
  if (hdr.find("'fortran_order': False") == std::string::npos) {
    Fail(fname + ": not C order");
  }
  std::smatch m;
  if (!std::regex_search(hdr, m, std::regex("'shape': \\(([^)]*)\\)"))) {
    Fail(fname + ": no shape in header");
  }
  std::vector<std::int64_t> s;
  std::string body = m[1];
  std::regex num("[0-9]+");
  for (auto it = std::sregex_iterator(body.begin(), body.end(), num);
       it != std::sregex_iterator(); ++it) {
    s.push_back(std::stoll(it->str()));
  }
  if (s.size() != 4 || s[0] != 5 || s[1] != want[0] || s[2] != want[1] || s[3] != want[2]) {
    std::ostringstream e;
    e << fname << ": shape (" << body << ") is not (5, " << want[0] << ", " << want[1]
      << ", " << want[2] << ") of the mesh";
    Fail(e.str());
  }
  return start + hlen;
}

// a number from meta.json by key, NAN when absent
double MetaValue(const std::string &meta, const std::string &key) {
  std::smatch m;
  std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
  if (std::regex_search(meta, m, re)) return std::stod(m[1]);
  return NAN;
}

void CheckMeta(Mesh *pm, Real gamma) {
  std::ifstream f(ic_dir_ + "/meta.json");
  if (!f) Fail("cannot open " + ic_dir_ + "/meta.json");
  std::stringstream ss;
  ss << f.rdbuf();
  std::string meta = ss.str();
  auto &ms = pm->mesh_size;
  const double run[4] = {ms.x1max - ms.x1min, ms.x2max - ms.x2min, ms.x3max - ms.x3min,
                         gamma};
  const char *keys[4] = {"Lx", "Ly", "Lz", "gamma"};
  const int nx[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  for (int n = 0; n < 4; ++n) {
    if (n < 3 && nx[n] == 1) continue;           // a collapsed direction has no length
    double v = MetaValue(meta, keys[n]);
    if (std::isnan(v)) Fail(std::string("meta.json has no ") + keys[n]);
    if (std::abs(v - run[n]) > 1e-12 * std::abs(run[n])) {
      std::ostringstream e;
      e.precision(17);
      e << "meta.json " << keys[n] << " = " << v << " but the input gives " << run[n];
      Fail(e.str());
    }
  }
}

// the global cell offsets (i0, j0, k0) of MeshBlock m of this rank
void BlockOrigin(Mesh *pm, int m, std::int64_t o[3]) {
  auto &indcs = pm->mb_indcs;
  int gid = pm->pmb_pack->pmb->mb_gid.h_view(m);
  LogicalLocation &ll = pm->lloc_eachmb[gid];
  o[0] = static_cast<std::int64_t>(ll.lx1) * indcs.nx1;
  o[1] = static_cast<std::int64_t>(ll.lx2) * indcs.nx2;
  o[2] = static_cast<std::int64_t>(ll.lx3) * indcs.nx3;
}

// read (write = false) or write this rank's interior cells of arr from/to fname at off
void BlockIO(Mesh *pm, DvceArray5D<Real> &arr, const std::string &fname, std::int64_t off,
             bool write, int nvar = 5) {
  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  std::int64_t S[3];
  MeshShape(pm, S);
  int nmb = pm->pmb_pack->nmb_thispack;
  auto h = Kokkos::create_mirror_view(arr);
  if (write) Kokkos::deep_copy(h, arr);
  int fd = write ? open(fname.c_str(), O_WRONLY) : open(fname.c_str(), O_RDONLY);
  if (fd < 0) Fail("cannot open " + fname);
  // LUNA order: k (z) fastest; a block spanning all of z is contiguous over j as well
  const bool zfull = (nx3 == S[2]);
  const int run = zfull ? nx2 * nx3 : nx3;
  std::vector<double> buf(run);
  for (int m = 0; m < nmb; ++m) {
    std::int64_t o[3];
    BlockOrigin(pm, m, o);
    for (int n = 0; n < nvar; ++n) {
      for (int i = 0; i < nx1; ++i) {
        for (int j = 0; j < (zfull ? 1 : nx2); ++j) {
          std::int64_t cell = ((n * S[0] + o[0] + i) * S[1] + o[1] + j) * S[2] + o[2];
          off_t pos = off + 8 * cell;
          if (write) {
            for (int q = 0; q < run; ++q) {
              int jj = zfull ? q / nx3 : j, kk = zfull ? q % nx3 : q;
              buf[q] = h(m, n, ks + kk, js + jj, is + i);
            }
            if (pwrite(fd, buf.data(), 8 * run, pos) != 8 * run) Fail("short write " + fname);
          } else {
            if (pread(fd, buf.data(), 8 * run, pos) != 8 * run) Fail("short read " + fname);
            for (int q = 0; q < run; ++q) {
              int jj = zfull ? q / nx3 : j, kk = zfull ? q % nx3 : q;
              h(m, n, ks + kk, js + jj, is + i) = buf[q];
            }
          }
        }
      }
    }
  }
  close(fd);
  if (!write) Kokkos::deep_copy(arr, h);
}

void ReadIC(Mesh *pm, DvceArray5D<Real> &arr) {
  std::int64_t S[3];
  MeshShape(pm, S);
  std::string f = ic_dir_ + "/U.npy";
  BlockIO(pm, arr, f, NpyDataOffset(f, S), false);
}

void WriteDump(Mesh *pm, const std::string &fname, int nvar) {
  std::int64_t S[3];
  MeshShape(pm, S);
  std::string h = NpyHeader(nvar, S);
  if (global_variable::my_rank == 0) {
    FILE *f = std::fopen(fname.c_str(), "wb");
    if (f == nullptr) Fail("cannot create " + fname);
    std::fwrite(h.data(), 1, h.size(), f);
    std::fclose(f);
    if (truncate(fname.c_str(), h.size() + 8 * nvar * S[0] * S[1] * S[2]) != 0) {
      Fail("cannot size " + fname);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  BlockIO(pm, pm->pmb_pack->phydro->u0, fname, h.size(), true, nvar);
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void LunaHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 1;
  pdata->label[0] = "luna-dump";
  pdata->hdata[0] = 0.0;
  if (!dump_) return;
  char num[16];
  std::snprintf(num, sizeof(num), "%05d", ndump_);
  WriteDump(pm, basename_ + ".luna." + num + ".npy", dump_vars_);
  if (global_variable::my_rank == 0) {
    FILE *f = std::fopen((basename_ + ".luna_dumps.txt").c_str(), ndump_ ? "a" : "w");
    std::fprintf(f, "%05d %.17e %d\n", ndump_, pm->time, pm->ncycle);
    std::fclose(f);
    pdata->hdata[0] = ndump_;
  }
  ++ndump_;
}

void LunaFinal(ParameterInput *pin, Mesh *pm) {
  if (final_dump_) {
    WriteDump(pm, basename_ + ".luna.final.npy", 5);
    if (global_variable::my_rank == 0) {
      FILE *f = std::fopen((basename_ + ".luna_final.txt").c_str(), "w");
      std::fprintf(f, "%.17e %d\n", pm->time, pm->ncycle);
      std::fclose(f);
    }
  }
  if (errors_) {
    ReadIC(pm, pm->pmb_pack->phydro->u1);
    pm->pgen->OutputErrors(pin, pm);
  }
}

// phi = phi0 sin(2 pi x / wavelength) along axis_: rho g into the momentum, rho v g into the
// energy, from the stage primitives w0 (u0 is partly updated when user sources run)
void SinPotential(Mesh *pm, const Real bdt) {
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pm->pmb_pack->nmb_thispack - 1;
  auto &u0 = pm->pmb_pack->phydro->u0;
  auto &w0 = pm->pmb_pack->phydro->w0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  const int dir = axis_;
  const Real k = 2.0 * M_PI / wavelength_;
  const Real phi0 = phi0_;
  Real xmin, xmax;
  int nmesh;
  if (dir == 1) {
    xmin = pm->mesh_size.x1min; xmax = pm->mesh_size.x1max; nmesh = pm->mesh_indcs.nx1;
  } else if (dir == 2) {
    xmin = pm->mesh_size.x2min; xmax = pm->mesh_size.x2max; nmesh = pm->mesh_indcs.nx2;
  } else {
    xmin = pm->mesh_size.x3min; xmax = pm->mesh_size.x3max; nmesh = pm->mesh_indcs.nx3;
  }
  const Real dx = (xmax - xmin) / nmesh;
  par_for("luna_sinpot", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int kk, const int j, const int i) {
    Real bmin = (dir == 1) ? size.d_view(m).x1min
              : ((dir == 2) ? size.d_view(m).x2min : size.d_view(m).x3min);
    int local = (dir == 1) ? i - is : ((dir == 2) ? j - js : kk - ks);
    int gi = static_cast<int>(floor((bmin - xmin) / dx + 0.5)) + local;
    Real x = xmin + (gi + 0.5) * dx;
    Real g = -phi0 * k * cos(k * x);
    Real src = bdt * g * w0(m, IDN, kk, j, i);
    u0(m, dir, kk, j, i) += src;
    u0(m, IEN, kk, j, i) += src * w0(m, dir, kk, j, i);
  });
}

}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  Mesh *pm = pmy_mesh_;
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pmhd != nullptr) Fail("needs <hydro> and no <mhd>");
  if (pm->multilevel) Fail("uniform meshes only (no SMR/AMR)");
  if (pmbp->phydro->nhydro != 5 || pmbp->phydro->nscalars != 0) {
    Fail("needs the ideal EOS (5 conserved variables) and no passive scalars");
  }
  ic_dir_ = pin->GetString("problem", "ic_dir");
  basename_ = pin->GetString("job", "basename");
  dump_ = pin->GetOrAddBoolean("problem", "dump", false);
  final_dump_ = pin->GetOrAddBoolean("problem", "final_dump", true);
  dump_vars_ = pin->GetOrAddInteger("problem", "dump_vars", 5);
  if (dump_vars_ < 1 || dump_vars_ > 5) Fail("dump_vars must be 1..5");
  errors_ = pin->GetOrAddBoolean("problem", "errors", false);
  if (dump_ && !user_hist) Fail("dump = true needs <problem> user_hist = true");
  if (user_hist) user_hist_func = LunaHistory;
  pgen_final_func = LunaFinal;
  std::string pot = pin->GetOrAddString("problem", "potential", "none");
  if (pot == "sin") {
    if (!user_srcs) Fail("potential = sin needs <problem> user_srcs = true");
    phi0_ = pin->GetOrAddReal("problem", "phi0", 1.0);
    wavelength_ = pin->GetOrAddReal("problem", "wavelength", 1.0);
    axis_ = pin->GetOrAddInteger("problem", "axis", 1);
    if (axis_ < 1 || axis_ > 3) Fail("axis must be 1..3");
    user_srcs_func = SinPotential;
  } else if (pot != "none") {
    Fail("potential must be none or sin, not " + pot);
  } else if (user_srcs) {
    Fail("user_srcs = true but no potential");
  }
  if (restart) return;
  CheckMeta(pm, pmbp->phydro->peos->eos_data.gamma);
  ReadIC(pm, pmbp->phydro->u0);
  if (global_variable::my_rank == 0) {
    std::cout << "luna_ic: read " << ic_dir_ << "/U.npy (" << pm->mesh_indcs.nx1 << " x "
              << pm->mesh_indcs.nx2 << " x " << pm->mesh_indcs.nx3 << ", "
              << pm->nmb_total << " MeshBlocks on " << global_variable::nranks
              << " ranks)" << std::endl;
    if (user_srcs) {
      std::cout << "luna_ic: potential sin, phi0 " << phi0_ << ", wavelength " << wavelength_
                << ", axis " << axis_ << std::endl;
    }
  }
}
