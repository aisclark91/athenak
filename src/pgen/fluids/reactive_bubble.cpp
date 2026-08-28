//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

#include <cmath>

#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"

namespace {
  Real r0;     // Maximum radius at t=t0.
  void SetReactiveSources(Mesh* pm, const Real bdt);
}


//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool is_expanding = pin->GetOrAddBoolean("problem", "is_expanding", false);
  r0  = pin->GetReal("problem", "r_blob");
  user_srcs_nuc_func = &SetReactiveSources;

  if (restart) return;

  // values inside the bubble:
  Real d_in    = pin->GetReal("problem", "d_in");
  Real p_in    = pin->GetReal("problem", "e_in");
  // values outside the bubble;
  Real d_out   = pin->GetReal("problem", "d_out");
  Real p_out   = pin->GetReal("problem", "e_out");

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int nmb = pmbp->nmb_thispack;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;


  // initialize Hydro variables ------------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    int nfluid = pmbp->phydro->nmhd;
    int nscalars = pmbp->phydro->nscalars;
    auto &w0_ = pmbp->phydro->w0;
    Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

    par_for("pgen_blast",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

      Real den;
      Real wv_x = 0.0;
      Real wv_y = 0.0;
      Real wv_z = 0.0;
      Real pres;

      if (rad < r_blob) {
        den = d_in;
        pres = p_in;
      } else {
        den = d_out;
        pres = e_out;
      }

      w0_(m,IDN,k,j,i) = den;
      w0_(m,IVX,k,j,i) = wv_x;
      w0_(m,IVY,k,j,i) = wv_y;
      w0_(m,IVZ,k,j,i) = wv_z;
      w0_(m,IEN,k,j,i) = pres/gm1;

      // We need to define the composition
      w0_(m,iye,k,j,i) = 0.0;
    });

    // Convert primitives to conserved
    pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);
  } 
//--------------- End initialization MHD variables----------
}

namespace {
//----------------------------------------------------------------------------------------
  void SetADMVariablesToExpanding(MeshBlockPack *pmbp) {
    const Real t = pmbp->pmesh->time;
    auto &adm = pmbp->padm->adm;
    auto &size = pmbp->pmb->mb_size;
    auto &indcs = pmbp->pmesh->mb_indcs;
    int nmb = pmbp->nmb_thispack;
    int &ng = indcs.ng;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
    int n1 = (indcs.nx1 > 1) ? (indcs.nx1 + 2*ng) : 1;
    int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

    Real a = 1.0;
    Real b = 0.0;

    par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      adm.g_dd(m,0,0,k,j,i) = a;
      adm.g_dd(m,0,1,k,j,i) = 0.0;
      adm.g_dd(m,0,2,k,j,i) = 0.0;
      adm.g_dd(m,1,1,k,j,i) = a;
      adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.g_dd(m,2,2,k,j,i) = a;

      adm.vK_dd(m,0,0,k,j,i) = 0.0;
      adm.vK_dd(m,0,1,k,j,i) = 0.0;
      adm.vK_dd(m,0,2,k,j,i) = 0.0;
      adm.vK_dd(m,1,1,k,j,i) = 0.0;
      adm.vK_dd(m,1,2,k,j,i) = 0.0;
      adm.vK_dd(m,2,2,k,j,i) = 0.0;

      adm.alpha(m,k,j,i) = a;
      adm.beta_u(m,0,k,j,i) = b*x1v;
      adm.beta_u(m,1,k,j,i) = b*x2v;
      adm.beta_u(m,2,k,j,i) = b*x3v;
    });
  }

  void SetReactiveSources(Mesh* pm, const Real bdt) {
    //Under this wrapper we want to integrate the source terms for the nuclear reactions.

    // We should: 1. Evaluate ye_eq in terms of the density and the EOS, 2. Integrate the reactive source term:
    // dy_e/dt = (ye_eq - ye)/tau, and 3. update the energy/pressure accordingly.   

    return;
  }

} // namespace
