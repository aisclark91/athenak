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

#include <math.h>

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
#include "units/units.hpp"

namespace {
  Real h0;
  void SetADMVariablesToFLRW(MeshBlockPack *pmbp);
}


//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool is_expanding = pin->GetOrAddBoolean("problem", "flrw", false);
  
  // Maximum velocity of the ejecta
  Real vmax = pin->GetReal("problem", "vmax");

  // This r_{m,0}
  Real rmax = pin->GetReal("problem", "rmax");

  // Expanding offset
  Real offset = pin->GetOrAddReal("problem", "offset", 0.0);

  if (is_expanding) {  
    h0 = vmax/rmax + offset;
    pmbp->padm->SetADMVariables = &SetADMVariablesToFLRW;
  }

  if (restart) return;

  // Central Engine Radius
  Real r0 = pin->GetReal("problem", "r0");


  // values for neutrals (hydro fluid)
  Real d_ejecta   = pin->GetOrAddReal("problem", "d_ejecta", 1.0);
  Real d_ism   = pin->GetOrAddReal("problem", "d_ism", 1.0);
  Real m_ejecta   = pin->GetOrAddReal("problem", "m_ejecta", 1.0);
  Real k_eff   = pin->GetOrAddReal("problem", "k_eff", 1.0);

  // whether to use a power-law, exponential-law or constant density tail
  std::string ism_dep = pin->GetOrAddString("problem", "ism_dep", "constant");
  bool power_law = (ism_dep.compare("power_law") == 0);
  bool exponential = (ism_dep.compare("exponential") == 0);
  bool constant = (ism_dep.compare("constant") == 0);
  if (!power_law && !exponential && !constant) {
    std::cout << "Defaulting to constant" << std::endl;
    constant = true;
  }

  Real n_ism;
  Real tau;
  // Select the tail parameters based on the choice of tail type
  if (power_law) {
    n_ism = pin->GetOrAddReal("problem", "n_ism", 3.0);
  } else if (exponential) {
    tau = pin->GetOrAddReal("problem", "tau", 1.0);
  }

  // magnetic field strenght
  Real b_amb = pin->GetOrAddReal("problem", "b_amb", 0.1);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // initialize MHD variables ------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    auto &w0_ = pmbp->pmhd->w0;
    Real gamma = pmbp->pmhd->peos->eos_data.gamma;
    Real gm1 = gamma - 1.0;
    if (pmbp->pcoord->is_dynamical_relativistic) {
      gm1 = 1.0; // DynGRMHD uses pressure, not energy.
    }

    // We will consider gaussian units: c=1, [M] = g, [L]=cm.

    par_for("pgen_blast1",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      Real vel_x;
      Real vel_y;
      Real vel_z;
      Real Gamma0;

      if (rad < rmax) {
        vel_x = vmax * x1v / rmax;
        vel_y = vmax * x2v / rmax;
        vel_z = vmax * x3v / rmax;
        Gamma0 = 1.0 / sqrt(1.0 - SQR(vel_x) - SQR(vel_y) - SQR(vel_z));
        vel_x *= Gamma0;
        vel_y *= Gamma0;
        vel_z *= Gamma0;
      } else {
        vel_x = 0.0;
        vel_y = 0.0;
        vel_z = 0.0; 
      } 

      Real den;
      Real rho_0 = m_ejecta/(4*M_PI*r0*r0);
      rho_0 *= (vmax / rmax) / (asin(vmax) - asin(vmax*r0/rmax)); 
      
      if (rad < r0) {
        den = rho_0;
      } else if (rad >= r0 && rad < rmax) {
        den = rho_0 * (r0*r0)/(rad*rad);
      } else {
        if (power_law) {
          Real log_k1  = log(d_ism) + n_ism*log(rmax);
          Real log_rho = log_k1 - n_ism*log(rad);
          den = exp(log_rho);
        } else if (exponential) {
          Real log_rho = log(d_ism) - (rad-rmax)/tau;
          den = exp(log_rho);
        } else if (exponential) {
          Real log_rho = log(d_ism) - (rad-rmax)/tau;
          den = exp(log_rho);
        } else if (constant) {
          den = d_ism;
        }
      }

      Real pres = k_eff * pow(den, gamma); 

      w0_(m,IDN,k,j,i) = den;
      w0_(m,IVX,k,j,i) = vel_x;
      w0_(m,IVY,k,j,i) = vel_y;
      w0_(m,IVZ,k,j,i) = vel_z;
      w0_(m,IEN,k,j,i) = pres/gm1;
    });

    // initialize magnetic fields
    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb = pmbp->nmb_thispack;
    DvceArray4D<Real> a3;
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;

    int ku = ke;
    if (ncells3 > 1) {
      ku = ke + 1;
    }

    par_for("pgen_potential", DevExeSpace(), 0,nmb-1,ks,ku,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f = LeftEdgeX(i-is,nx1,x1min,x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f = LeftEdgeX(j-js,nx2,x2min,x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f = LeftEdgeX(k-ks,nx3,x3min,x3max);

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      Real y = x2f;

      a3(m,k,j,i) = b_amb*y;
    });


    // initialize magnetic fields
    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_blast2",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;

      b0.x1f(m,k,j,i) = (a3(m,k,j+1,i) - a3(m,k,j,i))/dx2;
      b0.x2f(m,k,j,i) = -(a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = (a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2;
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = -(a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1;
      }
      if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_blast3",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });

    // Convert primitives to conserved
    if (!pmbp->pcoord->is_dynamical_relativistic) {
      pmbp->pmhd->peos->PrimToCons(w0_, bcc_, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
    }
  }  // End initialization MHD variables

  // Initialize ADM variables -----------------------------------------
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
  void SetADMVariablesToFLRW(MeshBlockPack *pmbp) {
    const Real t = pmbp->pmesh->time;
    auto &adm = pmbp->padm->adm;
    auto &size = pmbp->pmb->mb_size;
    auto &indcs = pmbp->pmesh->mb_indcs;
    int &ng = indcs.ng;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
    int nmb = pmbp->nmb_thispack;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

    // We want to set the Minkowski space before t_0 and FLRW after. 
    Real a;
    Real b;

    a = exp(h0*t);
    b = h0;

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

      adm.g_dd(m,0,0,k,j,i) = a*a;
      adm.g_dd(m,0,1,k,j,i) = 0.0;
      adm.g_dd(m,0,2,k,j,i) = 0.0;
      adm.g_dd(m,1,1,k,j,i) = a*a;
      adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.g_dd(m,2,2,k,j,i) = a*a;

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
} 
