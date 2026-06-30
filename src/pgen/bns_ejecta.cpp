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
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "units/units.hpp"

KOKKOS_INLINE_FUNCTION
Real Lengthfraction(Real a1,Real a2,Real b, Real c, Real R) {
  Real epsilon_frac = 0.0;
  if (R*R - b*b - c*c > 0) {
    Real s1 = (-a1 + sqrt(R*R - b*b - c*c))/(a2-a1);
    Real s2 = (-a1 - sqrt(R*R - b*b - c*c))/(a2-a1);
    Real smax = Kokkos::max(s1, s2);
    Real smin = Kokkos::min(s1, s2);
    if (smax > 1.0) {
      smax = 1.0;
    }
    if (smin < 0.0) {
      smin = 0.0;
    }
    epsilon_frac = Kokkos::abs(smax - smin);
  } else {
    epsilon_frac = 0.0;
  }
  return epsilon_frac;
}

KOKKOS_INLINE_FUNCTION
Real CalcVolFraction(Real x1min, Real x1max, Real x2min, Real x2max, Real x3min, Real x3max,
                      Real R) {
  // This is a helper function to compute the volume fraction of a cell that is filled with ejecta,
  // at the boundary between the ejecta and the ISM. We assume that the ejecta is a sphere of radius R, and that
  // the cell is a rectangular box defined by the limits x1min, x1max, x2min, x2max, x3min, x3max. 
  // We will assume that the center of the ejecta is at the origin, so the sphere is defined by the equation 
  // x1^2 + x2^2 + x3^2 = R^2. We will compute the volume fraction of the cell that is inside the sphere.
  Real epsilon_frac = 0.0;

  epsilon_frac += Lengthfraction(x2min, x2max, x1min, x3min, R);
  epsilon_frac += Lengthfraction(x1min, x1max, x2min, x3min, R);
  epsilon_frac += Lengthfraction(x2min, x2max, x1max, x3min, R);
  epsilon_frac += Lengthfraction(x1min, x1max, x2max, x3min, R);

  epsilon_frac += Lengthfraction(x2min, x2max, x1min, x3max, R);
  epsilon_frac += Lengthfraction(x1min, x1max, x2min, x3max, R);
  epsilon_frac += Lengthfraction(x2min, x2max, x1max, x3max, R);
  epsilon_frac += Lengthfraction(x1min, x1max, x2max, x3max, R);

  epsilon_frac += Lengthfraction(x3min, x3max, x1min, x2min, R);
  epsilon_frac += Lengthfraction(x3min, x3max, x1max, x2min, R);
  epsilon_frac += Lengthfraction(x3min, x3max, x1max, x2max, R);
  epsilon_frac += Lengthfraction(x3min, x3max, x1min, x2max, R);

  epsilon_frac /= 12.0;
  return epsilon_frac;
}

namespace {
  Real h0;
  Real t_eng;
  Real v_r;
  Real v_phi;
  Real Gamma_inf;
  Real r0;
  Real theta_j;
  Real Lj;
  Real sigma_r;
  Real sigma_phi;
  Real ratio;
  Real t_delay;
  void SetADMVariablesToFLRW(MeshBlockPack *pmbp);
  void SetCentralEngine(Mesh* pm, const Real bdt);
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool is_expanding = pin->GetOrAddBoolean("problem", "flrw", false);
  
  // Maximum velocity of the expansion
  Real vmax = pin->GetReal("problem", "vmax");

  // Maximum radius of the expansion
  Real rmax = pin->GetReal("problem", "rmax");

  if (is_expanding) {  
    h0 = vmax/rmax;
    if (pmbp->padm != nullptr) {
      pmbp->padm->SetADMVariables = &SetADMVariablesToFLRW;
    }
  } else {
    h0 = 0.0;
  }

  // Central Engine Radius:
  r0        = pin->GetReal("problem", "r0");
  theta_j   = pin->GetReal("problem", "theta_j");
  Lj        = pin->GetReal("problem", "Lj");
  v_r       = pin->GetReal("problem", "v_r");
  v_phi     = pin->GetReal("problem", "v_phi");
  Gamma_inf = pin->GetReal("problem", "Gamma_inf");
  sigma_r   = pin->GetReal("problem", "sigma_r");
  sigma_phi = pin->GetReal("problem", "sigma_phi");
  ratio     = pin->GetReal("problem", "ratio");
  t_eng     = pin->GetReal("problem", "t_eng");
  t_delay   = pin->GetReal("problem", "t_delay");

  user_srcs_func = &SetCentralEngine;

  if (restart) return;

  // Ejecta parameters:

  Real r_ejecta = pin->GetOrAddReal("problem", "r_ejecta", 1.0);
  Real v_ejecta = pin->GetOrAddReal("problem", "v_ejecta", 1.0);
  Real d_ejecta = pin->GetOrAddReal("problem", "d_ejecta", 1.0);
  Real d_ism    = pin->GetOrAddReal("problem", "d_ism", 1.0);
  Real m_ejecta = pin->GetOrAddReal("problem", "m_ejecta", 1.0);
  Real temp     = pin->GetOrAddReal("problem", "temp", 1.0);
 
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
  Real tau_ism;
  // Select the tail parameters based on the choice of tail type
  if (power_law) {
    n_ism = pin->GetOrAddReal("problem", "n_ism", 3.0);
  } else if (exponential) {
    tau_ism = pin->GetOrAddReal("problem", "tau_ism", 1.0);
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
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gamma = eos.gamma;
    Real gm1 = gamma - 1.0;
    if (pmbp->pcoord->is_dynamical_relativistic) {
      gm1 = 1.0; // DynGRMHD uses pressure, not energy.
    }
    // We will consider: c=1, [M] = g, [L]=km.
    const Real m_ejecta_1 = m_ejecta;
    const Real r_ejecta_1 = r_ejecta;
    const Real r0_1 = r0;
    auto& w0_ = pmbp->pmhd->w0;

    par_for("pgen_blast1",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1l = LeftEdgeX(i-is, nx1, x1min, x1max);
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1r = LeftEdgeX(i-is+1, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2l = LeftEdgeX(j-js, nx2, x2min, x2max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2r = LeftEdgeX(j-js+1, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3l = LeftEdgeX(k-ks, nx3, x3min, x3max);
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3r = LeftEdgeX(k-ks+1, nx3, x3min, x3max);

      Real rad_l = sqrt(SQR(x1l) + SQR(x2l) + SQR(x3l));
      Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      Real rad_r = sqrt(SQR(x1r) + SQR(x2r) + SQR(x3r));

      Real vel_x;
      Real vel_y;
      Real vel_z;
      Real Gamma0;

      if (rad < r_ejecta_1) {
        vel_x = v_ejecta * x1v / r_ejecta_1;
        vel_y = v_ejecta * x2v / r_ejecta_1;
        vel_z = v_ejecta * x3v / r_ejecta_1;
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
      Real rho_0 = m_ejecta_1/(4*M_PI*r0_1*r0_1);
      rho_0 *= 1.0/(r_ejecta_1*(1.0 - r0_1/r_ejecta_1)); 
      
      if (rad <= r0_1) {
        den = rho_0;
      } else if (rad > r0_1 && rad < r_ejecta_1) {
        den = rho_0 * SQR(r0_1/rad);
      } else {
        if (power_law) {
          Real log_k1  = log(d_ism) + n_ism*log(r_ejecta_1);
          Real log_rho = log_k1 - n_ism*log(rad);
          den = exp(log_rho);
        } else if (exponential) {
          Real log_rho = log(d_ism) - (rad-r_ejecta_1)/tau_ism;
          den = exp(log_rho);
        } else if (constant) {
          den = d_ism;
        }
      }

      // Reuse the Immerse BC fractional volume to define a more accurate boundary values.
      // if (rad_l < r0_1 && rad_r>r0_1) {
      //   Real epsilon_frac = CalcVolFraction(x1l, x1r, x2l, x2r, x3l, x3r, r0_1);
      //   den = rho_0 * epsilon_frac + rho_0 * SQR(r0_1/rad_r) * (1.0 - epsilon_frac);
      // } else if (rad_l < r_ejecta_1 && rad_r>r_ejecta_1) {
      //   Real epsilon_frac = CalcVolFraction(x1l, x1r, x2l, x2r, x3l, x3r, r_ejecta_1);
      //   den = rho_0 * SQR(r0_1/rad) * epsilon_frac + d_ism * (1.0 - epsilon_frac);
      // }

      Real pres = temp*den; 

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
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int nmb = pmbp->nmb_thispack;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

    // Set FLRW metric variables for the current mesh time.
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

  void SetCentralEngine(Mesh* pm, const Real beta_dt) {
    // This is where we would set the source terms for the central engine, if we wanted to.
    MeshBlockPack *pmbp = pm->pmb_pack;
    const Real t = pmbp->pmesh->time;
    
    if (t > t_eng || t < t_delay) return;
    Real tau = pmbp->pmesh->dt;
    
    auto &indcs = pmbp->pmesh->mb_indcs;
    int is = indcs.is;
    int js = indcs.js;
    int ks = indcs.ks;
    int ie = indcs.ie;
    int je = indcs.je;
    int ke = indcs.ke;
    int nmb1 = pmbp->nmb_thispack - 1;
    auto &size = pmbp->pmb->mb_size;
    auto &adm = pmbp->padm->adm;
    

    if (pmbp->pmhd != nullptr) {
      EOS_Data &eos = pmbp->pmhd->peos->eos_data;
      Real gamma = eos.gamma;
      DvceArray5D<Real> &u0 = pmbp->pmhd->u0;

      const Real Lj_1 = Lj;
      const Real v_r_1 = v_r;
      const Real v_phi_1 = v_phi;
      const Real Gamma_inf_1 = Gamma_inf;
      const Real r0_1 = r0;
      const Real theta_j_1 = theta_j;
      const Real sigma_r_1 = sigma_r;
      const Real sigma_phi_1 = sigma_phi;
      const Real ratio_1 = ratio;
    
      par_for("central_engine", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1_l = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
        Real x1_r = LeftEdgeX(i-is+1, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2_l = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        Real x2_r = LeftEdgeX(j-js+1, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3_l = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        Real x3_r = LeftEdgeX(k-ks+1, indcs.nx3, x3min, x3max);

        Real rad_l = sqrt(SQR(x1_l) + SQR(x2_l) + SQR(x3_l));
        Real rad_r = sqrt(SQR(x1_r) + SQR(x2_r) + SQR(x3_r));
        Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

        const Real& alpha = adm.alpha(m, k, j, i);

        Real theta = acos(x3v/rad);
        Real theta_min = M_PI - theta_j_1;
        if ((rad>0 && rad <= r0_1/alpha) && ((theta < theta_j_1) || (theta > theta_min))) {

          Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                 adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                 adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};

          Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                      g3d[S22], g3d[S23], g3d[S33]);
          Real vol = sqrt(detg);

          Real den;
          Real wvx;
          Real wvy;
          Real wvz;
          Real wv_x;
          Real wv_y;
          Real wv_z;
          Real pres;

          Real Gamma = 1.0 / sqrt(1.0 - (SQR(v_r_1) + SQR(v_phi_1)));
          Real Gamma_r = 1.0 / sqrt(1.0 - SQR(v_r_1));
          Real h = Gamma_inf_1/Gamma_r; 
          Real epsilon_frac = 1.0;

          if (rad_l < r0_1/alpha && rad_r > r0_1/alpha) {
            epsilon_frac = CalcVolFraction(x1_l, x1_r, x2_l, x2_r, x3_l, x3_r, r0_1/alpha);
          }

          den = Lj_1/(4*M_PI*SQR(r0_1)*v_r_1*SQR(Gamma_r)*h);

          if (r_cil == 0) {
            wvx = 0.0;
            wvy = 0.0;
            wvz = Gamma * (v_r_1*x3v/rad)/alpha;
          } else {
            Real x;
            if (theta > theta_min){
              x = M_PI - theta;
            } else {
              x = theta;
            }
            wvx = Gamma * (v_r_1*x1v/rad - v_phi_1*(x/theta_j_1)*x2v/r_cil)/alpha;
            wvy = Gamma * (v_r_1*x2v/rad + v_phi_1*(x/theta_j_1)*x1v/r_cil)/alpha;
            wvz = Gamma * (v_r_1*x3v/rad)/alpha;
          }

          // We need to extract w in the expanding cordinates instead of using Gamma_l, because we
          // converted wv from Lorentz to the expanding coordinates, but not w alone.
          Real v[3] = {wvx, wvy, wvz};
          Real v2 = Primitive::SquareVector(v, g3d);
          Real w = sqrt(1.0 + v2); 

          wv_x = g3d[S11]*wvx + g3d[S12]*wvy + g3d[S13]*wvz;
          wv_y = g3d[S12]*wvx + g3d[S22]*wvy + g3d[S23]*wvz;
          wv_z = g3d[S13]*wvx + g3d[S23]*wvy + g3d[S33]*wvz;

          // Magnetic Field Prescription
          Real eta = 1/(1+0.5*(sigma_r_1+sigma_phi_1))*(h+0.5*(sigma_r_1 + sigma_phi_1)); //express h in terms of h*
          Real pres_avg = (gamma - 1.0)/gamma * (eta - 1.0) * den;
          Real br = sqrt(2.0*sigma_r_1*pres_avg);
          Real factor = -SQR(ratio_1)/(1 + SQR(ratio_1)) + ratio_1*atan(1/ratio_1);
          Real factor_inv = 1.0/factor;
          Real bphi = Gamma_r * sqrt(pres_avg*sigma_phi_1*factor_inv);  //Check that pavg is correct, note that h-> thermal, not hstar
          
          // Adding the magnetic field. This will be used to compute the conserved 
          // variables contributions of the prescribed magnetic fields.
          Real bx;
          Real by;
          Real bz;
          if (r_cil == 0) {
            bx = 0.0;
            by = 0.0;
            bz = (br*x3v/rad)/SQR(alpha);
          } else {
            Real theta_m = ratio_1 * theta_j_1;
            Real x;
            if (theta > theta_min){
              x = M_PI - theta;
            } else {
              x = theta;
            }
            bx = (br*x1v/rad - bphi*(x/theta_m)*x2v/r_cil)/SQR(alpha);
            by = (br*x2v/rad + bphi*(x/theta_m)*x1v/r_cil)/SQR(alpha);
            bz = (br*x3v/rad)/SQR(alpha);
          }

          //Lowering the magnetic field:
          Real b_x = g3d[S11]*bx + g3d[S12]*by + g3d[S13]*bz;
          Real b_y = g3d[S12]*bx + g3d[S22]*by + g3d[S23]*bz;
          Real b_z = g3d[S13]*bx + g3d[S23]*by + g3d[S33]*bz;

          // We will use the Lorentz values of br and bphi to compute the pressure.
          // Since the pressure is an scalar, and no further corrections are needed.
          Real x;
          if (theta > theta_min){
            x = M_PI - theta;
          } else {
            x = theta;
          }
          Real xm = ratio_1 * theta_j_1;
          Real p[8];
          p[0] = (-2.0*SQR(bphi)*SQR(x))/(SQR(Gamma_r)*(SQR(xm) + SQR(x)));
          p[1] = (-2.0*SQR(bphi)*SQR(xm)*SQR(x))/(SQR(Gamma_r)*SQR(SQR(xm) + SQR(x)));
          p[2] = (bphi*br*v_phi_1*v_r_1*xm*(-2.0*SQR(x) - (SQR(xm) + SQR(x))*log(SQR(xm)
                /(SQR(xm) + SQR(x)))))/(theta_j_1*(SQR(xm) + SQR(x)));
          p[3] = (SQR(br)*SQR(v_phi_1)*SQR(x))/(2.0*SQR(theta_j_1));
          p[4] = -((bphi*br*v_phi_1*v_r_1*xm*log(1 + SQR(x)/SQR(xm)))/theta_j_1);
          p[5] = (den*SQR(Gamma)*h*SQR(v_phi_1)*SQR(x))/(2.0*SQR(theta_j_1));
          p[6] = (SQR(br)*SQR(v_phi_1)*SQR(x))/(2.0*SQR(theta_j_1));
          p[7] = (-2.0*bphi*br*v_phi_1*v_r_1*xm*log(1.0 + SQR(x)/SQR(xm)))/theta_j_1;

          pres = std::accumulate(std::begin(p), std::end(p), pres_avg);
    
          Real u0_den = u0(m,IDN,k,j,i);
          Real u0_mom1 = u0(m,IM1,k,j,i);
          Real u0_mom2 = u0(m,IM2,k,j,i);
          Real u0_mom3 = u0(m,IM3,k,j,i);
          Real u0_tau = u0(m,IEN,k,j,i);

          Real u0_den_eq = vol*den*w;
          Real b2 = b_x * bx + b_y * by + b_z * bz;
          Real u0_mom1_eq = vol*(den*h*w*wv_x + b2*wv_x/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_x);
          Real u0_mom2_eq = vol*(den*h*w*wv_y + b2*wv_y/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_y);
          Real u0_mom3_eq = vol*(den*h*w*wv_z + b2*wv_z/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_z);
          Real u0_tau_eq = vol*(den*h*w*w + b2 - pres - 0.5*(SQR(bx*wv_x/w + by*wv_y/w + bz*wv_z/w) + b2/(w*w)) - den*w);

          u0(m,IDN,k,j,i) += -alpha*vol*beta_dt*(u0_den - u0_den_eq)/tau * epsilon_frac;
          u0(m,IM1,k,j,i) += -alpha*vol*beta_dt*(u0_mom1 - u0_mom1_eq)/tau * epsilon_frac;
          u0(m,IM2,k,j,i) += -alpha*vol*beta_dt*(u0_mom2 - u0_mom2_eq)/tau * epsilon_frac;
          u0(m,IM3,k,j,i) += -alpha*vol*beta_dt*(u0_mom3 - u0_mom3_eq)/tau * epsilon_frac;
          u0(m,IEN,k,j,i) += -alpha*vol*beta_dt*(u0_tau - u0_tau_eq)/tau * epsilon_frac;
        }
      });
    }
    return;
  }
}