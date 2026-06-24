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
  t_eng     = pin->GetReal("problem", "t_eng");

  user_srcs_func = &SetCentralEngine;

  if (restart) return;

  // Ejecta parameters:

  Real r_ejecta   = pin->GetOrAddReal("problem", "r_ejecta", 1.0);
  Real v_ejecta   = pin->GetOrAddReal("problem", "v_ejecta", 1.0);
  Real d_ejecta   = pin->GetOrAddReal("problem", "d_ejecta", 1.0);
  Real d_ism      = pin->GetOrAddReal("problem", "d_ism", 1.0);
  Real m_ejecta   = pin->GetOrAddReal("problem", "m_ejecta", 1.0);
  Real temp      = pin->GetOrAddReal("problem", "temp", 1.0);
 
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
      
      if (rad < r0_1) {
        den = rho_0;
      } else if (rad >= r0_1 && rad < r_ejecta_1) {
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
    Real tau = pmbp->pmesh->dt;

    if (t > t_eng) return;
    
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
    
      par_for("central_engine", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real rad_l = sqrt(SQR(x1min) + SQR(x2min) + SQR(x3min));
        Real rad_r = sqrt(SQR(x1max) + SQR(x2max) + SQR(x3max));

        Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

        Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                              adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                              adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};

        const Real& alpha = adm.alpha(m, k, j, i);

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

        Real Gamma_l = 1.0 / sqrt(1.0 - (SQR(v_r_1) + SQR(v_phi_1)));
        Real h = Gamma_inf_1/Gamma_l;
        
        if (rad > 0 && rad <= r0_1/alpha) {
          Real epsilon_frac = 1.0;
          if (rad_l < r0_1/alpha && rad_r > r0_1/alpha) {
            epsilon_frac = CalcVolFraction(x1min, x1max, x2min, x2max, x3min, x3max, r0_1);
          }

          Real theta = acos(x3v/rad);
          Real theta_min = M_PI - theta_j_1;
          if ((theta < theta_j_1) || (theta > theta_min)) {
            den = Lj_1/(4*M_PI*SQR(r0_1)*v_r_1*SQR(Gamma_l)*h);
            if (r_cil == 0) {
              wvx = 0.0;
              wvy = 0.0;
              wvz = Gamma_l * (v_r_1*x3v/rad)/alpha;
            } else {
              wvx = Gamma_l * (v_r_1*x1v/rad - v_phi_1*x2v/r_cil)/alpha;
              wvy = Gamma_l * (v_r_1*x2v/rad + v_phi_1*x1v/r_cil)/alpha;
              wvz = Gamma_l * (v_r_1*x3v/rad)/alpha;
            }

            // We need to extract Gamma_exp instead of setting Gamma_l, because the
            // metric is not Minkwoskian.
            Real v[3] = {wvx, wvy, wvz};
            Real v2 = Primitive::SquareVector(v, g3d);
            Real w = sqrt(1.0 + v2); 

            wv_x = g3d[S11]*wvx + g3d[S12]*wvy + g3d[S13]*wvz;
            wv_y = g3d[S12]*wvx + g3d[S22]*wvy + g3d[S23]*wvz;
            wv_z = g3d[S13]*wvx + g3d[S23]*wvy + g3d[S33]*wvz;
          
            pres = (gamma - 1.0)/gamma * (h - 1.0) * den;

            Real u0_den = u0(m,IDN,k,j,i);
            Real u0_mom1 = u0(m,IM1,k,j,i);
            Real u0_mom2 = u0(m,IM2,k,j,i);
            Real u0_mom3 = u0(m,IM3,k,j,i);
            Real u0_tau = u0(m,IEN,k,j,i);

            u0(m,IDN,k,j,i) += -alpha*vol*beta_dt*(u0_den - vol*den*w)/tau * epsilon_frac;
            u0(m,IM1,k,j,i) += -alpha*vol*beta_dt*(u0_mom1 - vol*den*h*w*wv_x)/tau * epsilon_frac;
            u0(m,IM2,k,j,i) += -alpha*vol*beta_dt*(u0_mom2 - vol*den*h*w*wv_y)/tau * epsilon_frac;
            u0(m,IM3,k,j,i) += -alpha*vol*beta_dt*(u0_mom3 - vol*den*h*w*wv_z)/tau * epsilon_frac;
            u0(m,IEN,k,j,i) += -alpha*vol*beta_dt*(u0_tau - vol*(den*h*w*w - pres - den*w))/tau * epsilon_frac;
          }
        }
      });
    }
    return;
  }
}