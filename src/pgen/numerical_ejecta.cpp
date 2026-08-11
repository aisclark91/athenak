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
#include <vector>
#include <cstdlib>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "units/units.hpp"
#include "inputs/numerical_ejecta.hpp"

constexpr int kNTheta = 51;
constexpr int kNTime  = 4994;

namespace {
  // Expansion:
  Real h0;
  bool is_expanding;
  Real epsilon_h0;
  Real t_exp_max;

  // Central Engine Variables:
  // Real t_eng;
  // Real v_r;
  // Real v_phi;
  
  // Real Gamma_inf;
  Real r0_ejecta;
  // Real theta_j;
  // Real Lj;
  // Real sigma_r;
  // Real sigma_phi;
  // Real ratio;
  // Real t_delay; 

  //Numerical Ejecta:
  Real t_num;
  std::vector<Block> numerical_data;

  //Functors:
  void SetADMVariablesToFLRW(MeshBlockPack *pmbp);
  void SetNumericalEjecta(Mesh* pm, const Real bdt);
   // void SetCentralEngine(Mesh* pm, const Real bdt);
}


KOKKOS_INLINE_FUNCTION
Real Interpolate1D(Real &var_i, Real &var_i1, Real &x_i, Real &x_i1, Real &x) {
  Real m = (var_i1 - var_i)/(x_i1 - x_i);
  return m*(x - x_i) + var_i;
}


KOKKOS_INLINE_FUNCTION
int FindTimeIndex(int ith, const DualArray2D<Real> &time, Real t) {
  int size = time.view_device().extent(1);
  int index = 0;

  if(t <= time.d_view(ith,0)) {
    index = 0;
  }

  for(int j=1; j<size; j++) {
    if(t>time.d_view(ith,j-1) && t<=time.d_view(ith, j)) {
      index = j-1;
      break;
    }
  }

  if(t > time.d_view(ith, size-1)){
    index = size-2;
  }

  return index;
}


KOKKOS_INLINE_FUNCTION
int FindThetaIndex(const DualArray1D<Real> &theta, Real th) {

  int size = theta.view_device().extent(0);
  int index = 0;

  if (th <= theta.d_view(0)) {
    index = 0;
  }

  for(int i=1; i<size; i++) {
    if (th > theta.d_view(i-1) && th <= theta.d_view(i)) {
      index = i-1;
      break;
    }
  }

  if (th > theta.d_view(size-1)) {
    index = size-2;
  }

  return index;
}


KOKKOS_INLINE_FUNCTION
Real Interpolate2D(const DualArray1D<Real> &theta, const DualArray2D<Real> &time, 
  const DualArray2D<Real>& var_ej, Real index, Real th, Real t) {

  Real var;
  int size = time.view_device().extent(1);  
  int ith = FindThetaIndex(theta, th);
  int jt  = FindTimeIndex(ith, time, t);
  int kt  = FindTimeIndex(ith+1, time, t);
  
  Real jtmax = time.d_view(ith, size-1);
  Real ktmax = time.d_view(ith+1, size-1);

  Real th_i = theta.d_view(ith);
  Real tm_j = time.d_view(ith, jt);
  Real tm_j1 = time.d_view(ith, jt+1);
  Real var_i;

  if(t <= tm_j) {
    Real var_ij = var_ej.d_view(ith, jt);
    var_i = var_ij;
  } else if(t > tm_j && t < jtmax) {
    Real var_ij = var_ej.d_view(ith, jt);
    Real var_ij1 = var_ej.d_view(ith,jt+1);
    var_i = Interpolate1D(var_ij, var_ij1, tm_j, tm_j1, t);
  } else {
    Real var_ij1 = var_ej.d_view(ith,jt+1);
    var_i = var_ij1 * Kokkos::pow(t/jtmax, index);
  }

  Real th_i1 = theta.d_view(ith+1);
  Real tm_k = time.d_view(ith+1, kt);
  Real tm_k1 = time.d_view(ith+1, kt+1);
  Real var_i1;

  if(t <= tm_k) {
    Real var_ik = var_ej.d_view(ith+1, kt);
    var_i1 = var_ik;
  } else if(t > tm_k && t < ktmax) {
    Real var_ik = var_ej.d_view(ith+1, kt);
    Real var_ik1 = var_ej.d_view(ith+1,kt+1);
    var_i1 = Interpolate1D(var_ik, var_ik1, tm_k, tm_k1, t);
  } else {
    Real var_ik1 = var_ej.d_view(ith+1,kt+1);
    var_i1 = var_ik1 * Kokkos::pow(t/ktmax, index);
  }

  var = Interpolate1D(var_i, var_i1, th_i, th_i1, th);
  return var;
}


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


KOKKOS_INLINE_FUNCTION
Real TransitionEpsilon(Real dr, Real r0, Real rad) {
  // Instead of using a geometrical epsilon factor, we may use an smooth
  // activation function at r=r0 of a few dr in size with compact support
  Real var;
  if(rad <=(r0-dr)) {
    var = 1.0;
  } else if (rad>(r0-dr) && rad<(r0+dr)) {
    var = 0.5*(1.0 - Kokkos::sin(M_PI*(rad - r0)/(2.0*dr)));
  } else {
    var = 0.0;
  }
  return var;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  is_expanding = pin->GetOrAddBoolean("problem", "is_expanding", true);

  t_exp_max = pin->GetReal("problem", "t_exp_max");
  
  // Maximum velocity of the expansion
  Real vmax = pin->GetReal("problem", "vmax");

  // Maximum radius of the expansion
  Real rmax = pin->GetReal("problem", "rmax");

  epsilon_h0 = pin->GetReal("problem", "epsilon_h0");

  if (is_expanding) {  
    h0 = vmax/rmax;
    if (pmbp->padm != nullptr) {
      pmbp->padm->SetADMVariables = &SetADMVariablesToFLRW;
    }
  } else {
    h0 = 0.0;
  }

  // Central Engine Radius:
  r0_ejecta = pin->GetReal("problem", "r0");
  // theta_j   = pin->GetReal("problem", "theta_j");
  // Lj        = pin->GetReal("problem", "Lj");
  // v_r       = pin->GetReal("problem", "v_r");
  // v_phi     = pin->GetReal("problem", "v_phi");
  // Gamma_inf = pin->GetReal("problem", "Gamma_inf");
  // sigma_r   = pin->GetReal("problem", "sigma_r");
  // sigma_phi = pin->GetReal("problem", "sigma_phi");
  // ratio     = pin->GetReal("problem", "ratio");
  // t_eng     = pin->GetReal("problem", "t_eng");
  // t_delay   = pin->GetReal("problem", "t_delay");

  {
    // Reading:
    t_num = pin->GetReal("problem", "t_num");
    std::string filename = pin->GetString("problem", "file_path");
    NumericalEjectaData model(filename, kNTheta, kNTime);
    numerical_data = model.ComputeBlocks();
  }
 
  // Set an immerse bc that recreate the ejecta.
  user_srcs_func = &SetNumericalEjecta;

  // user_srcs_func = &SetCentralEngine;

  if (restart) return;

  // Whether to use a power-law, exponential-law or constant density tail
  Real d_ism          = pin->GetReal("problem", "d_ism");
  Real temp_ism       = pin->GetReal("problem", "temp_ism");
  std::string ism_dep = pin->GetOrAddString("problem", "ism_dep", "constant");
  bool power_law      = (ism_dep.compare("power_law") == 0);
  bool exponential    = (ism_dep.compare("exponential") == 0);
  bool constant       = (ism_dep.compare("constant") == 0);

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
    const Real r0 = r0_ejecta;
    const std::vector<Block> h_ejecta = numerical_data;
    // We define the primitive variables:
    auto& w0_ = pmbp->pmhd->w0;

    DualArray1D<Real> theta_ej("theta_arr", kNTheta);
    DualArray2D<Real> time_ej("time_arr", kNTheta, kNTime);
    DualArray2D<Real> vinfty_ej("velocity_arr", kNTheta, kNTime);
    DualArray2D<Real> mdot_ej("mdot_arr", kNTheta, kNTime);
    DualArray2D<Real> temp_ej("temperature_arr", kNTheta, kNTime);


    for (int i = 0; i < kNTheta; i++) {
      theta_ej.h_view(i) = h_ejecta[i].th;
    }

    for (int i = 0; i < kNTheta; i++) {
      for (int j = 0; j < kNTime; j++) {
        time_ej.h_view(i,j)   = h_ejecta[i].time[j];
        vinfty_ej.h_view(i,j) = h_ejecta[i].v_infty[j];
        mdot_ej.h_view(i,j)   = h_ejecta[i].mdot[j];
        temp_ej.h_view(i,j)   = h_ejecta[i].temperature[j];
      }
    }

    theta_ej.template modify<HostMemSpace>();
    theta_ej.template sync<DevExeSpace>();

    time_ej.template modify<HostMemSpace>();
    time_ej.template sync<DevExeSpace>();

    vinfty_ej.template modify<HostMemSpace>();
    vinfty_ej.template sync<DevExeSpace>();

    mdot_ej.template modify<HostMemSpace>();
    mdot_ej.template sync<DevExeSpace>();

    temp_ej.template modify<HostMemSpace>();
    temp_ej.template sync<DevExeSpace>();

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

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      Real dr = sqrt(SQR(dx1)+SQR(dx2)+SQR(dx3));

      Real rad_l = sqrt(SQR(x1l) + SQR(x2l) + SQR(x3l));
      Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      Real rad_r = sqrt(SQR(x1r) + SQR(x2r) + SQR(x3r));
      Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

      Real den;
      Real veloc;
      Real wvx;
      Real wvy;
      Real wvz;
      Real pres;

      const Real mu = 1.0; // see MNRAS 535, 3711–3731 (2024)
      const Real g_cm3_to_g_km3 = 1.0e15;
      const Real cm_s_1 = 3.335641e-11;
      const Real K_to_1 = 9.251087e-14/mu;
      const Real dyne_to_g_km3 = 1.1126500560536184e-6;

      if(rad <= r0 && rad > 0.0) {
        // We extract the density, veloc and pressure in cgs units.
        Real th = acos(Kokkos::fmin(1.0, Kokkos::fmax(-1.0, x3v/rad)));
        Real mdot = Interpolate2D(theta_ej, time_ej, mdot_ej, -0.5, th, 0.0);
        veloc = Interpolate2D(theta_ej, time_ej, vinfty_ej, -0.1, th, 0.0);
        Real gamma = 1.0/sqrt(1.0 - SQR(veloc/2.99792458e10));
        den = mdot/(4*M_PI*SQR(r0*1.0e5)*gamma*veloc);
        Real temp = Interpolate2D(theta_ej, time_ej, temp_ej, -0.2, th, 0.0);
        pres = 7.56573325e-15*pow(temp, 4);

        // Finally we convert to the right code units
        den *= g_cm3_to_g_km3;
        veloc *= cm_s_1;
        pres *= dyne_to_g_km3;

        if (r_cil == 0) {
          wvx = 0.0;
          wvy = 0.0;
          wvz = veloc * x3v / r0;
          Real Gamma0 = 1.0/sqrt(1.0-SQR(wvx)-SQR(wvy)-SQR(wvz));
          wvx *= Gamma0;
          wvy *= Gamma0;
          wvz *= Gamma0;
        } else {
          wvx = veloc * x1v / r0;
          wvy = veloc * x2v / r0;
          wvz = veloc * x3v / r0;
          Real Gamma0 = 1.0/sqrt(1.0-SQR(wvx)-SQR(wvy)-SQR(wvz));
          wvx *= Gamma0;
          wvy *= Gamma0;
          wvz *= Gamma0;
        }
      } else {
        if (power_law) {
          Real log_k1  = log(d_ism) + n_ism*log(r0);
          Real log_rho = log_k1 - n_ism*log(rad);
          den = exp(log_rho);
        } else if (exponential) {
          Real log_rho = log(d_ism) - (rad-r0)/tau_ism;
          den = exp(log_rho);
        } else if (constant) {
          den = d_ism;
        }
        wvx = 0.0;
        wvy = 0.0;
        wvz = 0.0;
        Real temp = temp_ism/K_to_1; 
        pres = 7.56573325e-15*pow(temp, 4);
        pres *= dyne_to_g_km3;
      } 

      w0_(m,IDN,k,j,i) = den;
      w0_(m,IVX,k,j,i) = wvx;
      w0_(m,IVY,k,j,i) = wvy;
      w0_(m,IVZ,k,j,i) = wvz;
      w0_(m,IEN,k,j,i) = pres/gm1;
    });

    // initialize magnetic fields
    // compute vector potential over all faces
    int ncells1 = (indcs.nx1 > 1)? (indcs.nx1 + 2*(indcs.ng)) : 1;
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

      // For now the magnetic field is zero.
      a3(m,k,j,i) = 0.0;
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
    int n1 = (indcs.nx1 > 1) ? (indcs.nx1 + 2*ng) : 1;
    int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

    // Set expanding metric variables for the current mesh time
    const Real t0 = t_exp_max;
    const Real epsilon = epsilon_h0;

    // We will construct a switch to let the ejecta expand from the immmerse b.c
    // turning off the expansion before t_exp_max.
    const Real b = h0/2.0*(1+tanh((t-t0)/(2.0*epsilon)));
    const Real a = exp(h0*t/2.0)*pow(cosh((t-t0)/(2.0*epsilon))/cosh(t0/(2.0*epsilon)), h0*epsilon);
    // const Real a = 1.0;
    // const Real b = 0.0;

    par_for("update_adm_vars", DevExeSpace(), 0, nmb-1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

  void SetNumericalEjecta(Mesh* pm, const Real beta_dt) {

    MeshBlockPack *pmbp = pm->pmb_pack;
    const Real t_code = pmbp->pmesh->time;
    Real tau = pmbp->pmesh->dt;

    if (t_code > t_num || t_code <= 0.0) return;

    auto &indcs = pmbp->pmesh->mb_indcs;
    int is = indcs.is;
    int js = indcs.js;
    int ks = indcs.ks;
    int ie = indcs.ie;
    int je = indcs.je;
    int ke = indcs.ke;
    int nmb = pmbp->nmb_thispack;
    auto &size = pmbp->pmb->mb_size;
    auto &adm = pmbp->padm->adm;

    if (pmbp->pmhd != nullptr) {
      EOS_Data &eos = pmbp->pmhd->peos->eos_data;
      Real gamma = eos.gamma;
      DvceArray5D<Real> &u0 = pmbp->pmhd->u0;

      const Real r0 = r0_ejecta;
      const std::vector<Block> h_ejecta = numerical_data;

      DualArray1D<Real> theta_tm("theta_arr", kNTheta);
      DualArray2D<Real> time_tm("time_arr", kNTheta, kNTime);
      DualArray2D<Real> vinfty_tm("velocity_arr", kNTheta, kNTime);
      DualArray2D<Real> mdot_tm("mdot_arr", kNTheta, kNTime);
      DualArray2D<Real> temp_tm("temperature_arr", kNTheta, kNTime);

      for (int i = 0; i < kNTheta; i++) {
        theta_tm.h_view(i) = h_ejecta[i].th;
      }

      for (int i = 0; i < kNTheta; i++) {
        for (int j = 0; j < kNTime; j++) {
          time_tm.h_view(i,j)   = h_ejecta[i].time[j];
          vinfty_tm.h_view(i,j) = h_ejecta[i].v_infty[j];
          mdot_tm.h_view(i,j)   = h_ejecta[i].mdot[j];
          temp_tm.h_view(i,j)   = h_ejecta[i].temperature[j];
        }
      }

      theta_tm.template modify<HostMemSpace>();
      theta_tm.template sync<DevExeSpace>();

      time_tm.template modify<HostMemSpace>();
      time_tm.template sync<DevExeSpace>();

      vinfty_tm.template modify<HostMemSpace>();
      vinfty_tm.template sync<DevExeSpace>();

      mdot_tm.template modify<HostMemSpace>();
      mdot_tm.template sync<DevExeSpace>();

      temp_tm.template modify<HostMemSpace>();
      temp_tm.template sync<DevExeSpace>(); 

      par_for("numerical_ejecta", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1l = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
        Real x1r = LeftEdgeX(i-is+1, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2l = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        Real x2r = LeftEdgeX(j-js+1, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3l = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        Real x3r = LeftEdgeX(k-ks+1, indcs.nx3, x3min, x3max);

        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        Real dr = sqrt(SQR(dx1)+SQR(dx2)+SQR(dx3));

        Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
        Real rad_l = sqrt(SQR(x1l)+SQR(x2l)+SQR(x3l));
        Real rad_r = sqrt(SQR(x1r)+SQR(x2r)+SQR(x3r));
        Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

        const Real& alpha = adm.alpha(m, k, j, i);
        const Real t_cgs = alpha*t_code/2.99792458e5;

        // ((rad_l - dr/2.0) <= r0/alpha && (rad_r + dr/2.0) >= r0/alpha) 

        if (rad <= (r0+2.0*dr)/alpha && rad > 0.0) {

          Real th = acos(Kokkos::fmin(1.0, Kokkos::fmax(-1.0, x3v/rad)));

          Real epsilon_frac = TransitionEpsilon(dr/alpha, r0/alpha, rad);

          // if (rad_l <= r0/alpha && rad_r >= r0/alpha) {
          //   epsilon_frac = CalcVolFraction(x1l, x1r, x2l, x2r, x3l, x3r, r0/alpha);
            // We are in the transition region, we need to compute the volume fraction
          
          // Find the four vertex (ith,jt), (ith,jt+1) (ith1,kt), (ith1, kt+1)
          Real mdot = Interpolate2D(theta_tm, time_tm, mdot_tm, -0.5, th, t_cgs);
          Real veloc = Interpolate2D(theta_tm, time_tm, vinfty_tm, -0.1, th, t_cgs);
          Real gamma = 1.0/sqrt(1.0 - SQR(veloc/2.99792458e10));
          Real den = mdot/(4*M_PI*SQR(r0*1.0e5)*gamma*veloc);
          Real temp = Interpolate2D(theta_tm, time_tm, temp_tm, -0.2, th, t_cgs);
          Real pres = 7.56573325e-15*pow(temp, 4);

          // We convert from cgs, and K to [M] = g, [L]= km, [c] = 1, and [T] = 1.
          const Real mu = 1.0; // see MNRAS 535, 3711–3731 (2024)
          const Real g_cm3_to_g_km3 = 1.0e15;
          const Real cm_s_1 = 3.335641e-11;
          const Real K_to_1 = 9.251087e-14/mu;
          const Real dyne_to_g_km3 = 1.1126500560536184e-6;

          den *= g_cm3_to_g_km3;
          veloc *= cm_s_1;
          pres *= dyne_to_g_km3;

          Real wvx;
          Real wvy;
          Real wvz;

          if (r_cil == 0) {
            wvx = 0.0;
            wvy = 0.0;
            wvz = veloc * x3v / rad;
            Real Gamma0 = 1.0/sqrt(1.0-SQR(wvx)-SQR(wvy)-SQR(wvz));
            wvx *= Gamma0 / alpha;
            wvy *= Gamma0/ alpha;
            wvz *= Gamma0 / alpha;
          } else {
            wvx = veloc * x1v / rad;
            wvy = veloc * x2v / rad;
            wvz = veloc * x3v / rad;
            Real Gamma0 = 1.0/sqrt(1.0-SQR(wvx)-SQR(wvy)-SQR(wvz));
            wvx *= Gamma0 / alpha;
            wvy *= Gamma0 / alpha;
            wvz *= Gamma0 / alpha;
          }

          // We need to extract w in the expanding cordinates instead of using Gamma_l, because we
          // converted (wv) from Lorentz to the expanding coordinates, but not w alone:

          Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                  adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                  adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};

          Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                      g3d[S22], g3d[S23], g3d[S33]);
          Real vol = sqrt(detg);

          Real v[3] = {wvx, wvy, wvz};
          Real v2 = Primitive::SquareVector(v, g3d);

          // if (1.0 - v2 <= 0) {
          //   std::cout << "The velocity is superluminal!" << std::endl
          //             << "Attempting to adjust..." << std::endl;
          //   Real fac = Kokkos::sqrt((1.0 - 1e-15)/v2);
          //   wvx *= fac;
          //   wvy *= fac;
          //   wvz *= fac;
          //   v2 = 1.0 - 1.0e-15;
          // }

          Real w = sqrt(1.0 + v2); 

          // In addition we lower wv^i:
          Real wv_x = g3d[S11]*wvx + g3d[S12]*wvy + g3d[S13]*wvz;
          Real wv_y = g3d[S12]*wvx + g3d[S22]*wvy + g3d[S23]*wvz;
          Real wv_z = g3d[S13]*wvx + g3d[S23]*wvy + g3d[S33]*wvz;

          // Now we are in position to create the "new vector u0_eq":
          Real u0_or[5];
          Real u0_eq[5];

          u0_or[0] = u0(m,IDN,k,j,i);
          u0_or[1] = u0(m,IM1,k,j,i);
          u0_or[2] = u0(m,IM2,k,j,i);
          u0_or[3] = u0(m,IM3,k,j,i);
          u0_or[4] = u0(m,IEN,k,j,i);

          Real h = 1.0 + (gamma-1.0)/gamma * pres/den;

          u0_eq[0] = vol*w*den;
          u0_eq[1] = vol*den*h*w*wv_x;
          u0_eq[2] = vol*den*h*w*wv_y;
          u0_eq[3] = vol*den*h*w*wv_z;
          u0_eq[4] = vol*(den*h*w*w - pres - w*den); 

          u0(m,IDN,k,j,i) += -alpha*vol*beta_dt*(u0_or[0] - u0_eq[0])/tau * epsilon_frac;
          u0(m,IM1,k,j,i) += -alpha*vol*beta_dt*(u0_or[1] - u0_eq[1])/tau * epsilon_frac;
          u0(m,IM2,k,j,i) += -alpha*vol*beta_dt*(u0_or[2] - u0_eq[2])/tau * epsilon_frac;
          u0(m,IM3,k,j,i) += -alpha*vol*beta_dt*(u0_or[3] - u0_eq[3])/tau * epsilon_frac;
          u0(m,IEN,k,j,i) += -alpha*vol*beta_dt*(u0_or[4] - u0_eq[4])/tau * epsilon_frac;
        }
      });
    }
  }
} 

  // void SetCentralEngine(Mesh* pm, const Real beta_dt) {
  //   // This is where we would set the source terms for the central engine, if we wanted to.
  //   MeshBlockPack *pmbp = pm->pmb_pack;
  //   const Real t = pmbp->pmesh->time;
    
  //   if (t > t_eng || t < t_delay) return;
  //   Real tau = pmbp->pmesh->dt;
    
  //   auto &indcs = pmbp->pmesh->mb_indcs;
  //   int is = indcs.is;
  //   int js = indcs.js;
  //   int ks = indcs.ks;
  //   int ie = indcs.ie;
  //   int je = indcs.je;
  //   int ke = indcs.ke;
  //   int nmb1 = pmbp->nmb_thispack - 1;
  //   auto &size = pmbp->pmb->mb_size;
  //   auto &adm = pmbp->padm->adm;
    

  //   if (pmbp->pmhd != nullptr) {
  //     EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  //     Real gamma = eos.gamma;
  //     DvceArray5D<Real> &u0 = pmbp->pmhd->u0;

  //     const Real Lj_1 = Lj;
  //     const Real v_r_1 = v_r;
  //     const Real v_phi_1 = v_phi;
  //     const Real Gamma_inf_1 = Gamma_inf;
  //     const Real r0_1 = r0;
  //     const Real theta_j_1 = theta_j;
  //     const Real sigma_r_1 = sigma_r;
  //     const Real sigma_phi_1 = sigma_phi;
  //     const Real ratio_1 = ratio;

    
  //     par_for("central_engine", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  //     KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        
  //       Real &x1min = size.d_view(m).x1min;
  //       Real &x1max = size.d_view(m).x1max;
  //       Real x1_l = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
  //       Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
  //       Real x1_r = LeftEdgeX(i-is+1, indcs.nx1, x1min, x1max);

  //       Real &x2min = size.d_view(m).x2min;
  //       Real &x2max = size.d_view(m).x2max;
  //       Real x2_l = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
  //       Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
  //       Real x2_r = LeftEdgeX(j-js+1, indcs.nx2, x2min, x2max);

  //       Real &x3min = size.d_view(m).x3min;
  //       Real &x3max = size.d_view(m).x3max;
  //       Real x3_l = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);
  //       Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
  //       Real x3_r = LeftEdgeX(k-ks+1, indcs.nx3, x3min, x3max);

  //       Real rad_l = sqrt(SQR(x1_l) + SQR(x2_l) + SQR(x3_l));
  //       Real rad_r = sqrt(SQR(x1_r) + SQR(x2_r) + SQR(x3_r));
  //       Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
  //       Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

  //       const Real& alpha = adm.alpha(m, k, j, i);

  //       Real theta = acos(x3v/rad);
  //       Real theta_min = M_PI - theta_j_1;
  //       if ((rad>0 && rad <= r0_1/alpha) && ((theta < theta_j_1) || (theta > theta_min))) {

  //         Real g3d[NSPMETRIC] = {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
  //                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
  //                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};

  //         Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
  //                                     g3d[S22], g3d[S23], g3d[S33]);
  //         Real vol = sqrt(detg);

  //         Real den;
  //         Real wvx;
  //         Real wvy;
  //         Real wvz;
  //         Real wv_x;
  //         Real wv_y;
  //         Real wv_z;
  //         Real pres;

  //         Real Gamma = 1.0 / sqrt(1.0 - (SQR(v_r_1) + SQR(v_phi_1)));
  //         Real Gamma_r = 1.0 / sqrt(1.0 - SQR(v_r_1));
  //         Real h = Gamma_inf_1/Gamma_r; 
  //         Real epsilon_frac = 1.0;

  //         if (rad_l < r0_1/alpha && rad_r > r0_1/alpha) {
  //           epsilon_frac = CalcVolFraction(x1_l, x1_r, x2_l, x2_r, x3_l, x3_r, r0_1/alpha);
  //         }

  //         den = Lj_1/(4*M_PI*SQR(r0_1)*v_r_1*SQR(Gamma_r)*h);

  //         if (r_cil == 0) {
  //           wvx = 0.0;
  //           wvy = 0.0;
  //           wvz = Gamma * (v_r_1*x3v/rad)/alpha;
  //         } else {
  //           Real x;
  //           if (theta > theta_min){
  //             x = M_PI - theta;
  //           } else {
  //             x = theta;
  //           }
  //           wvx = Gamma * (v_r_1*x1v/rad - v_phi_1*(x/theta_j_1)*x2v/r_cil)/alpha;
  //           wvy = Gamma * (v_r_1*x2v/rad + v_phi_1*(x/theta_j_1)*x1v/r_cil)/alpha;
  //           wvz = Gamma * (v_r_1*x3v/rad)/alpha;
  //         }

  //         // We need to extract w in the expanding cordinates instead of using Gamma_l, because we
  //         // converted (wv) from Lorentz to the expanding coordinates, but not w alone.
  //         Real v[3] = {wvx, wvy, wvz};
  //         Real v2 = Primitive::SquareVector(v, g3d);
  //         Real w = sqrt(1.0 + v2); 

  //         wv_x = g3d[S11]*wvx + g3d[S12]*wvy + g3d[S13]*wvz;
  //         wv_y = g3d[S12]*wvx + g3d[S22]*wvy + g3d[S23]*wvz;
  //         wv_z = g3d[S13]*wvx + g3d[S23]*wvy + g3d[S33]*wvz;

  //         // Magnetic Field Prescription
  //         Real eta = 1/(1+0.5*(sigma_r_1+sigma_phi_1))*(h+0.5*(sigma_r_1 + sigma_phi_1)); //express h in terms of h*
  //         Real pres_avg = (gamma - 1.0)/gamma * (eta - 1.0) * den;
  //         Real br = sqrt(2.0*sigma_r_1*pres_avg);
  //         Real factor = -SQR(ratio_1)/(1 + SQR(ratio_1)) + ratio_1*atan(1/ratio_1);
  //         Real factor_inv = 1.0/factor;
  //         Real bphi = Gamma_r * sqrt(pres_avg*sigma_phi_1*factor_inv);  //Check that pavg is correct, note that h-> thermal, not hstar
          
  //         // Adding the magnetic field. This will be used to compute the conserved 
  //         // variables contributions of the prescribed magnetic fields.
  //         Real bx;
  //         Real by;
  //         Real bz;
  //         if (r_cil == 0) {
  //           bx = 0.0;
  //           by = 0.0;
  //           bz = (br*x3v/rad)/SQR(alpha);
  //         } else {
  //           Real theta_m = ratio_1 * theta_j_1;
  //           Real x;
  //           if (theta > theta_min){
  //             x = M_PI - theta;
  //           } else {
  //             x = theta;
  //           }
  //           bx = (br*x1v/rad - bphi*(x/theta_m)*x2v/r_cil)/SQR(alpha);
  //           by = (br*x2v/rad + bphi*(x/theta_m)*x1v/r_cil)/SQR(alpha);
  //           bz = (br*x3v/rad)/SQR(alpha);
  //         }

  //         //Lowering the magnetic field:
  //         Real b_x = g3d[S11]*bx + g3d[S12]*by + g3d[S13]*bz;
  //         Real b_y = g3d[S12]*bx + g3d[S22]*by + g3d[S23]*bz;
  //         Real b_z = g3d[S13]*bx + g3d[S23]*by + g3d[S33]*bz;

  //         // We will use the Lorentz values of br and bphi to compute the pressure.
  //         // Since the pressure is an scalar, and no further corrections are needed.
  //         Real x;
  //         if (theta > theta_min){
  //           x = M_PI - theta;
  //         } else {
  //           x = theta;
  //         }
  //         Real xm = ratio_1 * theta_j_1;
  //         Real p[8];
  //         p[0] = (-2.0*SQR(bphi)*SQR(x))/(SQR(Gamma_r)*(SQR(xm) + SQR(x)));
  //         p[1] = (-2.0*SQR(bphi)*SQR(xm)*SQR(x))/(SQR(Gamma_r)*SQR(SQR(xm) + SQR(x)));
  //         p[2] = (bphi*br*v_phi_1*v_r_1*xm*(-2.0*SQR(x) - (SQR(xm) + SQR(x))*log(SQR(xm)
  //               /(SQR(xm) + SQR(x)))))/(theta_j_1*(SQR(xm) + SQR(x)));
  //         p[3] = (SQR(br)*SQR(v_phi_1)*SQR(x))/(2.0*SQR(theta_j_1));
  //         p[4] = -((bphi*br*v_phi_1*v_r_1*xm*log(1 + SQR(x)/SQR(xm)))/theta_j_1);
  //         p[5] = (den*SQR(Gamma)*h*SQR(v_phi_1)*SQR(x))/(2.0*SQR(theta_j_1));
  //         p[6] = (SQR(br)*SQR(v_phi_1)*SQR(x))/(2.0*SQR(theta_j_1));
  //         p[7] = (-2.0*bphi*br*v_phi_1*v_r_1*xm*log(1.0 + SQR(x)/SQR(xm)))/theta_j_1;

  //         pres = std::accumulate(std::begin(p), std::end(p), pres_avg);
    
  //         Real u0_den = u0(m,IDN,k,j,i);
  //         Real u0_mom1 = u0(m,IM1,k,j,i);
  //         Real u0_mom2 = u0(m,IM2,k,j,i);
  //         Real u0_mom3 = u0(m,IM3,k,j,i);
  //         Real u0_tau = u0(m,IEN,k,j,i);

  //         // Check ideal_grmhd.cpp for the correct routine to make the prim to cons, conversion

  //         Real u0_den_eq = vol*den*w;
  //         Real b2 = b_x * bx + b_y * by + b_z * bz;
  //         Real u0_mom1_eq = vol*(den*h*w*wv_x + b2*wv_x/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_x);
  //         Real u0_mom2_eq = vol*(den*h*w*wv_y + b2*wv_y/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_y);
  //         Real u0_mom3_eq = vol*(den*h*w*wv_z + b2*wv_z/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_z);
  //         Real u0_tau_eq = vol*(den*h*w*w + b2 - pres - 0.5*(SQR(bx*wv_x/w + by*wv_y/w + bz*wv_z/w) + b2/(w*w)) - den*w);

  //         u0(m,IDN,k,j,i) += -alpha*vol*beta_dt*(u0_den - u0_den_eq)/tau * epsilon_frac;
  //         u0(m,IM1,k,j,i) += -alpha*vol*beta_dt*(u0_mom1 - u0_mom1_eq)/tau * epsilon_frac;
  //         u0(m,IM2,k,j,i) += -alpha*vol*beta_dt*(u0_mom2 - u0_mom2_eq)/tau * epsilon_frac;
  //         u0(m,IM3,k,j,i) += -alpha*vol*beta_dt*(u0_mom3 - u0_mom3_eq)/tau * epsilon_frac;
  //         u0(m,IEN,k,j,i) += -alpha*vol*beta_dt*(u0_tau - u0_tau_eq)/tau * epsilon_frac;
  //       }
  //     });
  //   }
  //   return;
  // }
// }
