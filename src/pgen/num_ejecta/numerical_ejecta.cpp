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
#include "pgen/num_ejecta/numerical_ejecta.hpp"


namespace {
  // Expansion:
  Real h0;
  Real mask_tol_ej;
  Real delta_mdot_ej;
  bool is_expanding;
  Real epsilon_h0;
  Real t_exp_max;
  int kNTheta_ej;
  int kNTime_ej;
  Real nu_mdot_ej;
  Real nu_vel_ej;
  Real nu_temp_ej;

  // Central Engine Variables:
  Real Lj_eng;
  Real vr_eng;
  Real vp_eng;
  Real Gamma_inf_eng;
  Real theta_j_eng;
  Real sigma_r_eng;
  Real sigma_phi_eng;
  Real ratio_eng;
  Real r0_ejecta;
  Real epsilon_tol_ej;
  Real delay;
  Real t_eng;
  bool set_eng;

  // Pulsar Wind:
  Real t_waiting; 
  Real t_wind;    
  bool set_wind;
  Real B_star_wind;  
  Real R_star_wind; 
  Real P_star_wind;  
  Real Gammaw_wind;  
  Real sigma_w_wind; 
  Real u_w_wind;

  //Numerical Ejecta:
  Real t_num;
  std::vector<Block> numerical_data;

  //Functors:
  void SetADMVariablesToFLRW(MeshBlockPack *pmbp);
  void SetNumericalEjecta(Mesh* pm, const Real bdt);
  void SetCentralEngine(Mesh* pm, const Real bdt);
  void SetPulsarWind(Mesh* pm, const Real bdt);
  void SetUserSources(Mesh* pm, const Real bdt);
}


KOKKOS_INLINE_FUNCTION
Real Bw_phi(const Real &th, const Real &B_star, const Real &R_star, const Real &P_star, const Real &r0) {
   //speed of light
  Real c_cgs = units::Units::speed_of_light_cgs;
  Real Lw = 2/(3*c_cgs)*SQR(B_star/(1.0e14))*Kokkos::pow((R_star/1.0e6), 4)*Kokkos::pow(P_star, -2); 
  Real Bw2 = 3.0*Lw*SQR(sin(th))/(2.0*r0*c_cgs);
  Real Bw = sqrt(Bw2); 
  return Bw;
}

KOKKOS_INLINE_FUNCTION
Real den_w(const Real &th, const Real &B_star, const Real &R_star, const Real &P_star, 
           const Real &Gammaw, const Real &sigma_w, const Real &r0) {
  Real c_cgs = units::Units::speed_of_light_cgs;
  Real Lw = 2/(3*c_cgs)*SQR(B_star/(1.0e14))*Kokkos::pow((R_star/1.0e6), 4)*Kokkos::pow(P_star, -2); 
  Real Bw2 = 3.0*Lw/(2.0*r0*1.0e5*c_cgs);
  Real rhow = Bw2/(SQR(Gammaw)*SQR(c_cgs)*sigma_w); 
  return rhow;
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
Real TransitionEpsilon(const Real &epsilon_tol, const Real &r0, const Real &rad) {
  // Smooth bump function with compact support: defined (and nonzero) only
  // for (rad - r0) in (-epsilon_tol, +epsilon_tol), i.e. a radial band of
  // half-width epsilon_tol straddling r=r0, and identically zero everywhere
  // else. This localizes the ejecta prescription to a thin shell around
  // r=r0 rather than filling its entire interior.
  Real var;
  Real dist = rad - r0;
  Real eps_inv = 1.0/epsilon_tol;
  Real s = fmin(fmax(dist*eps_inv, -1.0), 1.0);
  var =  0.5*(1.0 - s - M_1_PI*sin(M_PI*s)); 
  return var;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  nu_mdot_ej = pin->GetReal("problem", "nu_mdot");
  nu_vel_ej = pin->GetReal("problem", "nu_vel");
  nu_temp_ej = pin->GetReal("problem", "nu_temp");

  delta_mdot_ej = pin->GetReal("problem", "delta_mdot_ej");
  mask_tol_ej  = pin->GetReal("problem", "mask_tol_ej");

  is_expanding = pin->GetOrAddBoolean("problem", "is_expanding", true);

  t_exp_max = pin->GetReal("problem", "t_exp_max");
  delay     = pin->GetReal("problem", "delay");
  t_eng     = pin->GetReal("problem", "t_eng");
  set_eng   = pin->GetOrAddBoolean("problem", "set_eng", true); 
  
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
  Lj_eng          = pin->GetReal("problem", "Lj_eng");
  vr_eng          = pin->GetReal("problem", "vr_eng"); 
  vp_eng          = pin->GetReal("problem", "vp_eng");
  Gamma_inf_eng   = pin->GetReal("problem", "Gamma_inf_eng");
  theta_j_eng     = pin->GetReal("problem", "theta_j_eng");
  sigma_r_eng     = pin->GetReal("problem", "sigma_r_eng");
  sigma_phi_eng   = pin->GetReal("problem", "sigma_phi_eng");
  ratio_eng       = pin->GetReal("problem", "ratio_eng");
  r0_ejecta       = pin->GetReal("problem", "r0_ejecta");
  epsilon_tol_ej  = pin->GetReal("problem", "epsilon_tol_ej");

  // Pulsar Wind:
  t_waiting    = pin->GetReal("problem", "t_waiting");
  t_wind       = pin->GetReal("problem", "t_wind");
  set_wind     = pin->GetOrAddBoolean("problem", "set_wind", true);
  B_star_wind  = pin->GetReal("problem", "B_star_wind");
  R_star_wind  = pin->GetReal("problem", "R_star_wind");
  P_star_wind  = pin->GetReal("problem", "P_star_wind");
  Gammaw_wind  = pin->GetReal("problem", "Gammaw_wind");
  sigma_w_wind = pin->GetReal("problem", "sigma_w_wind");
  u_w_wind     = pin->GetReal("problem", "u_w_wind");

  {
    // Reading:
    t_num = pin->GetReal("problem", "t_num");
    std::string filename = pin->GetString("problem", "file_path");
    NumericalEjectaData model(filename);
    numerical_data = model.ComputeBlocks();
    kNTheta_ej = model.thsize();
    kNTime_ej = model.tsize();
  }
 
  // Set an immerse bc that recreate the ejecta.
  user_srcs_func = &SetUserSources;

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

  Real n_ism = pin->GetOrAddReal("problem", "n_ism", 3.0);
  Real tau_ism = pin->GetOrAddReal("problem", "tau_ism", 1.0);

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
    const Real epsilon_tol = epsilon_tol_ej;
    const std::vector<Block> h_ejecta = numerical_data;
    const Real nu_mdot = nu_mdot_ej;
    const Real nu_vel = nu_vel_ej;
    const Real nu_temp = nu_temp_ej;
    const Real delta_mdot = delta_mdot_ej;
    const Real mask_tol = mask_tol_ej;
    const int kNTheta = kNTheta_ej;
    const int kNTime = kNTime_ej;
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
      Real dl = Kokkos::min(dx1, Kokkos::min(dx2, dx3));

      Real rad_l = sqrt(SQR(x1l) + SQR(x2l) + SQR(x3l));
      Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      Real rad_r = sqrt(SQR(x1r) + SQR(x2r) + SQR(x3r));
      Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

      Real den;
      Real wvx;
      Real wvy;
      Real wvz;
      Real w;
      Real pres;

      const Real cm_to_km = 1.0e-5;
      const Real km_to_cm = 1/cm_to_km;
      const Real c_cgs = units::Units::speed_of_light_cgs;
      const Real s_to_km = c_cgs * cm_to_km;
      const Real km_to_s = 1/s_to_km;  
      const Real a_cgs = units::Units::rad_constant_cgs;

      // We convert from cgs, and K to [M] = g, [L]= km, [c] = 1, and [T] = 1.
      const Real mu = 1.0; // see MNRAS 535, 3711–3731 (2024)
      const Real g_cm3_to_g_km3 = 1.0/(cm_to_km*cm_to_km*cm_to_km);
      const Real cm_s_1 = 1/c_cgs;
      const Real K_to_1 = units::Units::k_boltzmann_cgs*SQR(cm_s_1)/
                              (mu*units::Units::atomic_mass_unit_cgs);
      const Real dyne_to_g_km3 = 1/SQR(c_cgs)*g_cm3_to_g_km3;

      Real band = epsilon_tol * dl;
      Real epsilon_frac = TransitionEpsilon(band, r0, rad);
      if (epsilon_frac > mask_tol) {
        
        Real th = acos(Kokkos::fmin(1.0, Kokkos::fmax(-1.0, x3v/rad)));
      
        // We extract the density, veloc and pressure in cgs units.
        Real mdot = Interpolate2D(theta_ej, time_ej, mdot_ej, nu_mdot, th, 0.0);

        if (r_cil > 0.0 && delta_mdot > 0.0) {
          mdot *= (1.0 + delta_mdot*x2v/r_cil);
        }

        Real veloc = Interpolate2D(theta_ej, time_ej, vinfty_ej, nu_vel, th, 0.0);
        w = 1.0/sqrt(1.0 - SQR(veloc/c_cgs));
        den = mdot/(4*M_PI*SQR(rad*km_to_cm)*w*veloc);
        Real temp = Interpolate2D(theta_ej, time_ej, temp_ej, nu_temp, th, 0.0);
        pres = a_cgs*pow(temp, 4)/3.0;

        den *= g_cm3_to_g_km3;
        veloc *= cm_s_1;
        pres *= dyne_to_g_km3;

        if (r_cil > 0.0) {
          wvx = veloc * x1v / rad;
          wvy = veloc * x2v / rad;
          wvz = veloc * x3v / rad;
          wvx *= w;
          wvy *= w;
          wvz *= w;
        } else {
          wvx = 0.0;
          wvy = 0.0;
          wvz = veloc * x3v / rad;
          wvz *= w;
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
        Real temp = temp_ism; 
        pres = a_cgs*pow(temp, 4)/3.0;
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

    Real x = (t-t0)/(2*epsilon);
    Real y = t0/(2*epsilon);
    Real log_cosh_x = abs(x) + log1p(exp(-2.0*abs(x)));
    Real log_cosh_y = abs(y) + log1p(exp(-2.0*abs(y)));
    Real exponent = h0*t/2 + h0*epsilon*(log_cosh_x - log_cosh_y);
    const Real a = exp(exponent);
    const Real b = h0/2.0*(1+tanh((t-t0)/(2.0*epsilon)));
    // We will construct a switch to let the ejecta expand from the immmerse b.c
    // turning off the expansion before t_exp_max.
    // const Real b = h0/2.0*(1+tanh((t-t0)/(2.0*epsilon)));
    // const Real a = exp(h0*t/2.0)*pow(cosh((t-t0)/(2.0*epsilon))/cosh(t0/(2.0*epsilon)), h0*epsilon);
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
    //Real tau = pmbp->pmesh->dt;
    Real tau = beta_dt/5.0;

    auto &indcs = pmbp->pmesh->mb_indcs;
    int is = indcs.is;
    int js = indcs.js;
    int ks = indcs.ks;
    int ie = indcs.ie;
    int je = indcs.je;
    int ke = indcs.ke;
    int nmb = pmbp->nmb_thispack;
    auto &size = pmbp->pmb->mb_size;

    if (pmbp->pmhd != nullptr) {
      EOS_Data &eos = pmbp->pmhd->peos->eos_data;
      Real gamma = eos.gamma;
      DvceArray5D<Real> &u0 = pmbp->pmhd->u0;

      const Real r0 = r0_ejecta;
      const Real epsilon_tol = epsilon_tol_ej;
      const std::vector<Block> h_ejecta = numerical_data;
      const int kNTheta = kNTheta_ej;
      const int kNTime  = kNTime_ej;
      const Real nu_mdot = nu_mdot_ej;
      const Real nu_vel = nu_vel_ej;
      const Real nu_temp = nu_temp_ej;
      const Real delta_mdot = delta_mdot_ej;
      const Real mask_tol = mask_tol_ej;

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
        Real dl = Kokkos::min(dx1, Kokkos::min(dx2, dx3));

        Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
        Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

        const Real cm_to_km = 1.0e-5;
        const Real km_to_cm = 1/cm_to_km;
        const Real c_cgs = units::Units::speed_of_light_cgs;
        const Real s_to_km = c_cgs * cm_to_km;
        const Real km_to_s = 1/s_to_km;  
        const Real t_cgs = t_code*km_to_s;
        const Real a_cgs = units::Units::rad_constant_cgs;

        Real band = epsilon_tol * dl;
        Real epsilon_frac = TransitionEpsilon(band, r0, rad);
        if (epsilon_frac > mask_tol) {

          Real th = acos(Kokkos::fmin(1.0, Kokkos::fmax(-1.0, x3v/rad)));

          // Find rhp. veloc, and pres in cgs units
          Real mdot = Interpolate2D(theta_tm, time_tm, mdot_tm, nu_mdot, th, t_cgs);

          if (r_cil > 0.0 && delta_mdot > 0.0) {
            mdot *= (1.0 + delta_mdot*x2v/r_cil);
          }

          Real veloc = Interpolate2D(theta_tm, time_tm, vinfty_tm, nu_vel, th, t_cgs);
          Real w = 1.0/sqrt(1.0 - SQR(veloc/c_cgs));
          Real den = mdot/(4*M_PI*SQR(rad*km_to_cm)*w*veloc);
          Real temp = Interpolate2D(theta_tm, time_tm, temp_tm, nu_temp, th, t_cgs);
          Real pres = a_cgs*pow(temp, 4)/3.0;

          // We convert from cgs, and K to [M] = g, [L]= km, [c] = 1, and [T] = 1.
          const Real mu = 1.0; // see MNRAS 535, 3711–3731 (2024)
          const Real g_cm3_to_g_km3 = 1.0/(cm_to_km*cm_to_km*cm_to_km);
          const Real cm_s_1 = 1/c_cgs;
          const Real K_to_1 = units::Units::k_boltzmann_cgs*SQR(cm_s_1)/
                                  (mu*units::Units::atomic_mass_unit_cgs);
          const Real dyne_to_g_km3 = 1/SQR(c_cgs)*g_cm3_to_g_km3;

          den *= g_cm3_to_g_km3;
          veloc *= cm_s_1;
          pres *= dyne_to_g_km3;

          Real wvx;
          Real wvy;
          Real wvz;

          if(r_cil > 0) {
            wvx = w*veloc * x1v / rad;
            wvy = w*veloc * x2v / rad;
            wvz = w*veloc * x3v / rad;
          } else {
            wvx = 0.0;
            wvy = 0.0;
            wvz = w*veloc * x3v / rad;
          }

          Real h = 1.0 + (gamma-1.0)/gamma * pres/den;

          Real u0_or[5];
          Real u0_eq[5];

          u0_or[0] = u0(m,IDN,k,j,i);
          u0_or[1] = u0(m,IM1,k,j,i);
          u0_or[2] = u0(m,IM2,k,j,i);
          u0_or[3] = u0(m,IM3,k,j,i);
          u0_or[4] = u0(m,IEN,k,j,i);

          u0_eq[0] = w*den;
          u0_eq[1] = den*h*w*wvx;
          u0_eq[2] = den*h*w*wvy;
          u0_eq[3] = den*h*w*wvz;
          u0_eq[4] = den*h*w*w - pres - w*den;

          u0(m,IDN,k,j,i) = u0_eq[0] + (u0_or[0] - u0_eq[0])*exp(-epsilon_frac*beta_dt/tau);
          u0(m,IM1,k,j,i) = u0_eq[1] + (u0_or[1] - u0_eq[1])*exp(-epsilon_frac*beta_dt/tau);
          u0(m,IM2,k,j,i) = u0_eq[2] + (u0_or[2] - u0_eq[2])*exp(-epsilon_frac*beta_dt/tau);
          u0(m,IM3,k,j,i) = u0_eq[3] + (u0_or[3] - u0_eq[3])*exp(-epsilon_frac*beta_dt/tau);
          u0(m,IEN,k,j,i) = u0_eq[4] + (u0_or[4] - u0_eq[4])*exp(-epsilon_frac*beta_dt/tau);

          //u0(m,IDN,k,j,i) += -beta_dt*(u0_or[0] - u0_eq[0])/tau * epsilon_frac;
          //u0(m,IM1,k,j,i) += -beta_dt*(u0_or[1] - u0_eq[1])/tau * epsilon_frac;
          //u0(m,IM2,k,j,i) += -beta_dt*(u0_or[2] - u0_eq[2])/tau * epsilon_frac;
          //u0(m,IM3,k,j,i) += -beta_dt*(u0_or[3] - u0_eq[3])/tau * epsilon_frac;
          //u0(m,IEN,k,j,i) += -beta_dt*(u0_or[4] - u0_eq[4])/tau * epsilon_frac;
        }
      });
    }
  }
 
  void SetCentralEngine(Mesh* pm, const Real beta_dt) {
    // This is where we would set the source terms for the central engine, if we wanted to.
    if (!set_eng) {
      return;
    }

    MeshBlockPack *pmbp = pm->pmb_pack; 
    Real tau = beta_dt/5.0;
    const Real t_code = pmbp->pmesh->time; 
    auto &indcs = pmbp->pmesh->mb_indcs;
    int is = indcs.is;
    int js = indcs.js;
    int ks = indcs.ks;
    int ie = indcs.ie;
    int je = indcs.je;
    int ke = indcs.ke;
    int nmb1 = pmbp->nmb_thispack - 1;
    auto &size = pmbp->pmb->mb_size; 

    if (pmbp->pmhd != nullptr) {
      EOS_Data &eos = pmbp->pmhd->peos->eos_data;
      Real gamma = eos.gamma;
      DvceArray5D<Real> &u0 = pmbp->pmhd->u0;

      const Real Lj = Lj_eng;
      const Real vr = vr_eng;
      const Real vp = vp_eng;
      const Real Gamma_inf = Gamma_inf_eng;
      const Real r0 = r0_ejecta;
      const Real theta_j = theta_j_eng;
      const Real sigma_r = sigma_r_eng;
      const Real sigma_phi = sigma_phi_eng;
      const Real ratio = ratio_eng;
      const Real epsilon_tol = epsilon_tol_ej;

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

        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        Real dl = Kokkos::min(dx1, Kokkos::min(dx2, dx3));

        Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

        Real band = epsilon_tol * dl;
        Real epsilon_frac = TransitionEpsilon(band, r0, rad);
        Real theta = acos(x3v/rad);
        Real theta_min = M_PI - theta_j;

        if ((epsilon_frac > 1.0e-3) && ((theta < theta_j) || (theta > theta_min))) {

          Real den;
          Real wvx;
          Real wvy;
          Real wvz;
          Real pres;
          
          // Check units here, and finally at wind
          const Real cm_to_km = 1.0e-5;
          const Real km_to_cm = 1/cm_to_km;
          const Real c_cgs = units::Units::speed_of_light_cgs;
          const Real s_to_km = c_cgs * cm_to_km;
          const Real km_to_s = 1/s_to_km;  
          const Real t_cgs = t_code*km_to_s;
          const Real a_cgs = units::Units::rad_constant_cgs;

          Real x;
          if (theta > theta_min){
            x = M_PI - theta;
          } else {
            x = theta;
          }

          Real w = 1.0 / sqrt(1.0 - (SQR(vr) + SQR(vp*(x/theta_j))));
          Real w_r = 1.0 / sqrt(1.0 - SQR(vr));
          Real h = Gamma_inf/w_r; 

          // We convert from cgs, and K to [M] = g, [L]= km, [c] = 1, and [T] = 1.
          const Real mu = 1.0; // see MNRAS 535, 3711–3731 (2024)
          const Real g_cm3_to_g_km3 = 1.0/(cm_to_km*cm_to_km*cm_to_km);
          const Real cm_s_1 = 1/c_cgs;
          const Real K_to_1 = units::Units::k_boltzmann_cgs*SQR(cm_s_1)/(mu*units::Units::atomic_mass_unit_cgs);
          const Real dyne_to_g_km3 = 1/SQR(c_cgs)*g_cm3_to_g_km3;

          den = Lj/(4*M_PI*SQR(rad*km_to_cm)*(vr*c_cgs)*SQR(w_r)*h*SQR(c_cgs));
          den *= dyne_to_g_km3;

          // vr and vp do not need to be converted to cgs because they are already multiples of c.
          if (r_cil == 0) {
            wvx = 0.0;
            wvy = 0.0;
            wvz = w * vr*x3v/rad;
          } else {
            wvx = w * vr*x1v/rad - vp*(x/theta_j)*x2v/r_cil;
            wvy = w * vr*x2v/rad + vp*(x/theta_j)*x1v/r_cil;
            wvz = w * vr*x3v/rad;
          }

          // Magnetic Field Prescription. Because [pres_avg]=[den] we do not need to convert bphi
          Real eta = 1/(1+0.5*(sigma_r+sigma_phi))*(h+0.5*(sigma_r + sigma_phi)); //express h in terms of h*
          Real pres_avg = (gamma - 1.0)/gamma * (eta - 1.0) * den;
          Real br = sqrt(2.0*sigma_r*pres_avg);
          Real factor = -SQR(ratio)/(1 + SQR(ratio)) + ratio*atan(1/ratio);
          Real factor_inv = 1.0/factor;
          Real bphi = w_r * sqrt(pres_avg*sigma_phi*factor_inv);

          // Adding the magnetic field. This will be used to compute the conserved 
          // variables contributions of the prescribed magnetic fields.
          Real bx;
          Real by;
          Real bz;
          if (r_cil == 0) {
            bx = 0.0;
            by = 0.0;
            bz = br*x3v/rad;
          } else {
            Real theta_m = ratio * theta_j;
            bx = br*x1v/rad - bphi*(x/theta_m)*x2v/r_cil;
            by = br*x2v/rad + bphi*(x/theta_m)*x1v/r_cil;
            bz = br*x3v/rad;
          }

          Real b2 = SQR(bx) + SQR(by) + SQR(bz);

          // We will use the Lorentz values of br and bphi to compute the pressure.
          // Since the pressure is an scalar, and no further corrections are needed.
          Real xm = ratio * theta_j;
          Real p[8];
          p[0] = (-2.0*SQR(bphi)*SQR(x))/(SQR(w_r)*(SQR(xm) + SQR(x)));
          p[1] = (-2.0*SQR(bphi)*SQR(xm)*SQR(x))/(SQR(w_r)*SQR(SQR(xm) + SQR(x)));
          p[2] = (bphi*br*vp*vr*xm*(-2.0*SQR(x) - (SQR(xm) + SQR(x))*log(SQR(xm)
                /(SQR(xm) + SQR(x)))))/(theta_j*(SQR(xm) + SQR(x)));
          p[3] = (SQR(br)*SQR(vp)*SQR(x))/(2.0*SQR(theta_j));
          p[4] = -((bphi*br*vp*vr*xm*log(1 + SQR(x)/SQR(xm)))/theta_j);
          p[5] = (den*SQR(w)*h*SQR(vp)*SQR(x))/(2.0*SQR(theta_j));
          p[6] = (SQR(br)*SQR(vp)*SQR(x))/(2.0*SQR(theta_j));
          p[7] = (-2.0*bphi*br*vp*vr*xm*log(1.0 + SQR(x)/SQR(xm)))/theta_j;

          pres = 0.0;
          for(int ii=0; ii<8; ii++) {
          pres += p[ii];
          }

          Real u0_orig[5];
          u0_orig[0] = u0(m,IDN,k,j,i);
          u0_orig[1] = u0(m,IM1,k,j,i);
          u0_orig[2] = u0(m,IM2,k,j,i);
          u0_orig[3] = u0(m,IM3,k,j,i);
          u0_orig[4] = u0(m,IEN,k,j,i);

          // Check ideal_grmhd.cpp for the correct routine to make the prim to cons, conversion
          Real u0_eq[5];
          u0_eq[0] = den*w;
          u0_eq[1] = den*h*w*wvx + b2*wvx/w  - (bx*wvx/w + by*wvy/w + bz*wvz/w)*bx;
          u0_eq[2] = den*h*w*wvy + b2*wvy/w  - (bx*wvx/w + by*wvy/w + bz*wvz/w)*by;
          u0_eq[3] = den*h*w*wvz + b2*wvz/w  - (bx*wvx/w + by*wvy/w + bz*wvz/w)*bz;
          u0_eq[4] = den*h*w*w + b2 - pres - 0.5*(SQR(bx*wvx/w + by*wvy/w + bz*wvz/w) + b2/(w*w)) - den*w;

          u0(m,IDN,k,j,i) = u0_eq[0] + (u0_orig[0] - u0_eq[0])*exp(-epsilon_frac*beta_dt/tau);
          u0(m,IM1,k,j,i) = u0_eq[1] + (u0_orig[1] - u0_eq[1])*exp(-epsilon_frac*beta_dt/tau);
          u0(m,IM2,k,j,i) = u0_eq[2] + (u0_orig[2] - u0_eq[2])*exp(-epsilon_frac*beta_dt/tau);
          u0(m,IM3,k,j,i) = u0_eq[3] + (u0_orig[3] - u0_eq[3])*exp(-epsilon_frac*beta_dt/tau);
          u0(m,IEN,k,j,i) = u0_eq[4] + (u0_orig[4] - u0_eq[4])*exp(-epsilon_frac*beta_dt/tau);
          
          //u0(m,IDN,k,j,i) += -beta_dt*(u0_orig[0] - u0_eq[0])/tau * epsilon_frac;
          //u0(m,IM1,k,j,i) += -beta_dt*(u0_orig[1] - u0_eq[1])/tau * epsilon_frac;
          //u0(m,IM2,k,j,i) += -beta_dt*(u0_orig[2] - u0_eq[2])/tau * epsilon_frac;
          //u0(m,IM3,k,j,i) += -beta_dt*(u0_orig[3] - u0_eq[3])/tau * epsilon_frac;
          //u0(m,IEN,k,j,i) += -beta_dt*(u0_orig[4] - u0_eq[4])/tau * epsilon_frac;
        }
      });
    }
    return;
  }

  void SetPulsarWind(Mesh *pm, const Real beta_dt) {

    // This is where we would set the source terms for the pulsar wind, if we wanted to.
    if (!set_wind) {
      return;
    }

    MeshBlockPack *pmbp = pm->pmb_pack; 
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

      const Real B_star  = B_star_wind;
      const Real R_star  = R_star_wind;
      const Real P_star  = P_star_wind;
      const Real Gammaw  = Gammaw_wind;
      const Real u_wind  = u_w_wind;
      const Real sigma_w = sigma_w_wind;
      const Real r0      = r0_ejecta;
      const Real epsilon_tol = epsilon_tol_ej;

      par_for("pulsar_wind", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
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

        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        Real dl = Kokkos::min(dx1, Kokkos::min(dx2, dx3));

        Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        Real r_cil = sqrt(SQR(x1v) + SQR(x2v));

        const Real& alpha = adm.alpha(m, k, j, i);

        Real band = epsilon_tol * dl;
        Real epsilon_frac = TransitionEpsilon(band, r0/alpha, rad);
        Real theta = acos(x3v/rad);

        if (epsilon_frac > 1.0e-3) {

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
          Real vr;
          Real wv_x;
          Real wv_y;
          Real wv_z;
          Real pres;
          Real bphi;

          // Check units here, and finally at wind
          const Real cm_to_km = 1.0e-5;
          const Real km_to_cm = 1/cm_to_km;
          const Real c_cgs = units::Units::speed_of_light_cgs;
          const Real s_to_km = c_cgs * cm_to_km;
          const Real km_to_s = 1/s_to_km;  
          const Real a_cgs = units::Units::rad_constant_cgs;

          den = den_w(theta, B_star, R_star, P_star, Gammaw, sigma_w, r0/alpha);
          vr = c_cgs*sqrt(1- SQR(1/Gammaw));
          pres = u_wind/3.0/(4.0*M_PI*SQR(alpha*rad*km_to_cm/alpha));
          bphi = Bw_phi(theta, B_star, R_star, P_star, r0/alpha); 

          // Change of units from cgs to code:
          const Real mu = 1.0; // see MNRAS 535, 3711–3731 (2024)
          const Real g_cm3_to_g_km3 = 1.0/(cm_to_km*cm_to_km*cm_to_km);
          const Real cm_s_1 = 1/c_cgs;
          const Real K_to_1 = units::Units::k_boltzmann_cgs*SQR(cm_s_1)/(mu*units::Units::atomic_mass_unit_cgs);
          const Real dyne_to_g_km3 = 1/SQR(c_cgs)*g_cm3_to_g_km3;

          den *= g_cm3_to_g_km3;
          vr /= c_cgs;
          pres *= dyne_to_g_km3;
          bphi *= sqrt(dyne_to_g_km3);

          Real h = den + ((gamma -1)/gamma)*pres + SQR(bphi/Gammaw);
          
          if (r_cil == 0) {
            wvx = 0.0;
            wvy = 0.0;
            wvz = Gammaw * (vr*x3v/rad)/alpha;
          } else {
            wvx = Gammaw * (vr*x1v/rad)/alpha;
            wvy = Gammaw * (vr*x2v/rad)/alpha;
            wvz = Gammaw * (vr*x3v/rad)/alpha;
          }

          Real v[3] = {wvx, wvy, wvz};
          Real v2 = Primitive::SquareVector(v, g3d);
          Real w = sqrt(1.0 + v2); 

          wv_x = g3d[S11]*wvx + g3d[S12]*wvy + g3d[S13]*wvz;
          wv_y = g3d[S12]*wvx + g3d[S22]*wvy + g3d[S23]*wvz;
          wv_z = g3d[S13]*wvx + g3d[S23]*wvy + g3d[S33]*wvz;

          Real bx;
          Real by;
          Real bz;
          if (r_cil == 0) {
            bx = 0.0;
            by = 0.0;
            bz = 0.0;
          } else {
            bx = -bphi*(x2v/r_cil)/SQR(alpha);
            by =  bphi*(x1v/r_cil)/SQR(alpha);
            bz = 0.0;
          }

          //Lowering the magnetic field:
          Real b_x = g3d[S11]*bx + g3d[S12]*by + g3d[S13]*bz;
          Real b_y = g3d[S12]*bx + g3d[S22]*by + g3d[S23]*bz;
          Real b_z = g3d[S13]*bx + g3d[S23]*by + g3d[S33]*bz;
          Real b2 = b_x * bx + b_y * by + b_z * bz;

          // Now we need to reconstruct the conservative variables

          Real u0_orig[5];
          u0_orig[0] = u0(m,IDN,k,j,i);
          u0_orig[1] = u0(m,IM1,k,j,i);
          u0_orig[2] = u0(m,IM2,k,j,i);
          u0_orig[3] = u0(m,IM3,k,j,i);
          u0_orig[4] = u0(m,IEN,k,j,i);

          Real u0_eq[5];
          u0_eq[0] = vol*den*w;
          u0_eq[1] = vol*(den*h*w*wv_x + b2*wv_x/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_x);
          u0_eq[2] = vol*(den*h*w*wv_y + b2*wv_y/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_y);
          u0_eq[3] = vol*(den*h*w*wv_z + b2*wv_z/w  - (bx*wv_x/w + by*wv_y/w + bz*wv_z/w)*b_z);
          u0_eq[4] = vol*(den*h*w*w + b2 - pres - 0.5*(SQR(bx*wv_x/w + by*wv_y/w + bz*wv_z/w) + b2/(w*w)) - den*w);

          u0(m,IDN,k,j,i) += -alpha*vol*beta_dt*(u0_orig[0] - u0_eq[0])/tau * epsilon_frac;
          u0(m,IM1,k,j,i) += -alpha*vol*beta_dt*(u0_orig[1] - u0_eq[1])/tau * epsilon_frac;
          u0(m,IM2,k,j,i) += -alpha*vol*beta_dt*(u0_orig[2] - u0_eq[2])/tau * epsilon_frac;
          u0(m,IM3,k,j,i) += -alpha*vol*beta_dt*(u0_orig[3] - u0_eq[3])/tau * epsilon_frac;
          u0(m,IEN,k,j,i) += -alpha*vol*beta_dt*(u0_orig[4] - u0_eq[4])/tau * epsilon_frac;
        }
      });
    } 
    return;
  }

  void SetUserSources(Mesh* pm, const Real bdt) {
    const Real t = pm->time;
    if (t > 0.0 && t <= t_num) {
      SetNumericalEjecta(pm, bdt);
    } else if (t > t_num && t <= (t_num + delay)) {
      return;
    } else if (t > (t_num + delay) && t <= (t_num + delay + t_eng)) {
      SetCentralEngine(pm, bdt);
    } else if (t > (t_num + delay + t_eng) && t <= (t_num + delay + t_eng + t_waiting)) {
      return;
    } else if (t > (t_num + delay + t_eng + t_waiting) && t <= (t_num + delay + t_eng + t_waiting + t_wind)) {
      SetPulsarWind(pm, bdt);
    } else {
      return;
    }  
  }
}


