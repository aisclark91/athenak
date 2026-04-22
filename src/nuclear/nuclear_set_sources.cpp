
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file set_sources.cpp
//! \brief calculate source term for mass fraction conservation


#include "athena.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "nuclear/nuclear.hpp"
#include "utils/tov/tov_tabulated.hpp"


namespace nuclear {
    TaskStatus Sources(Driver* d, int stage) {

        // Compute Sources to be used in the nuclear reaction network. As a first example we will code
        // the Urca Process for the electrons only. It would be interesing to add muons as well when possible.

        RetrieveSource_<tov::TabulatedEOS>(d, stage);

        return TaskStatus::complete;
    }

    template <class EOSPolicy>
    TaskStatus RetrieveSource_(Driver* d, int stage) {
        
        // Retrieve the chemical potentials from the EOS to compute the afinity vector.
        RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        auto nmb1 = pmy_pack->nmb_thispack - 1;

        par_for("nuclear_relax_source", DevExeSpace(), 0, nmb1, ks, ke, js,
            je, is, ie,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {

                // We populate the Ye in beta equilibrium from the <tov::TabulatedEOS> table
                // given at the pgen.

                // // Compute the state thermodynamic quantities from the EOS.
                // Real rho = w0_(m, IDN, k, j, i);
                // Real nb = rho / mn;
                // Real p = w0_(m, IPR, k, j, i);
               
                // Real T = eos.GetTemperatureFromP(nb, p, &Y);
                // Real e = eos.GetEnergy(nb, T, Y_part);  //We may need to modify this?

                // // Fraction that contains the neutrino number densities for e-, e+, mu-, mu+, tau-, tau+ respectively.
                // Real n_nu[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

                // //Compute the beta equilibrium composition, given the electron mass fraction given the EOS:
                // Real Y_part[3] = {Y, 0., 0.};  // Leptons Mass fractions
                // Real Y_lep[3]{};
                // eos.GetLeptonFractions(nb, Y_part, n_nu, Y_lep);
                // Real Y_guess[3] = {Y_lep[0], Y_lep[1], Y_lep[2]};

                // Real Y_lepton_eq[3]{};
                // Real T_eq{};
                // bool res = eos.GetBetaEquilibriumTrapped(
                //                 nb, e, Y_lep, T_eq, &Y_lepton_eq[0], T, Y_guess);
                
                // Real Y_eq = Y_lepton_eq[0];

                // Define the source term for the electron fraction:
                Real Y = w0_(m, IE, k, j, i);
                source = inv_tau * (Y_tabular_eq(m, k, j, i) - Y);  
                s0_(m, k, j, i) = source;    
            }
        );
        return TaskStatus::complete;
}

}
