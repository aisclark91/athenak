
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
#include "utils/tov/tov_utils.hpp"
#include "utils/tov/tov_tabulated.hpp"


namespace nuclear {
    TaskStatus Sources(Driver* d, int stage) {

        // Compute Sources to be used in the nuclear reaction network. As a first example we will code
        // the Urca Process for the electrons only. It would be interesing to add muons as well when possible.

        RetrieveSource_< >(d, stage);

        return TaskStatus::complete;
    }

    template <class EOSPolicy, class ErrorPolicy>
    TaskStatus RetrieveSource_(Driver* d, int stage) {
        
        // Retrieve the chemical potentials from the EOS to compute the afinity vector.
        RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        auto nmb1 = pmy_pack->nmb_thispack - 1;

        Real mb{};
        Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
            static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(
            pmy_pack->pdyngr)->eos.ps.GetEOSMutable();
        mb = eos.GetBaryonMass();
        

        par_for("nuclear_relax_source", DevExeSpace(), 0, nmb1, ks, ke, js,
            je, is, ie,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
                // Compute the state thermodynamic quantities from the EOS.
                Real rho = w0_(m, IDN, k, j, i);
                Real nb = rho / mn;
                Real p = w0_(m, IPR, k, j, i);
                Real T = eos.GetTemperatureFromP(nb, p, &Ye);

                Real ye[10]{5.0e-1,
                            1.0e-1,
                            5.0e-2,
                            1.0e-2,
                            5.0e-3,
                            1.0e-3,
                            5.0e-4,
                            1.0e-4,
                            5.0e-5,
                            1.0e-5};

                Real ye_eq_value = eos.ColdBetaEquilibrium(nb, T, ye); 
                y_eq(m, k, j, i) = ye_eq_value;
            }
        );
        return TaskStatus::complete;
}

}
