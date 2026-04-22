//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file integrate.cpp
//! \brief integrate the source term

#include "athena.hpp"
#include "nuclear.hpp"


namespace nuclear {

    TaskStatus Integrate(Driver* d, int stage) {
        // Integrate the source term for the nuclear reaction network. We will use a simple forward Euler method for now, but we can implement more sophisticated methods in the future.

        RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        auto nmb1 = pmy_pack->nmb_thispack - 1;

        par_for("nuclear_integrate", DevExeSpace(), 0, nmb1, ks, ke, js,
            je, is, ie,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
                w0_(m, IE, k, j, i) += s0_(m, k, j, i) * dtnew;  // Update the electron fraction using the source term and the new timestep.
            }
        );

        return TaskStatus::complete;
    }


}