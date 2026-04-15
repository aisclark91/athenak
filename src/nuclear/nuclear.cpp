//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.cpp
//  \brief implementation of grey M1 radiation class

#include "radiation_m1/radiation_m1.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"

namespace nuclear {
Nuclear::Nuclear(MeshBlockPack *pmb_pack, ParameterInput *pin) :
      : pmy_pack(ppack),
        xnuc("xnuc", 1, 1, 1, 1, 1) {
        // set up parameters and flags
        ishydro = pin->DoesBlockExist("hydro");
        isunits = pin->DoesBlockExist("units");
        ismhd = pin->DoesBlockExist("mhd");
        isadm = pin->DoesBlockExist("adm");

        int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));

        // initialize nuclear reaction network and species mass fractions here
        nspecies = NUC_TOTAL_SPECIES;

        // allocate memory for evolved variables
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int ncells1 = indcs.nx1 + 2 * (indcs.ng);
        int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
        int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;
        Kokkos::realloc(xnuc, nmb, nspecies, ncells3, ncells2, ncells1);

        // allocate boundary buffers for evolved (cell-centered) variables
        pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
        pbval_u->InitializeBuffers(nvarstot);
    }

Nuclear::~Nuclear() { delete pbval_u; }    
}