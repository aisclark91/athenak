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
Nuclear::Nuclear(MeshBlockPack *ppack, ParameterInput *pin) :
      : pmy_pack(ppack) {
        // set up parameters and flags
        ishydro = pin->DoesBlockExist("hydro");
        isunits = pin->DoesBlockExist("units");
        ismhd = pin->DoesBlockExist("mhd");
        isadm = pin->DoesBlockExist("adm");

        // int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));

        // initialize nuclear reaction network and species mass fractions here
        atol = pin->GetOrAddReal("nuclear", "atol", 1.0e-10);
        rtol = pin->GetOrAddReal("nuclear", "rtol", 1.0e-10);
        inv_tau = pin->GetOrAddReal("nuclear", "inv_tau", 1.0e-3); // relaxation timescale in seconds, we will need to convert it to code units.
    }

Nuclear::~Nuclear(){}    
}