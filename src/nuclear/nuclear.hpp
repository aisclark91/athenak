#ifndef NUCLEAR_HPP
#define NUCLEAR_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.hpp
//  \brief definitions for Grey M1 radiation class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "bvals/bvals.hpp"
#include "parameter_input.hpp"

namespace nuclear {

    struct NuclearTaskIDs {
    TaskID Nuclear_spec;
    TaskID Nuclear_sources;
    TaskID Nuclear_newdt;
    TaskID Nuclear_integrate;
    TaskID Nuclear_free;
};

    class Nuclear {
        public:
            Nuclear(MeshBlockPack* ppack, ParameterInput* pin);
            ~Nuclear();

            int nspecies;                            // no. of species
            bool ismhd;                              // flag to check if <mhd> present
            bool ishydro;                            // flag to check if <hydro> present
            bool isunits;                            // flag to check if <units> present
            bool isadm; 

            NuclearTaskIDs id;          // container to hold names of TaskIDs
            Real dtnew{};
            Real sources;
            Real inv_tau;

            DvceArray4D<Real> y_eq;
            DvceArray5D<Real> w0_;

            if (!ismhd) {
                std::cout << "The nuclear module requires the MHD module to be active." << std::endl;
                exit(EXIT_FAILURE);
            else {
                w0_ = pmy_pack->pmhd->w0;
            }
            
            void AssembleNuclearTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
            // ...in "before_stagen_tl" list
            TaskStatus SetSpec(Driver* d, int stage);  //Maybe not necessary, we can catch them from the mhd scalars.

            // ...in "stagen_tl" list
            TaskStatus Sources(Driver* d, int stage);
            TaskStatus NewTimeStep(Driver* d, int stage);
            TaskStatus UpdateSpec(Driver* d, int stage);

            // ...in "after_stagen_tl" list
            TaskStatus FreeSpec(Driver* d, int stage);

            // Enforce EOS consistency after the update is done (only when mhd is on)
            template <class EOSPolicy, class ErrorPolicy>
            TaskStatus Consistent_(Driver* pdrive, int stage);

             template <class EOSPolicy, class ErrorPolicy>
            TaskStatus Source_(Driver* pdrive, int stage);

        private:
            MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack
    };
}