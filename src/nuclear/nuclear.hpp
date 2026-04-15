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
    TaskID Nuclear_irecv;
    TaskID Nuclear_copyu;
    TaskID Nuclear_closure;
    TaskID Nuclear_flux;
    TaskID Nuclear_sendf;
    TaskID Nuclear_recvf;
    TaskID Nuclear_rkupdt;
    TaskID Nuclear_mattersrc;
    TaskID Nuclear_restu;
    TaskID Nuclear_sendu;
    TaskID Nuclear_recvu;
    TaskID Nuclear_bcs;
    TaskID Nuclear_prol;
    TaskID Nuclear_newdt;
    TaskID Nuclear_csend;
    TaskID Nuclear_crecv;
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
            
            DvceArray5D<Real> xnuc;// flag to check if <adm> present

            void AssembleRadiationM1Tasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
            // ...in "before_stagen_tl" list
            TaskStatus InitRecv(Driver* d, int stage);
            // ...in "stagen_tl" list

            TaskStatus CalcClosure(Driver* d, int stage);
            TaskStatus SendFlux(Driver* d, int stage);
            TaskStatus RecvFlux(Driver* d, int stage);
            TaskStatus TimeUpdate(Driver* d, int stage);
            TaskStatus CalcNuclearSources(Driver* pdrive, int stage);
            TaskStatus RestrictU(Driver* d, int stage);
            TaskStatus SendU(Driver* d, int stage);
            TaskStatus RecvU(Driver* d, int stage);
            TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
            TaskStatus Prolongate(Driver* pdrive, int stage);
            TaskStatus NewTimeStep(Driver* d, int stage);
            // ...in "after_stagen_tl" list
            TaskStatus ClearSend(Driver* d, int stage);
            TaskStatus ClearRecv(Driver* d, int stage);  // also in Driver::Initialize

            // eos related quantites (only when mhd is on)
            template <class EOSPolicy, class ErrorPolicy>
            TaskStatus CalcOpacityYe_(Driver* pdrive, int stage);
            // template <class EOSPolicy, class ErrorPolicy>
            // TaskStatus CalcOpacityPhotons_(Driver* pdrive, int stage);
            template <class EOSPolicy, class ErrorPolicy, int M1_NGHOST>
            TaskStatus TimeUpdate_(Driver* d, int stage);

         private:
            MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack
};

    

}