#include <algorithm>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mhd/mhd.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "nuclear/nuclear.hpp"

namespace nuclear {
  TaskStatus Nuclear::Integrate(Driver *pdrive, int stage) {
    // This correspond to the caller that performs the integration of the network
    // dt_nuc should be computed from e/edot
    if (pmy_pack->pmesh->pgen->user_nuc_srcs) {
      Real bdt = std::min(pmy_pack->pmesh->dt, dtnew);
      (pmy_pack->pmesh->pgen->user_srcs_nuc_func)(pmy_pack->pmesh, bdt);
    }

    return TaskStatus::complete;
  }
}
