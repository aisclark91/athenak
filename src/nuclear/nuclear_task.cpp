#include <iostream>
#include <map>
#include <memory>
#include <mhd/mhd.hpp>
#include <string>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "nuclear/nuclear.hpp"

namespace nuclear {
	void Nuclear::AssembleNuclearTasks(
	  std::map<std::string, std::shared_ptr<TaskList>> tl) {
	  TaskID none(0);
	  id.Nuclear_dt = tl["opsplit_stagen"]->AddTask(&Nuclear::NewTimeStep, this, none, "Nuclear::CalculateTimestep");
	  id.Nuclear_integrate = tl["opsplit_stagen"]->AddTask(&Nuclear::Integrate, this, id.Nuclear_dt, "Nuclear::Integrate");
	}
}


