// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file solver_godunov_mhd_ct.cpp
 * \brief kalypsso solver for mhd with constraint transport.
 */
#include <cstdlib>
#include <cstdio>
#include <string>

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/real_type.h>   // choose between single and double precision
#include <kalypsso/core/HydroParams.h> // read parameter file

#ifdef KALYPSSO_CORE_USE_MPI
#  include <mpi.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/SolverBase.h>
#include <godunov_mhd_ct/SolverGodunovMHD.h>

#include <godunov_mhd_ct/init/InitShockTube.h>

#include <kalypsso/utils/config/ConfigMap.h>

#include <kalypsso/core/ComputeError.h>
#include <kalypsso/core/OutputParams.h>
#include <kalypsso/core/ComputeDataSliceAlongLine.h>

#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>

// banner
#include "kalypsso_core_version.h"
#include <kalypsso/core/kalypsso_core_git_info.h>
#include <kalypsso/core/kalypsso_core_build_info.h>

#ifdef KALYPSSO_CORE_USE_CPPTRACE
#  include <kalypsso/core/cpptrace_utils.h>
#endif // KALYPSSO_CORE_USE_CPPTRACE

namespace kalypsso
{

/* ============================================================ */
/* ============================================================ */
/* ============================================================ */
template <size_t dim, typename device_t>
void
run_simulation(ParallelEnv const &       par_env,
               ConfigMap const &         config_map,
               [[maybe_unused]] int &    argc,
               [[maybe_unused]] char **& argv)
{

  // test: create a HydroParams object
  HydroParams params = HydroParams(config_map);

  // initialize workspace memory (U, U2, ...)
  auto solver =
    godunov_mhd_ct::SolverGodunovMHD<dim, device_t>::create(par_env, params, config_map);

  // diagnostics
  const auto conservativity_check_enabled =
    config_map.getBool("diagnostic", "conservativity_checks", true);

  // start computation
  if (par_env.rank() == 0)
  {
    KALYPSSO_INFO("Start computation....");
  }

  solver->profiling_mgr().get_whole_region().start();

  // register conservative integral values at initial time
  if (conservativity_check_enabled)
    solver->register_volume_integrals(true);

  // MHD solver time loop
  solver->run();

  // register conservative integral values at final time
  if (conservativity_check_enabled)
  {
    solver->register_volume_integrals(false);
    solver->print_conservativity_check_report();
  }

  // save last time step as a regular checkpoint
  const auto output_params = OutputParams(config_map);
  if (output_params.nOutput != 0)
  {
    // save solution, this is not a pure checkpoint
    // just a regular output with all required fields for a checkpoint
    constexpr bool pure_checkpoint = false;
    solver->save_solution(pure_checkpoint);

    // save p4est mesh
    auto derived_solver = dynamic_cast<godunov_mhd_ct::SolverGodunovMHD<dim, device_t> *>(solver);
    const std::string mesh_filename = derived_solver->output_basename() + ".p4est";
    derived_solver->template save_p4est_mesh<dim>(mesh_filename,
                                                  derived_solver->amr_mesh()->forest());
  }

  solver->profiling_mgr().get_whole_region().stop();

  // =================================================================
  // here we do something specific to a given test case
  // =================================================================

  // =================================================================================
  // when doing a shock tube problem (Brio-Wu, ...), dump a 1D slice of data
  // =================================================================================
  if (!solver->problem_name().compare("shock_tube"))
  {
    auto       solver_mhd = dynamic_cast<godunov_mhd_ct::SolverGodunovMHD<dim, device_t> *>(solver);
    const auto st_params = MHDShockTubeParams(config_map);
    const auto st_name = config_map.getString("shock-tube", "name", "shock_tube");

    const auto cell_var_ids = std::vector<int32_t>{
      solver_mhd->model().get_fieldmap()[core::models::MHD::ID],
      solver_mhd->model().get_fieldmap()[core::models::MHD::IP],
      solver_mhd->model().get_fieldmap()[core::models::MHD::IU],
      solver_mhd->model().get_fieldmap()[core::models::MHD::IV],
      solver_mhd->model().get_fieldmap()[core::models::MHD::IW],
    };

    const auto cell_var_names =
      std::vector<std::string>{ "rho", "pressure", "rhou", "rhov", "rhow" };

    const auto face_var_ids = std::vector<int32_t>{ IX, IY, IZ };

    const auto face_var_names = std::vector<std::string>{ "Bx", "By", "Bz" };

    kalypsso::core::ComputeDataSliceAlongLine<dim, device_t>::apply(
      solver_mhd->U(),
      solver_mhd->Bface(),
      0,
      solver_mhd->mesh_map()->get_amr_mesh_info().local_num_quadrants(),
      st_params.direction,
      solver_mhd->mesh_map()->orchard_keys(),
      cell_var_ids,
      cell_var_names,
      face_var_ids,
      face_var_names,
      st_name,
      par_env,
      config_map);

  } // shock tube post-processing

  KALYPSSO_INFO("final time is {:010.2f}\n", solver->current_time());

  solver->print_monitoring_info_final();

  delete solver;

} // run_simulation

} // namespace kalypsso

// ===============================================================
// ===============================================================
// ===============================================================
int
main(int argc, char * argv[])
{

  {
    // create parallel environment (p4est, MPI, kokkos, ...)
    kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
    // logger setup
    kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

#ifdef KALYPSSO_CORE_USE_CPPTRACE
    kalypsso::cpptrace_initialize();
#endif // KALYPSSO_CORE_USE_CPPTRACE

    // parse command line arguments
    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--version"))
    {
      if (par_env.rank() == 0)
      {
        kalypsso::GitRevisionInfo::print();
        kalypsso::BuildInfo::print();
      }
      return EXIT_SUCCESS;
    }
    else if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
    {
      if (par_env.rank() == 0)
      {
        // clang-format off
        std::cout << "Example cmdline: \"mpirun -np 1 ./solver_godunov_mhd_ct --ini test_blast_2D_block.ini\"\n";
        // clang-format on
      }
      return EXIT_SUCCESS;
    }

    // print kalypsso banner
    if (par_env.rank() == 0)
    {
      kalypsso::GitRevisionInfo::print();
      kalypsso::BuildInfo::print();
    }

    // check if user passed a custom ini filename
    // provide a default input filename if user did'nt set one
    std::string input_filename = kalypsso::cmdline_get_string(argv, argv + argc, "--ini");

    if (input_filename.size() == 0)
      input_filename = "test_blast_2D_block.ini";

    // only MPI rank 0 actually reads input file, and broadcast it to all other MPI processor
    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(input_filename);

    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[solver_godunov_mhd_ct] Wrong dimension");

    // run some test
    if (dim == 2)
    {
      kalypsso::run_simulation<2, kalypsso::DefaultDevice>(par_env, config_map, argc, argv);
    }
    else if (dim == 3)
    {
      kalypsso::run_simulation<3, kalypsso::DefaultDevice>(par_env, config_map, argc, argv);
    }
    else
    {
      if (par_env.rank() == 0)
      {
        std::cerr << "We shouldn't be here; please check your input file.\n";
      }
    }
  }

  return EXIT_SUCCESS;

} // end main
