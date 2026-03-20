// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file init.cpp
 */
#include <godunov_mhd_ct/init/init.h>

#include <godunov_mhd_ct/SolverGodunovMHD.h>

#include <godunov_mhd_ct/init/InitBlast.h>
#include <godunov_mhd_ct/init/InitFieldLoopAdvection.h>
#include <godunov_mhd_ct/init/InitKelvinHelmholtz.h>
#include <godunov_mhd_ct/init/InitOrszagTang.h>
#include <godunov_mhd_ct/init/InitRotor.h>
#include <godunov_mhd_ct/init/InitRayleighTaylor.h>
#include <godunov_mhd_ct/init/InitShockTube.h>
#include <godunov_mhd_ct/init/InitDiamagCavity.h>

#include <kalypsso/core/io_utils.h>
#ifdef KALYPSSO_CORE_USE_HDF5
#  include <kalypsso/core/HDF5_Xdmf_Reader.h>
#endif

namespace kalypsso
{

namespace godunov_mhd_ct
{
// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
init(SolverGodunovMHD<dim, device_t> & solver)
{
  // check if we are performing a re-start run (default : false)
  bool restartEnabled = solver.config_map().getBool("restart", "enabled", false);

  if (restartEnabled)
  {
    // load data from input data file
    init_restart(solver);
  }
  else
  {
    const auto problem_name = solver.problem_name();

    if (!problem_name.compare("blast"))
    {
      InitBlast<dim, device_t>::apply(solver);
    }
    else if (!problem_name.compare("field_loop_advection"))
    {
      InitFieldLoopAdvection<dim, device_t>::apply(solver);
    }
    else if (!problem_name.compare("rotor"))
    {
      InitRotor<dim, device_t>::apply(solver);
    }
    else if (!problem_name.compare("orszag_tang"))
    {
      InitOrszagTang<dim, device_t>::apply(solver);
    }
    else if (!problem_name.compare("kelvin_helmholtz"))
    {
      InitKelvinHelmholtz<dim, device_t>::apply(solver);
    }
    else if (!problem_name.compare("rayleigh_taylor"))
    {
      InitRayleighTaylor<dim, device_t>::apply(solver);
    }
    else if (!problem_name.compare("shock_tube"))
    {
      InitShockTube<dim, device_t>::apply(solver);
    }
    else if (!problem_name.compare("cavity"))
    {
      InitDiamagCavity<dim, device_t>::apply(solver);
    }
    else
    {
      KALYPSSO_WARN("Problem : {} is not recognized / implemented.", problem_name);
      KALYPSSO_WARN("Use default - orszag_tang");
      InitOrszagTang<dim, device_t>::apply(solver);
    }

    // print mesh info
    {
      const auto amr_mesh_info = solver.mesh_map()->get_amr_mesh_info();
      if (solver.par_env().rank() == 0)
      {
        amr_mesh_info.print();
      }
    }
  }
} // init (regular)

template void
init<2, kalypsso::DefaultDevice>(SolverGodunovMHD<2, kalypsso::DefaultDevice> & solver);
template void
init<3, kalypsso::DefaultDevice>(SolverGodunovMHD<3, kalypsso::DefaultDevice> & solver);

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
init_restart([[maybe_unused]] SolverGodunovMHD<dim, device_t> & solver)
{

#ifdef KALYPSSO_CORE_USE_HDF5

  const auto config_map = solver.config_map();

  const std::string outputDir = config_map.getString("output", "outputDir", ".");
  const auto        filePrefix = config_map.getString("restart", "filePrefix", "");

  const std::string hdf5_filename = outputDir + "/" + filePrefix + ".h5";
  const std::string p4est_filename = outputDir + "/" + filePrefix + ".p4est";

  if (solver.par_env().rank() == 0)
  {
    if (!file_exists(hdf5_filename))
    {
      Kokkos::abort("HDF5 restart file doesn't exist");
    }
    if (!file_exists(p4est_filename))
    {
      Kokkos::abort("p4est restart file doesn't exist");
    }
  }

  // read p4est mesh and setup AMRmesh
  solver.amr_mesh()->create_p4est_resources_from_file(
    solver.par_env(), solver.config_map(), p4est_filename.c_str());

  // refresh AMR
  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // read solver data from HDF5 file
  {
    [[maybe_unused]] uint64_t total_num_bytes = 0;

    // resize user data U, U2, Bface, Bface2
    solver.resize_solver_data();

    auto reader =
      HDF5_Xdmf_Reader<dim, device_t>(solver.par_env(), solver.config_map(), solver.mesh_map());
    reader.open(hdf5_filename);

    // read time and number of time steps
    reader.read_scalar_attribute("run", "time", solver.current_time());
    reader.read_scalar_attribute("run", "iteration", solver.iteration());
    solver.begin_time() = solver.current_time();

    // retrieve available / allowed names: fieldManager, and field map (fm)
    const auto & fm = solver.model().get_fieldmap();

    // a map containing ID and name of the variable to write
    const auto id2names = solver.model().get_id2names_map();

    for (auto & iter : id2names)
    {
      auto varId = static_cast<typename core::models::MHD::VarId>(iter.first);

      // get variables string name
      const auto varName = id2names.at(varId);

      if (varId != core::models::MHD::IBX and varId != core::models::MHD::IBY and
          varId != core::models::MHD::IBZ)
      {
        total_num_bytes += reader.read_quadrant_attribute(
          solver.Uhost(), fm[varId], varName, 0, solver.Uhost().num_quadrants());
      }

    } // end for iter

    // load Bx, By, Bz all at once as face centered variables
    total_num_bytes += reader.read_quadrant_attribute(solver.Bface(), "magnetic_field");

    reader.close();

    Kokkos::deep_copy(solver.U().logical_view(), solver.Uhost().logical_view());
    Kokkos::deep_copy(solver.U2().logical_view(), solver.Uhost().logical_view());
    // copying Bface is already taken care of
  }

#else
  Kokkos::abort("Restarting a run is only available through HDF5 checkpoint files.");
#endif

} // init_restart

template void
init_restart<2, kalypsso::DefaultDevice>(SolverGodunovMHD<2, kalypsso::DefaultDevice> & solver);
template void
init_restart<3, kalypsso::DefaultDevice>(SolverGodunovMHD<3, kalypsso::DefaultDevice> & solver);

} // namespace godunov_mhd_ct

} // namespace kalypsso
