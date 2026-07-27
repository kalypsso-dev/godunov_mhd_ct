// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SolverGodunovMHD.cpp
 *
 * Class SolverGodunovMHD implementation.
 */
#include <godunov_mhd_ct/SolverGodunovMHD.h>

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/config_utils.h> // for get_block_sizes
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h>

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/utils/log/kalypsso_log.h>

// AMR services
#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshPartitioner.h>
#endif // KALYPSSO_CORE_USE_MPI
#include <kalypsso/core/ComputeRefineFlags.h>
#include <kalypsso/core/UserDataRemapper.h>
#include <kalypsso/core/AMRMeshMonitoring.h>
#include <kalypsso/core/LinearCombination.h>

// Init conditions functors
#include <godunov_mhd_ct/init/init.h>

// Compute functors
#include <godunov_mhd_ct/border_conditions/FillOutside.h>
#include <godunov_mhd_ct/scheme/ComputeDtMHDFunctor.h>
#include <godunov_mhd_ct/utils/ComputeDerivedQuantities.h>

// profiling colors
#include <godunov_mhd_ct/profiling.h>

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
#  include <kalypsso/core/CheckFaceBorderCompatibility.h>
#endif

#include <kalypsso/utils/monitoring/memory_utils.h>

#include <algorithm> // std::for_each

namespace kalypsso
{

namespace godunov_mhd_ct
{

// =======================================================
// =========== CLASS SolverGodunovMHD IMPL ===============
// =======================================================

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
SolverGodunovMHD<dim, device_t>::SolverGodunovMHD(ParallelEnv const & par_env,
                                                  HydroParams const & params,
                                                  ConfigMap const &   config_map)
  : SolverBase(par_env, params, config_map)
  , m_block_sizes(get_block_sizes<dim>(config_map))
  , m_brick_sizes(get_brick_sizes<dim>(config_map))
  , m_is_brick_periodic(get_brick_periodicity<dim>(config_map))
  , m_model(config_map)
  , m_mhd_settings(config_map)
  , m_amr_mesh(std::make_shared<AMRmesh<dim>>(par_env, config_map))
  , m_mesh_map(std::make_shared<MeshMap<dim, device_t>>(config_map, par_env))
#ifdef KALYPSSO_CORE_USE_MPI
  , m_mesh_ghosts_exchanger(config_map, par_env, *m_amr_mesh, *m_mesh_map)
#endif // KALYPSSO_CORE_USE_MPI
  , m_amr_context(0)
  , m_time_integrator(TimeIntegratorConfig::get_time_integrator(config_map))
  , m_conservativity_check()
  , m_godunov_impl_version(config_map.getInteger("hydro", "implementation_version", 0))
  , m_U()
  , m_Uhost()
  , m_U2()
  , m_U_RK()
  , m_Bface("Bface", m_block_sizes, 0)
  , m_Bface2("Bface2", m_block_sizes, 0)
  , m_Bface_RK("Bface_RK", m_block_sizes, 0)
  , m_bc_types(BorderConditionsConfig<BC_MHD>::read_border_condition<dim>(config_map, par_env))
#ifdef KALYPSSO_CORE_USE_HDF5
  , m_hdf5_writer(std::make_shared<HDF5_Xdmf_Writer_t>(m_par_env, m_config_map, m_mesh_map))
#endif // KALYPSSO_CORE_USE_HDF5
{
  // TODO: evaluate if we should abort earlier, i.e. before this constructor
  if (m_godunov_impl_version != 0)
  {
    KALYPSSO_ERROR("Wrong value for input parameter \"hydro/implementation_version={}\". Please "
                   "review your input parameter file (allowed values: 0, 1 or 2).",
                   m_godunov_impl_version);
    Kokkos::abort("Wrong value for input parameter hydro/implementation_version");
  }


  /*
   * memory pre-allocation.
   *
   * Note that m_Uhost is not just a view to U, m_Uhost will be used
   * to save data from multiple other device array.
   * That's why we didn't use create_mirror_view to initialize m_Uhost.
   */

  // minimal number of cells only used for initial memory allocation
  // afterwards memory resizing will append
  int32_t nbOcts = 1 << params.level_min;

  nbOcts = params.dimType == TWO_D ? nbOcts * nbOcts : nbOcts * nbOcts * nbOcts;

  /*
   * setup parameters related to block AMR
   */
  const int32_t ghostWidth = 2;

  if (m_block_sizes[IX] < 2 * ghostWidth)
  {
    KALYPSSO_ERROR("bx should be >= 2*ghostWidth.");
    Kokkos::abort("bx should be >= 2*ghostWidth");
  }
  if (m_block_sizes[IY] < 2 * ghostWidth)
  {
    KALYPSSO_ERROR("by should be >= 2*ghostWidth.");
    Kokkos::abort("by should be >= 2*ghostWidth");
  }

  if constexpr (dim == 3)
  {
    if (m_block_sizes[IZ] < 2 * ghostWidth)
    {
      KALYPSSO_ERROR("bz should be >= 2*ghostWidth.");
      Kokkos::abort("bz should be >= 2*ghostWidth");
    }
  }

  /*
   * main data array memory allocation
   */

  m_U = DataArrayBlock_t("U", m_block_sizes, HYDRO_NBVAR_CELL, nbOcts);
  m_Uhost = DataArrayBlock_t::create_host_mirror_view(m_U);

  m_U2 = DataArrayBlock_t("U2", m_block_sizes, HYDRO_NBVAR_CELL, nbOcts);

  m_Bface.resize(nbOcts);
  m_Bface2.resize(nbOcts);

  if (m_time_integrator == +TimeIntegrator::RK2_SSP)
  {
    m_U_RK = DataArrayBlock_t("U_RK", m_block_sizes, HYDRO_NBVAR_CELL, nbOcts);
    m_Bface_RK.resize(nbOcts);
  }

  m_godunov_implem = create_godunov_implem(m_godunov_impl_version);

  /*
   * perform init condition
   */
  // analytical initial conditions or restart
  init(*this);

  // copy U into U2
  Kokkos::deep_copy(m_U2.logical_view(), m_U.logical_view());
  Kokkos::deep_copy(m_Bface2.logical_view(), m_Bface.logical_view());

  if (m_time_integrator == +TimeIntegrator::RK2_SSP)
  {
    Kokkos::deep_copy(m_U_RK.logical_view(), m_U.logical_view());
    Kokkos::deep_copy(m_Bface_RK.logical_view(), m_Bface.logical_view());
  }
  else if (m_time_integrator == +TimeIntegrator::RK3_SSP)
  {
    Kokkos::abort("RK3_SSP is not supported currently");
  }

  // compute initialize time step
  this->compute_dt();

  if (par_env.rank() == 0)
  {
    KALYPSSO_INFO("================================================");
    KALYPSSO_INFO("Solver is {}", this->solver_name());
    KALYPSSO_INFO("Problem (init condition) is {}", m_problem_name);
    KALYPSSO_INFO("Time integrator is {}", m_time_integrator._to_string());
    KALYPSSO_INFO("================================================");

    // print parameters on screen
    params.print();
  }

} // SolverGodunovMHD<dim, device_t>::SolverGodunovMHD

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
SolverGodunovMHD<dim, device_t>::~SolverGodunovMHD()
{

  delete m_godunov_implem;

} // SolverGodunovMHD<dim, device_t>::~SolverGodunovMHD

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::resize_auxiliary_data()
{

  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, AMR_CYCLE_RESIZE_AUX);

  m_Uhost.resize(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

  assertm(m_godunov_implem != nullptr,
          "[SolverGodunovMHD::resize_auxiliary_data] godunov_implem pointer has invalid value.");

  m_godunov_implem->resize_auxiliary_data();

} // SolverGodunovMHD<dim, device_t>::resize_auxiliary_data

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::resize_solver_data()
{

  m_U.resize(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());
  m_U2.resize(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

  // resize Bface (magnetic field components)
  m_Bface.resize(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());
  m_Bface2.resize(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

  if (m_time_integrator == +TimeIntegrator::RK2_SSP)
  {
    m_U_RK.resize(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());
    m_Bface_RK.resize(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());
  }

  resize_auxiliary_data();

} // SolverGodunovMHD<dim, device_t>::resize_solver_data

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::update_mesh(bool reset_p4est_ghost)
{

  if (reset_p4est_ghost)
  {
    m_amr_mesh->reset_ghost();
  }

  // update number of outside quad
  m_mesh_map->compute_outside_quad_info(m_amr_mesh->forest(), m_amr_mesh->ghost());

  // update orchard keys array and hashmap
  m_mesh_map->update_hashmap(m_amr_mesh->forest(), m_amr_mesh->ghost(), true);

  // update orchard keys in mirror quadrants (useful for computing primitive variables in mirror and
  // doing MPI comm)
  m_mesh_map->update_mirror_orchard_keys(m_amr_mesh->ghost());

  // recompute/update number of quadrants per types (owned, ghost, outside, outside_ghost)
  m_mesh_map->update_amr_mesh_info(m_amr_mesh->forest(), m_amr_mesh->ghost());

  m_mesh_map->update_conformal_status();

} // SolverGodunovMHD<dim, device_t>::update_mesh

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::do_amr_cycle()
{
  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, AMR_CYCLE);

  /*
   * Following steps:
   *
   * 1. MPI synchronize user data to update ghost cell values + Update outside quadrants userdata
   * (border condition)
   * 2. resize AMR context
   *    mark cell for refinement / coarsening (requiring up to date ghost)
   * 3. adapt mesh
   * 4. remap user data to the new mesh
   * 5. load balance mesh and user data
   */

  // 1. User data communication / fill ghost data, ghost data
  //    may be needed, e.g. if refine/coarsen condition is a gradient
  synchronize_mpi_ghost_data(m_U, m_Bface);
  fill_outside_quadrants(m_U, m_Bface);

  // 2. resize AMR flags (stored in AMRContext)
  //    and mark cell for refinement / coarsening
  m_amr_context.resize_amrflags(m_mesh_map->get_amr_mesh_info().local_num_quadrants());
  mark_cells();

  // 3. adapt mesh (refine/coarsen/balance) + re-compute connectivity

  // 3.1 store hashmap before mesh adaptation
  auto hashmap_before_amr = m_mesh_map->hashmap_clone();

  // 3.2 adapt mesh (apply refinement using p4est)
  adapt_mesh();

  // 4. map data to new data array
  map_userdata_after_adapt(hashmap_before_amr);

  // 5. update orchard keys and hashmap (so that ghost is also up to date)
  // This is probably (TBC) not needed because already done above inside adapt_mesh
  // constexpr bool do_reset_ghosts = true;
  // update_mesh(do_reset_ghosts);

  // 6. resize auxiliary data
  resize_auxiliary_data();

  // call base class member (increment number of AMR cycle done)
  SolverBase::do_amr_cycle();

  const auto memory_monitoring_verbose =
    m_config_map.getBool("run", "memory_monitoring_verbose", false);
  if (memory_monitoring_verbose)
  {
    print_device_mem_info<device_t>();
  }

} // SolverGodunovMHD<dim, device_t>::do_amr_cycle

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::do_load_balancing()
{

  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, LOAD_BALANCING);

#ifdef KALYPSSO_CORE_USE_MPI
  if (m_par_env.nRanks() > 1)
  {

    MeshPartitioner<dim, device_t> mesh_partitioner(m_config_map, m_par_env);

    // 1. partition mesh
    {
      KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, LOAD_BALANCING_PARTITION_MESH);
      mesh_partitioner.partition_mesh(this->m_amr_mesh->forest());
    }

    // 2. partition user data from U to U2, U2 is reallocated
    {
      KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, LOAD_BALANCING_PARTITION_USERDATA);
      mesh_partitioner.repartition_userdata(m_U, m_U2);
      mesh_partitioner.repartition_userdata(m_Bface, m_Bface2);
    }

    // 3. update orchard keys and hashmap (so that ghost is also up to date)
    {
      KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, LOAD_BALANCING_UPDATE_MESH);
      constexpr bool do_reset_ghosts = true;
      update_mesh(do_reset_ghosts);
    }

    {
      KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, LOAD_BALANCING_RESIZE);

      // 4. resize U, U2 and auxiliary arrays to the final size (taking into account ghost, outside
      // quads, ...)
      resize_solver_data();

      // 5. swap U and U2; U is the most up to date
      my_swap(m_U, m_U2);
      my_swap(m_Bface, m_Bface2);

      Kokkos::deep_copy(m_U2.logical_view(), m_U.logical_view());
      Kokkos::deep_copy(m_Bface2.logical_view(), m_Bface.logical_view());
    }
  }
#endif // KALYPSSO_CORE_USE_MPI

} // SolverGodunovMHD<dim, device_t>::do_load_balancing

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
real_t
SolverGodunovMHD<dim, device_t>::compute_dt_local()
{
  KALYPSSO_PROFILING_REGION_DEVICE(m_profiling_mgr, NUM_CFL_LOCAL);

  real_t dt;
  real_t invDt = ZERO_F;

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = m_model.get_fieldmap();

  // call device functor - compute invDt
  ComputeDtMHDFunctor<dim, device_t>::apply(m_config_map,
                                            m_mesh_map->orchard_keys(),
                                            m_mesh_map->get_amr_mesh_info().local_num_quadrants(),
                                            m_mhd_settings,
                                            fm,
                                            m_block_sizes,
                                            m_U,
                                            m_Bface,
                                            invDt);

  dt = m_mhd_settings.hydro.cfl / invDt;

  return dt;

} // SolverGodunovMHD<dim, device_t>::compute_dt_local

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::next_iteration_impl()
{
  // mesh adaptation (perform refine / coarsen)
  // at the end: most current data will be stored in U
  // amr cycle is automatically disabled when level_min == level_max
  if (should_do_amr_cycle())
  {
    KALYPSSO_INFO("AMR cycle at time t={} step={}", m_t, m_iteration);
    do_amr_cycle();
  }

  if (should_do_load_balancing())
  {
    do_load_balancing();
  }

  // output
  if (m_output_params.enable_output)
  {
    // check if a pure checkpoint is requested
    if (should_do_checkpoint())
    {
      // save p4est mesh for restart
      const std::string mesh_filename = this->output_basename() + ".p4est";
      this->save_p4est_mesh<dim>(mesh_filename, m_amr_mesh->forest());

      KALYPSSO_INFO("Checkpoint at time t={} step={} dt={}", m_t, m_iteration, m_dt);

      this->m_times_saved_restart++;

      // doing a pure checkpoint ?
      const bool pure_checkpoint = should_save_solution() ? false : true;
      save_solution(pure_checkpoint);
    }
    else if (should_save_solution())
    {
      KALYPSSO_INFO("Output results at time t={} step={} dt={}", m_t, m_iteration, m_dt);

      // doing a regular output, that can also be a checkpoint;
      // in that case at least all necessary (conservative) variables will be saved
      constexpr bool pure_checkpoint = false;
      save_solution(pure_checkpoint);

      // also print monitoring info when saving data
      print_monitoring_info();
    } // end output
  } // end enable output

  if (should_print_monitoring_info())
  {
    print_monitoring_info();
  }

  // make sure to always print monitoring information (levels histogram) at begin
  // even when saving data is disabled
  if ((!should_save_solution() or !m_output_params.enable_output) and m_iteration == 0)
    print_monitoring_info();

  // compute new dt
  {
    KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, NUM_CFL);
    compute_dt();
  }

  if (m_iteration % m_nlog == 0)
  {
    KALYPSSO_INFO("time step={:07d} (dt={:10.8f} t={:10.8f})", m_iteration, m_dt, m_t);
  }

  // perform one step integration
  godunov_unsplit(m_dt);

  // compute number of cell-update in current MPI proc.
  m_total_num_cell_updates +=
    static_cast<uint64_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants() * m_U.num_cells());

} // SolverGodunovMHD<dim, device_t>::next_iteration_impl

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::godunov_unsplit(real_t dt)
{

  const auto time_integrator = TimeIntegratorConfig::get_time_integrator(m_config_map);
  if (time_integrator == +TimeIntegrator::HANCOCK)
  {
    // we need conservative variables in ghost cell to be up to date
    synchronize_mpi_ghost_data(m_U, m_Bface);
    fill_outside_quadrants(m_U, m_Bface);

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders before time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");

    // check that normal magnetic field component at block border is "continuous"
    core::CheckFaceBorderCompatibility<dim, device_t>::apply(m_Bface,
                                                             m_mesh_map->hashmap(),
                                                             m_mesh_map->orchard_keys(),
                                                             m_mesh_map->get_amr_mesh_info(),
                                                             m_config_map,
                                                             m_brick_sizes,
                                                             m_is_brick_periodic,
                                                             m_par_env);
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders before time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");
#endif

    // perform actual time step integration
    m_godunov_implem->do_time_step(m_U, m_U2, m_Bface, m_Bface2, dt);

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders after  time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");

    synchronize_mpi_ghost_data(m_U, m_Bface2);
    fill_outside_quadrants(m_U, m_Bface2);

    // check that normal magnetic field component at block border is "continuous"
    core::CheckFaceBorderCompatibility<dim, device_t>::apply(m_Bface2,
                                                             m_mesh_map->hashmap(),
                                                             m_mesh_map->orchard_keys(),
                                                             m_mesh_map->get_amr_mesh_info(),
                                                             m_config_map,
                                                             m_brick_sizes,
                                                             m_is_brick_periodic,
                                                             m_par_env);
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders after  time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");
#endif

    // we need to copy U2 into U to be ready for next time step
    Kokkos::deep_copy(m_U.logical_view(), m_U2.logical_view());

    // we need to copy Bface2 into Bface to be ready for next time step
    Kokkos::deep_copy(m_Bface.logical_view(), m_Bface2.logical_view());
  } // end if HANCOCK

  else if (m_time_integrator == +TimeIntegrator::RK2_SSP)
  {
    const auto num_octants =
      static_cast<int32_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

    // we need conservative variables in ghost cell to be up to date
    synchronize_mpi_ghost_data(m_U, m_Bface);
    fill_outside_quadrants(m_U, m_Bface);
    Kokkos::deep_copy(m_U_RK.logical_view(), m_U.logical_view());
    Kokkos::deep_copy(m_Bface_RK.logical_view(), m_Bface.logical_view());

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders before time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");

    // check that normal magnetic field component at block border is "continuous"
    core::CheckFaceBorderCompatibility<dim, device_t>::apply(m_Bface,
                                                             m_mesh_map->hashmap(),
                                                             m_mesh_map->orchard_keys(),
                                                             m_mesh_map->get_amr_mesh_info(),
                                                             m_config_map,
                                                             m_brick_sizes,
                                                             m_is_brick_periodic,
                                                             m_par_env);
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders before time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");
#endif

    // perform actual time step integration
    m_godunov_implem->do_time_step(m_U, m_U_RK, m_Bface, m_Bface_RK, dt);

    LinearCombination<dim, device_t>::apply(
      m_U, m_U_RK, TimeIntegratorConfig::RK2_SSP_STAGE1_COEFS, num_octants);

    LinearCombination<dim, device_t>::apply(
      m_Bface, m_Bface_RK, TimeIntegratorConfig::RK2_SSP_STAGE1_COEFS, num_octants);

    // we need conservative variables in ghost cell to be up to date
    synchronize_mpi_ghost_data(m_U_RK, m_Bface_RK);
    fill_outside_quadrants(m_U_RK, m_Bface_RK);
    Kokkos::deep_copy(m_U2.logical_view(), m_U_RK.logical_view());
    Kokkos::deep_copy(m_Bface2.logical_view(), m_Bface_RK.logical_view());

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders before time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");

    // check that normal magnetic field component at block border is "continuous"
    core::CheckFaceBorderCompatibility<dim, device_t>::apply(m_Bface_RK,
                                                             m_mesh_map->hashmap(),
                                                             m_mesh_map->orchard_keys(),
                                                             m_mesh_map->get_amr_mesh_info(),
                                                             m_config_map,
                                                             m_brick_sizes,
                                                             m_is_brick_periodic,
                                                             m_par_env);
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders before time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");
#endif

    // perform actual time step integration
    m_godunov_implem->do_time_step(m_U_RK, m_U2, m_Bface_RK, m_Bface2, dt);

    LinearCombination<dim, device_t>::apply(
      m_U, m_U2, TimeIntegratorConfig::RK2_SSP_STAGE2_COEFS, num_octants);

    LinearCombination<dim, device_t>::apply(
      m_Bface, m_Bface2, TimeIntegratorConfig::RK2_SSP_STAGE2_COEFS, num_octants);

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders after  time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");

    // check that normal magnetic field component at block border is "continuous"
    core::CheckFaceBorderCompatibility<dim, device_t>::apply(m_Bface2,
                                                             m_mesh_map->hashmap(),
                                                             m_mesh_map->orchard_keys(),
                                                             m_mesh_map->get_amr_mesh_info(),
                                                             m_config_map,
                                                             m_brick_sizes,
                                                             m_is_brick_periodic,
                                                             m_par_env);
    KALYPSSO_INFO("==============================================================================");
    KALYPSSO_INFO("Checking magnetic field flux at block borders after  time step {}", m_iteration);
    KALYPSSO_INFO("==============================================================================");
#endif

    // we need to copy U2 into U to be ready for next time step
    Kokkos::deep_copy(m_U.logical_view(), m_U2.logical_view());

    // we need to copy Bface2 into Bface to be ready for next time step
    Kokkos::deep_copy(m_Bface.logical_view(), m_Bface2.logical_view());
  } // end if RK2_SSP

} // SolverGodunovMHD<dim, device_t>::godunov_unsplit

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::save_solution_impl(bool pure_checkpoint)
{

  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, IO);

  // if (m_output_params.exabrick_enabled)
  //   save_solution_exabrick();

  if (m_output_params.hdf5_enabled)
    save_solution_hdf5(pure_checkpoint);

} // SolverGodunovMHD<dim, device_t>::save_solution_impl()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::print_monitoring_info()
{

  auto monitor = AMRMeshMonitoring<dim, device_t>(m_config_map);

  monitor.print_level_info(m_par_env, *m_mesh_map);

  // also print memory footprint
  KALYPSSO_INFO("Effective memory footprint (per MPI process) : {:.2f} MBytes",
                1e-6 * static_cast<double>(total_mem_size_in_bytes()));
  KALYPSSO_INFO("Effective memory footprint (per owned cell ) : {:.2f} Bytes",
                1.0 * static_cast<double>(total_mem_size_in_bytes()) / m_U.num_cells() /
                  m_U.num_quadrants());

} // SolverGodunovMHD<dim, device_t>::print_monitoring_info()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::print_monitoring_info_final()
{

  print_monitoring_info();

  m_profiling_mgr.print_timings();

  // compute total number of cell-updates by reducing over all MPI processes.
  uint64_t cell_updates = 0;
#ifdef KALYPSSO_CORE_USE_MPI
  m_par_env.comm().template MPI_Allreduce<MpiComm::SUM>(
    &m_total_num_cell_updates, &cell_updates, 1);
#else
  cell_updates = m_total_num_cell_updates;
#endif // KALYPSSO_CORE_USE_MPI

  auto total_time_opt = m_profiling_mgr.total_time();

  if (total_time_opt.has_value())
  {
    KALYPSSO_INFO("Total number of Mcell-updates/s : {:10.2f}\n",
                  static_cast<double>(cell_updates) / total_time_opt.value() * 1e-6);
  }
  else
  {
    KALYPSSO_INFO("Total number of cell-updates/s : not available\n");
  }

} // SolverGodunovMHD<dim, device_t>::print_monitoring_info_final

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::register_volume_integrals(bool is_reference)
{
  const auto num_octants = m_mesh_map->get_amr_mesh_info().local_num_quadrants();

  m_conservativity_check.register_value(m_U,
                                        0,
                                        num_octants,
                                        m_mesh_map->orchard_keys(),
                                        m_model.get_fieldmap()[models::MHD::ID],
                                        "density",
                                        m_config_map,
                                        m_par_env,
                                        is_reference);

  m_conservativity_check.register_value(m_U,
                                        0,
                                        num_octants,
                                        m_mesh_map->orchard_keys(),
                                        m_model.get_fieldmap()[models::MHD::IE],
                                        "total_energy",
                                        m_config_map,
                                        m_par_env,
                                        is_reference);

  m_conservativity_check.register_value(m_U,
                                        0,
                                        num_octants,
                                        m_mesh_map->orchard_keys(),
                                        m_model.get_fieldmap()[models::MHD::IU],
                                        "rho_u",
                                        m_config_map,
                                        m_par_env,
                                        is_reference);

  m_conservativity_check.register_value(m_U,
                                        0,
                                        num_octants,
                                        m_mesh_map->orchard_keys(),
                                        m_model.get_fieldmap()[models::MHD::IV],
                                        "rho_v",
                                        m_config_map,
                                        m_par_env,
                                        is_reference);

  if constexpr (dim == 3)
  {
    m_conservativity_check.register_value(m_U,
                                          0,
                                          num_octants,
                                          m_mesh_map->orchard_keys(),
                                          m_model.get_fieldmap()[models::MHD::IW],
                                          "rho_w",
                                          m_config_map,
                                          m_par_env,
                                          is_reference);
  }

} // SolverGodunovMHD<dim, device_t>::register_volume_integrals

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::print_conservativity_check_report() const
{

  m_conservativity_check.print_report(m_par_env);

} // SolverGodunovMHD<dim, device_t>::print_conservativity_check_report

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
uint64_t
SolverGodunovMHD<dim, device_t>::total_mem_size_in_bytes()
{
  uint64_t total = 0;

  // conservative variables at begin and end of time step
  total += m_U.allocated_size_in_bytes();
  total += m_U2.allocated_size_in_bytes();

  if (m_time_integrator != +TimeIntegrator::HANCOCK)
  {
    total += m_U_RK.allocated_size_in_bytes();
  }

  // memory footprint on implementation detail
  total += m_godunov_implem->total_mem_size_in_bytes();

#ifdef KALYPSSO_CORE_USE_MPI
  total += m_mesh_ghosts_exchanger.allocated_size_in_bytes();
#endif

  return total;
} // SolverGodunovMHD<dim, device_t>::total_mem_size_in_bytes

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
auto
SolverGodunovMHD<dim, device_t>::get_derived_quantity(DERIVED_QUANTITY derived_quantity)
  -> DataArrayBlock_t
{
  const auto & fm = m_model.get_fieldmap();
  const auto   local_num_quadrants =
    static_cast<int64_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants());

  return ComputeDerivedQuantities<dim, device_t>::run(
    m_U, m_Bface, fm, derived_quantity, m_mhd_settings, 0, local_num_quadrants);
} // SolverGodunovMHD<dim, device_t>::get_derived_quantity

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
auto
SolverGodunovMHD<dim, device_t>::get_derived_quantity_on_host(DERIVED_QUANTITY derived_quantity)
  -> DataArrayBlockHost_t
{
  const auto data = get_derived_quantity(derived_quantity);

  const auto data_host = DataArrayBlock_t::create_host_mirror_view_and_copy(data);

  return data_host;

} // SolverGodunovMHD<dim, device_t>::get_derived_quantity_on_host

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::save_solution_hdf5([[maybe_unused]] bool pure_checkpoint)
{

#ifdef KALYPSSO_CORE_USE_HDF5

  // retrieve available / allowed names: fieldManager, and field map (fm)
  const auto & fm = m_model.get_fieldmap();

  // a map containing ID and name of the variable to write
  const auto id2names = m_model.get_id2names_map();

  // get list of variables to write ?
  const auto write_variables =
    m_config_map.getStringVector("output", "write_variables", std::vector<std::string>{});

  // prepare output filename
  std::string outputDir = m_config_map.getString("output", "outputDir", "./");

  [[maybe_unused]] uint64_t total_num_bytes = 0;

  const auto local_num_quadrants =
    static_cast<int64_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants());

  // actual writing (cell data)
  {
    // copy device data to host (note: Uhost has been resized in resize_solver_data)
    Kokkos::deep_copy(m_Uhost.logical_view(), m_U.logical_view());

    m_hdf5_writer->set_block_mode();
    m_hdf5_writer->update_mesh_info();

    // open the new file and write our stuff
    std::string basename = this->output_basename();

    HostTimer io_timer;
    io_timer.start();
    m_hdf5_writer->open(basename, outputDir);
    total_num_bytes += m_hdf5_writer->write_header(m_t);
    total_num_bytes += m_hdf5_writer->write_amr_metadata(m_mesh_map->orchard_keys_host());

    // write time and number of time steps
    m_hdf5_writer->write_scalar_attribute("run", "time", m_t);
    m_hdf5_writer->write_scalar_attribute("run", "iteration", m_iteration);

    // write user data (all enabled field)
    for (auto & iter : id2names)
    {
      auto varId = static_cast<typename models::MHD::VarId>(iter.first);

      // get variables string name
      const auto varName = id2names.at(varId);

      if (is_present(write_variables, varName) or should_do_checkpoint())
      {
        if (varId != models::MHD::IBX and varId != models::MHD::IBY and varId != models::MHD::IBZ)
        {
          total_num_bytes += m_hdf5_writer->write_quadrant_attribute(
            m_Uhost, fm[varId], varName, 0, local_num_quadrants);
        }
      }

    } // end for iter

    if (should_do_checkpoint())
    {
      // save Bx, By, Bz all at once as face centered variables
      total_num_bytes += m_hdf5_writer->write_quadrant_attribute(m_Bface, "magnetic_field");
    }

    if (!pure_checkpoint)
    {
      if (is_present(write_variables, std::string{ "Bx" }))
      {
        const auto Bdata = FaceDataArrayBlock_t::to_DataArrayBlockCentered(m_Bface, IX);
        const auto Bdata_host = DataArrayBlock_t::create_host_mirror_view_and_copy(Bdata);
        total_num_bytes +=
          m_hdf5_writer->write_quadrant_attribute(Bdata_host, 0, "Bx", 0, local_num_quadrants);
      }

      if (is_present(write_variables, std::string{ "By" }))
      {
        const auto Bdata = FaceDataArrayBlock_t::to_DataArrayBlockCentered(m_Bface, IY);
        const auto Bdata_host = DataArrayBlock_t::create_host_mirror_view_and_copy(Bdata);
        total_num_bytes +=
          m_hdf5_writer->write_quadrant_attribute(Bdata_host, 0, "By", 0, local_num_quadrants);
      }

      if (is_present(write_variables, std::string{ "Bz" }))
      {
        const auto Bdata = FaceDataArrayBlock_t::to_DataArrayBlockCentered(m_Bface, IZ);
        const auto Bdata_host = DataArrayBlock_t::create_host_mirror_view_and_copy(Bdata);
        total_num_bytes +=
          m_hdf5_writer->write_quadrant_attribute(Bdata_host, 0, "Bz", 0, local_num_quadrants);
      }

      if (is_present(write_variables, std::string{ "divB" }))
      {
        const auto divBdata =
          FaceDataArrayBlock_t::compute_divergence(m_Bface, m_mesh_map->orchard_keys());
        const auto divBdata_host = DataArrayBlock_t::create_host_mirror_view_and_copy(divBdata);

        total_num_bytes +=
          m_hdf5_writer->write_quadrant_attribute(divBdata_host, 0, "divB", 0, local_num_quadrants);
      }

      // deal with derived quantities
      for (DERIVED_QUANTITY derived_var : DERIVED_QUANTITY::_values())
      {
        auto name = std::string{ derived_var._to_string() };
        // get lower case name
        std::for_each(
          name.begin(), name.end(), [](char & c) { c = static_cast<char>(std::tolower(c)); });

        if (is_present(write_variables, name))
        {
          const auto derived_quantity_host = get_derived_quantity_on_host(derived_var);

          total_num_bytes += m_hdf5_writer->write_quadrant_attribute(
            derived_quantity_host, 0, name, 0, local_num_quadrants);
        }
      }
    } // end if !pure_checkpoint

    // close the file
    m_hdf5_writer->write_footer();
    m_hdf5_writer->close();
    io_timer.stop();

    // print hard drive bandwidth
    const auto hdf5_verbose = m_config_map.getBool("output", "hdf5_verbose", false);
    if (hdf5_verbose)
    {
      // compute hard-drive bandwidth
      uint64_t global_num_bytes = 0;
#  ifdef KALYPSSO_CORE_USE_MPI
      m_par_env.comm().template MPI_Reduce<MpiComm::SUM>(&total_num_bytes, &global_num_bytes, 1, 0);
#  else
      global_num_bytes = total_num_bytes;
#  endif // KALYPSSO_CORE_USE_MPI

      if (m_par_env.rank() == 0)
      {
        const auto write_bw = static_cast<double>(global_num_bytes) / io_timer.elapsed() * 1e-9;
        std::cout << "########################################################\n";
        std::cout << "################### HDF5 bandwidth #####################\n";
        std::cout << "########################################################\n";
        std::cout << "Total number of bytes written : " << global_num_bytes << " in "
                  << io_timer.elapsed() << "seconds\n";
        std::cout << "Write bandwidth : " << write_bw << " GBytes/s\n";
      }
    }
  }

  // writing leaf data (optional)
  const auto write_leaf_data = m_config_map.getBool("output", "enable_writing_leaf_data", false);
  if (write_leaf_data)
  {
    const auto write_conformal_status =
      m_config_map.getBool("output", "write_conformal_status", false);

    m_hdf5_writer->set_leaf_mode();
    m_hdf5_writer->set_write_mesh_info(true);

    const std::string outputPrefix = m_config_map.getString("output", "outputPrefix", "output");
    const std::string basename = outputPrefix + "_quad_" + this->output_time_suffix();

    m_hdf5_writer->open(basename, outputDir);
    m_hdf5_writer->write_header(m_t);

    if (write_conformal_status)
    {
      m_hdf5_writer->write_quadrant_leaf_scalar(m_mesh_map->conformal_status_host(),
                                                "conformal_status");
    }

    m_hdf5_writer->write_footer();
    m_hdf5_writer->close();
  }

#else

  if (m_par_env.rank() == 0)
    std::cerr << "You need to re-run cmake and enable HDF5 to have HDF5 output available. Also set "
                 "hdf5_enabled variable to true in the input parameter file for the run.\n";

#endif // KALYPSSO_CORE_USE_HDF5

} // SolverGodunovMHD<dim, device_t>::save_solution_hdf5

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::synchronize_mpi_ghost_data(
  [[maybe_unused]] DataArrayBlock_t const &     Udata,
  [[maybe_unused]] FaceDataArrayBlock_t const & Bdata)
{

#ifdef KALYPSSO_CORE_USE_MPI

  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, AMR_CYCLE_SYNC_MPI_GHOSTS);

  m_mesh_ghosts_exchanger.exchange(Udata);
  m_mesh_ghosts_exchanger.exchange(Bdata);

#endif // KALYPSSO_CORE_USE_MPI

} // SolverGodunovMHD<dim, device_t>::synchronize_mpi_ghost_data

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::fill_outside_quadrants(DataArrayBlock_t const &     Udata,
                                                        FaceDataArrayBlock_t const & Bdata)
{

  // assert data has expected size (in terms of number of octants/quadrants)
  assertm(m_U.num_quadrants() == m_mesh_map->get_amr_mesh_info().local_num_quadrants_total(),
          "[SolverGodunovMHD::fill_outside_quadrants] Input array has the wrong size.");
  assertm(m_Bface.num_quadrants() == m_mesh_map->get_amr_mesh_info().local_num_quadrants_total(),
          "[SolverGodunovMHD::fill_outside_quadrants] Input array has the wrong size.");

  // apply border condition, this will be a no-op when mesh is periodic
  FillOutsideCellFunctor<dim, device_t>::apply(Udata,
                                               Bdata,
                                               m_mesh_map->get_amr_mesh_info(),
                                               m_mesh_map->orchard_keys(),
                                               m_mesh_map->hashmap(),
                                               m_model.get_fieldmap(),
                                               m_config_map,
                                               m_par_env);

} // SolverGodunovMHD<dim, device_t>::fill_outside_quadrants

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::mark_cells()
{

  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, AMR_CYCLE_MARK_CELLS);

  /*
   * coarsen threshold must be smaller than refine threshold (here we take refine threshold divided
   * by 8)
   */

  m_amr_context.reset_amrflags();

  // which refine criterion to use ?
  auto const refine_threshold = m_config_map.getReal("amr", "epsilon_refine", KALYPSSO_NUM(0.01));
  auto const coarsen_threshold =
    m_config_map.getReal("amr", "epsilon_coarsen", KALYPSSO_NUM(0.00125));
  auto const epsilon_lohner = m_config_map.getReal("amr", "epsilon_lohner", KALYPSSO_NUM(0.02));

  auto const names = m_config_map.getStringVector(
    "amr", "variable_used_to_compute_refine_criterion", std::vector<std::string>{ "rho" });

  for (auto const & name : names)
  {
    const auto maybe_derived_quantity = DERIVED_QUANTITY::_from_string_nocase_nothrow(name.c_str());
    if (*maybe_derived_quantity)
    {
      const int           ivar_to_refine = 0;
      RefineIndicatorData refine_params{ static_cast<int>(m_params.level_min),
                                         static_cast<int>(m_params.level_max),
                                         RefineIndicatorData::get_indicator(m_config_map),
                                         refine_threshold,
                                         coarsen_threshold,
                                         ivar_to_refine,
                                         epsilon_lohner };

      const auto & fm = m_model.get_fieldmap();

      const auto local_num_quadrants =
        static_cast<int64_t>(m_mesh_map->get_amr_mesh_info().local_num_quadrants());

      // compute derived quantity
      const auto derived_quantity = ComputeDerivedQuantities<dim, device_t>::run(
        m_U, m_Bface, fm, name, m_mhd_settings, 0, local_num_quadrants);

      ComputeRefineFlags<dim, device_t>::run(m_mesh_map->hashmap(),
                                             m_mesh_map->orchard_keys(),
                                             m_mesh_map->get_amr_mesh_info().local_num_quadrants(),
                                             m_brick_sizes,
                                             m_is_brick_periodic,
                                             derived_quantity,
                                             m_amr_context.m_amrflags_d,
                                             refine_params);
    }
    else
    {
      // retrieve which variable is used to compute refine criterion
      int const ivar_to_refine = [this, name]() {
        // a map containing ID and name of each variable of the model (rho, ...)
        auto const id2names = m_model.get_id2names_map();
        for (auto & iter : id2names)
        {
          auto const varId = static_cast<typename models::MHD::VarId>(iter.first);
          auto const varName = id2names.at(varId);
          if (varName == name)
            return varId;
        }
        // default value (density, aka rho)
        return models::MHD::ID;
      }();

      RefineIndicatorData refine_params{ static_cast<int>(m_params.level_min),
                                         static_cast<int>(m_params.level_max),
                                         RefineIndicatorData::get_indicator(m_config_map),
                                         refine_threshold,
                                         coarsen_threshold,
                                         ivar_to_refine,
                                         epsilon_lohner };

      ComputeRefineFlags<dim, device_t>::run(m_mesh_map->hashmap(),
                                             m_mesh_map->orchard_keys(),
                                             m_mesh_map->get_amr_mesh_info().local_num_quadrants(),
                                             m_brick_sizes,
                                             m_is_brick_periodic,
                                             m_U,
                                             m_amr_context.m_amrflags_d,
                                             refine_params);
    }
  }

  // amr context will adapt mesh on CPU, so we need flags on host
  Kokkos::deep_copy(m_amr_context.m_amrflags_h, m_amr_context.m_amrflags_d);

} // SolverGodunovMHD<dim, device_t>::mark_cells

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::adapt_mesh()
{

  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, AMR_CYCLE_ADAPT_MESH);

  //
  // apply AMR cycle: refine + coarsen + 2:1 balance
  //
  [[maybe_unused]] const auto changed = m_amr_context.adapt_mesh(m_amr_mesh->forest());
  // KALYPSSO_INFO("Mesh changed ? {}", changed);

  // update orchard key, hashmap, etc....
  // const auto do_reset_ghosts = true;
  update_mesh(changed);

} // SolverGodunovMHD<dim, device_t>::adapt_mesh

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
SolverGodunovMHD<dim, device_t>::map_userdata_after_adapt(amr_hashmap_t amr_hashmap_before_adapt)
{

  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, AMR_CYCLE_USERDATA_REMAP);

  // clone the AMR orchard keys of the old mesh.
  //
  // they are needed in case, one wants to compute stencil computation over the old map
  // this happens when doing prolongation with linear extrapolation using limited slopes
  // (limited slopes require stencil computations).
  auto orchard_keys_device_old = m_mesh_map->orchard_keys_clone();

  // update the AMR keys for the new mesh
  m_mesh_map->update_orchard_keys(m_amr_mesh->forest(), m_amr_mesh->ghost());

  UserDataRemapper<dim, device_t> userdataRemapper(
    amr_hashmap_before_adapt,
    m_mesh_map->orchard_keys(),
    orchard_keys_device_old,
    m_mesh_map->get_amr_mesh_info().local_num_quadrants(),
    m_block_sizes,
    m_config_map);

  ExecutionSpace exec_space;

  // remap cell-centered data from m_U to m_U2
  {
    m_U2.resize_and_reset(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

    userdataRemapper.remap_block_data(exec_space, m_U, m_U2);

    // swap U2 and U, so that most up to data data are always in U
    my_swap(m_U, m_U2);

    m_U2.resize_and_reset(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());
    Kokkos::deep_copy(exec_space, m_U2.logical_view(), m_U.logical_view());

    if (m_time_integrator == +TimeIntegrator::RK2_SSP)
    {
      m_U_RK.resize_and_reset(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());
    }
  }

  // remap face-center data from m_Bface to m_Bface2
  {
    m_Bface2.resize_and_reset(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());

    userdataRemapper.remap_block_data(exec_space, m_Bface, m_Bface2);

    // swap Bface2 and Bface, so that most up to data data are always in Bface
    my_swap(m_Bface, m_Bface2);

    m_Bface2.resize_and_reset(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());
    Kokkos::deep_copy(exec_space, m_Bface2.logical_view(), m_Bface.logical_view());

    if (m_time_integrator == +TimeIntegrator::RK2_SSP)
    {
      m_Bface_RK.resize_and_reset(m_mesh_map->get_amr_mesh_info().local_num_quadrants_total());
    }
  }

#ifdef KALYPSSO_CORE_ENABLE_DEBUG
  KALYPSSO_INFO("===============================================================================");
  KALYPSSO_INFO("[Remap magnetic field] Checking flux continuity at block borders at time step {}",
                m_iteration);
  KALYPSSO_INFO("===============================================================================");

  synchronize_mpi_ghost_data(m_U, m_Bface);
  fill_outside_quadrants(m_U, m_Bface);

  // check that normal magnetic field component at block border is "continuous"
  core::CheckFaceBorderCompatibility<dim, device_t>::apply(m_Bface,
                                                           m_mesh_map->hashmap(),
                                                           m_mesh_map->orchard_keys(),
                                                           m_mesh_map->get_amr_mesh_info(),
                                                           m_config_map,
                                                           m_brick_sizes,
                                                           m_is_brick_periodic,
                                                           m_par_env);
  KALYPSSO_INFO("===============================================================================");
  KALYPSSO_INFO("[Remap magnetic field] Checking flux continuity at block borders at time step {}",
                m_iteration);
  KALYPSSO_INFO("===============================================================================");
#endif

  Kokkos::fence();

} // SolverGodunovMHD<dim, device_t>::map_userdata_after_adapt

// explicit template instantiation
template class SolverGodunovMHD<2, kalypsso::DefaultDevice>;
template class SolverGodunovMHD<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
