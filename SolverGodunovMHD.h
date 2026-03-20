// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SolverGodunovMHD.h
 *
 * Class SolverGodunovMHD definition.
 *
 * Main class for solving compressible MHD with
 * MUSCL-Hancock scheme for 2D/3D + constraint transport.
 */
#ifndef KALYPSSO_GODUNOV_MHD_SOLVER_GODUNOV_MHD_H_
#define KALYPSSO_GODUNOV_MHD_SOLVER_GODUNOV_MHD_H_

#include <cstdio>
#include <cstdbool>
#include <cassert> // assert

// shared
#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/SolverBase.h>
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/models/MHD.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/TimeIntegratorConfig.h>

#include <kalypsso/utils/monitoring/ProfilingManager.h>

// godunov implementation details
#include <godunov_mhd_ct/scheme/GodunovImplemBase.h>
#include <godunov_mhd_ct/scheme/GodunovImplemV0.h>
// #include <godunov_mhd_ct/scheme/GodunovImplemV1.h>

// AMR services
#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshGhostsExchanger.h>
#endif // KALYPSSO_CORE_USE_MPI
#include <kalypsso/core/AMRContext.h>

// border conditions
#include <godunov_mhd_ct/border_conditions/FillOutside.h>

// for IO
#ifdef KALYPSSO_CORE_USE_HDF5
#  include <kalypsso/core/HDF5_Xdmf_Writer.h>
#endif

// monitoring / diagnostic
#include <kalypsso/core/ConservativityCheck.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

/**
 * \class SolverGodunovMHD
 *
 * Main data structure for solving 2D/3D MHD with MUSCL-Hancock + constraint transport.
 *
 * @note:
 * - Hydrodynamics variable (rho, momentum and total energy) are cell-centered
 * - magnetic field variables are face-centered in a component-wise manner, i.e. Bx is centered on
 *   x-faces, By on face-y, etc...
 *
 * @note on implementation: The trade-off explored here is to design the main data array
 * structure to hold block data without ghost cells, but then apply
 * all operators in a piecewise way. More precisely each MPI task
 * allocates memory to hold a fixed amount of block data with
 * ghost cells included and processes the complete list of octree
 * leaves block group by group.
 *
 * Just for clarification:
 * - variables names suffix "_ghost" means variables in the MPI
 *   ghost octants
 * - variables names suffix "_g"     means variables in the block
 *   data ghost cells (from an octant to neighbor octant)
 *
 * \tparam dim is dimension (integer: 2 or 3)
 * \tparam device_t is a kokkos device class (e.g. Kokkos::CudaSpace::device_type)
 */
template <size_t dim, typename device_t>
class SolverGodunovMHD : public kalypsso::SolverBase
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  //! type alias for orchard keys on device
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  //! type alias for orchard keys on host
  using orchard_key_view_host_t = typename orchard_key_base_t<device_t>::view_host_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;

#ifdef KALYPSSO_CORE_USE_HDF5
  using HDF5_Xdmf_Writer_t = HDF5_Xdmf_Writer<dim, device_t>;
#endif

  using ExecutionSpace = typename device_t::execution_space;

  // =================================================================================
  //! constructor
  SolverGodunovMHD(ParallelEnv const & par_env,
                   HydroParams const & params,
                   ConfigMap const &   config_map);
  //! destructor
  virtual ~SolverGodunovMHD();

  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase *
  create(ParallelEnv const & par_env, HydroParams const & params, ConfigMap const & config_map)
  {
    SolverGodunovMHD<dim, device_t> * solver =
      new SolverGodunovMHD<dim, device_t>(par_env, params, config_map);

    return solver;
  }

  /*
   * methods
   */

  //! resize only auxiliary data array
  void
  resize_auxiliary_data();

  //! resize all workspace data array
  void
  resize_solver_data();

  //! update orchard keys by recomputing them from current forest state, and update hashmap
  //!
  //! \param[in] reset_p4est_ghost boolean value to enforce recompute p4est's ghost data structure
  void
  update_mesh(bool reset_p4est_ghost);

  std::string
  solver_name() const override
  {
    if constexpr (dim == 2)
    {
      return "Godunov_MHD_CT_2D";
    }
    else if constexpr (dim == 3)
    {
      return "Godunov_MHD_CT_3D";
    }
    return "unknown_solver";
  };


  //! do_amr_cycle is supposed to be called after
  //! the numerical scheme (godunov_unsplit)
  void
  do_amr_cycle() override;

  //! do_load_balancing can be called less often than
  //! do_amr_cycle
  void
  do_load_balancing() override;

  //! compute time step inside an MPI process, at shared memory level.
  real_t
  compute_dt_local() override;

  //! perform 1 time step (time integration).
  void
  next_iteration_impl() override;

  //! full numerical scheme (time integration from t_n to t_{n+1})
  //!
  //! this is just a wrapper around actual implementation : e.g. \see godunov_unsplit_version0
  void
  godunov_unsplit(real_t dt);

public:
  //! write output file.
  void
  save_solution_impl(bool pure_checkpoint) override;

  //! print monitoring info during run
  void
  print_monitoring_info() override;

  //! print monitoring info at the end (timing, cell-updates/s, etc...)
  void
  print_monitoring_info_final() override;

  //! return a shared pointer to amr_mesh
  std::shared_ptr<AMRmesh<dim>>
  amr_mesh()
  {
    return m_amr_mesh;
  }

  //! return a shared pointer to our MeshMap
  std::shared_ptr<MeshMap<dim, device_t>>
  mesh_map()
  {
    return m_mesh_map;
  }

  ConfigMap const &
  config_map() const
  {
    return this->m_config_map;
  }

  core::models::MHD const &
  model() const
  {
    return m_model;
  }

  MHDSettings const &
  mhd_settings() const
  {
    return m_mhd_settings;
  }

  block_size_t<dim> const &
  block_sizes() const
  {
    return m_block_sizes;
  }

  brick_size_t<dim> const &
  brick_sizes() const
  {
    return m_brick_sizes;
  }

  DataArrayBlock_t
  U()
  {
    return m_U;
  }

  DataArrayBlock_t
  U2()
  {
    return m_U2;
  }

  DataArrayBlockHost_t
  Uhost()
  {
    return m_Uhost;
  }

  FaceDataArrayBlock_t
  Bface()
  {
    return m_Bface;
  }

  //! helper to register volume integrals
  void
  register_volume_integrals(bool is_reference);

  //! print conservativity check report
  void
  print_conservativity_check_report() const;

  //! monitoring total memory allocated on device.
  uint64_t
  total_mem_size_in_bytes();

private:
  //! block sizes as an array without ghost
  block_size_t<dim> m_block_sizes;

  //! p4est brick connectivity sizes
  brick_size_t<dim> m_brick_sizes;

  //! array of bool to tell if mesh is periodic or not
  Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! model
  core::models::MHD m_model;

  //! MHD parameters
  const MHDSettings m_mhd_settings;

  //! The main AMR object
  std::shared_ptr<AMRmesh<dim>> m_amr_mesh;

  //! mesh map is a helper class for creating a hashmap of orchard key to memory address
  std::shared_ptr<MeshMap<dim, device_t>> m_mesh_map;

#ifdef KALYPSSO_CORE_USE_MPI
  //! MPI communications to exchange ghost block userdata
  MeshGhostsExchanger<dim, real_t, device_t> m_mesh_ghosts_exchanger;
#endif // KALYPSSO_CORE_USE_MPI

  //! Perform AMR mesh adaptation (no user data involved here)
  AMRContext<dim, device_t> m_amr_context;

  //! time integrator
  const TimeIntegrator m_time_integrator;

  //! conservativity diagnostic
  core::ConservativityCheck<dim, device_t> m_conservativity_check;

  //! Godunov time integration implementation version
  //!
  //! version 0: do compute fluxes and store them;  in a separate kokkos functor perform update
  //! (no atomics required), but the sub-domain decomposition is more complex to do (so it is not
  //! implemented here); all intermediate ghosted block array must be sized upon the total number of
  //! local quadrants (could be interesting to monitor memory footprint during the run and compared
  //! with implem version 0).
  //!
  //! version 1: variant of version 0, but do a spatial sub-cycling
  //!
  //! version 2: don't store fluxes; just compute flux and use them to perform atomic updates in a
  //! single kokkos functor; we can do a sub-domain decomposition (which help decreasing memory
  //! footprint due to ghosted block array).
  int m_godunov_impl_version;

  GodunovImplemBase<dim, device_t> * m_godunov_implem;

#ifdef KALYPSSO_CORE_USE_MPI
  GodunovImplemBase<dim, device_t> *
  create_godunov_implem(int impl_version)
  {
    if (impl_version == 0)
    {
      GodunovImplemBase<dim, device_t> * impl_ptr =
        GodunovImplemV0<dim, device_t>::create(m_par_env,
                                               m_params,
                                               m_config_map,
                                               m_profiling_mgr,
                                               *m_amr_mesh,
                                               *m_mesh_map,
                                               m_mesh_ghosts_exchanger);
      return impl_ptr;
    }
    else
    {
      Kokkos::abort("Wrong value for input parameter hydro/implementation_version");
    }
    return nullptr;
  } // create_godunov_implem
#else
  GodunovImplemBase<dim, device_t> *
  create_godunov_implem(int impl_version)
  {
    if (impl_version == 0)
    {
      GodunovImplemBase<dim, device_t> * impl_ptr = GodunovImplemV0<dim, device_t>::create(
        m_par_env, m_params, m_config_map, m_profiling_mgr, *m_amr_mesh, *m_mesh_map);
      return impl_ptr;
    }
    else
    {
      Kokkos::abort("Wrong value for input parameter hydro/implementation_version");
    }
    return nullptr;
  } // create_godunov_implem
#endif // KALYPSSO_CORE_USE_MPI

  //
  // user data
  //

  //! hydrodynamics conservative variables arrays at t_n - no ghost
  DataArrayBlock_t m_U;

  //! mirror DataArrayBlock U on host memory space  - no ghost
  DataArrayBlockHost_t m_Uhost;

  //! hydrodynamics conservative variables arrays at t_{n+1} - no ghost
  DataArrayBlock_t m_U2;

  //! hydrodynamics conservative variables arrays only used when using Runge-Kutta time integration
  DataArrayBlock_t m_U_RK;

  //! face centered magnetic field at time t_n
  FaceDataArrayBlock_t m_Bface;

  //! face centered magnetic field at time t_{n+1}
  FaceDataArrayBlock_t m_Bface2;

  //! face centered magneetic field only used when using Runge-Kutta time integration
  FaceDataArrayBlock_t m_Bface_RK;

  /**
   * border condition (other than periodic), specified in the following order: XMIN, XMAX, YMIN,
   * YMAX, ....
   */
  BorderConditionsConfig<BC_MHD>::bc_array_t<dim> m_bc_types;

#ifdef KALYPSSO_CORE_USE_HDF5
  std::shared_ptr<HDF5_Xdmf_Writer_t> m_hdf5_writer;
#endif // KALYPSSO_CORE_USE_HDF5

private:
  //! HDF5 output
  void
  save_solution_hdf5(bool pure_checkpoint);

  /*
   * the following routines are necessary for amr cycle.
   */

  //! synchronize MPI ghost data / only necessary when MPI is activated
  //! this will also do external border only when mesh is periodic
  void
  synchronize_mpi_ghost_data(DataArrayBlock_t Udata, FaceDataArrayBlock_t Bdata);

  //! when mesh is not periodic, the number of outside quadrants is not zero, so we need to fill
  //! outside quadrant user data according to the border condition. This is where we apply border
  //! conditions, when BC is not periodic.
  //!
  //! Just to be clear, let's repeat that array m_U must be sized upon the total number of
  //! octants (owned, ghost and outside).
  void
  fill_outside_quadrants(DataArrayBlock_t Udata, FaceDataArrayBlock_t Bdata);

  //! mark cells for refinement
  void
  mark_cells();

  //! adapt mesh and recompute connectivity
  void
  adapt_mesh();

  //! map data from old U to new U after adapting mesh
  //!
  //! \param[in] amr_hashmap_before_adapt is the amr hashmap before mesh adaptation
  void
  map_userdata_after_adapt(amr_hashmap_t amr_hashmap_before_adapt);

}; // class SolverGodunovMHD

// explicit template instantiation
extern template class SolverGodunovMHD<2, kalypsso::DefaultDevice>;
extern template class SolverGodunovMHD<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_SOLVER_GODUNOV_MHD_H_
