// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV0.h
 *
 * Godunov time integration implementation detail version v0.
 */
#ifndef KALYPSSO_GODUNOV_MHD_GODUNOV_IMPLEM_V0_H_
#define KALYPSSO_GODUNOV_MHD_GODUNOV_IMPLEM_V0_H_

#include <godunov_mhd_ct/scheme/GodunovImplemBase.h>

#include <kalypsso/core/utils_block.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{
/**
 * \class GodunovImplemV0
 *
 * version 0 features:
 *
 * - do compute fluxes and store them;
 * - in a separate kokkos functor perform update  (no atomic memory operations required),
 *  - the sub-domain decomposition is a bit more complex to do (so it is not currently
 *    implemented here);
 * - all intermediate ghosted block array must be sized upon the total number of
 *   local quadrants (could be interesting to monitor memory footprint during the run
 *   and compared with implem version 1).
 *
 * \sa GodunovImplemV0
 */
template <size_t dim, typename device_t>
class GodunovImplemV0 : public GodunovImplemBase<dim, device_t>
{
public:
  using GodunovImplemBase_t = GodunovImplemBase<dim, device_t>;
  using DataArrayBlock_t = typename GodunovImplemBase_t::DataArrayBlock_t;
  using DataArrayGhostedBlock_t = typename GodunovImplemBase_t::DataArrayGhostedBlock_t;
  using FaceDataArrayBlock_t = typename GodunovImplemBase_t::FaceDataArrayBlock_t;


  GodunovImplemV0(ParallelEnv const &      par_env,
                  HydroParams const &      params,
                  ConfigMap const &        config_map,
                  ProfilingManager &       profiling_manager,
                  AMRmesh<dim> const &     amr_mesh,
                  MeshMap<dim, device_t> & mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
                  ,
                  MeshGhostsExchanger<dim, real_t, device_t> & mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
                  )
    : GodunovImplemBase_t(par_env,
                          params,
                          config_map,
                          profiling_manager,
                          amr_mesh,
                          mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
                          ,
                          mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
                          )
    , m_Q_ghosted(this->m_block_sizes,
                  this->m_block_sizes + 2 * 2,
                  get_shift<dim>(-2),
                  "Q_ghosted (owned and ghosts)",
                  this->m_nbvar_mhd_face,
                  0)
    , m_Q2_ghosted(this->m_block_sizes,
                   this->m_block_sizes + 2 * 1,
                   get_shift<dim>(-1),
                   "Q2_ghosted (owned and ghosts)",
                   this->m_nbvar_mhd,
                   0)
    , m_Q_ghosted_mg(this->m_block_sizes,
                     this->m_block_sizes + 2 * 2,
                     get_shift<dim>(-2),
                     "Q_ghosted_mg",
                     this->m_nbvar_mhd_face,
                     0)
    , m_Bface_ghosted("Bface_ghosted", this->m_block_sizes, 2, 0)
    , m_Slopes_x(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_x",
                 this->m_nbvar_mhd,
                 0)
    , m_Slopes_y(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_y",
                 this->m_nbvar_mhd,
                 0)
    , m_Slopes_z(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_z",
                 this->m_nbvar_mhd,
                 0)
    , m_Fluxes("Fluxes", get_flux_block_sizes<dim>(this->m_block_sizes, IX), this->m_nbvar_mhd, 0)
    , m_elec_field(this->m_block_sizes,
                   this->m_block_sizes + 2 * 2,
                   get_shift<dim>(-2),
                   "electric_field",
                   dim == 2 ? 1 : 3,
                   0)
    , m_sFaceMag(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "sFaceMag",
                 dim == 2 ? 2 : 3,
                 0)
    , m_emf("EMF", this->m_block_sizes + 1, dim == 2 ? 1 : 3, 0)
    , m_emf2("EMF2", this->m_block_sizes + 1, dim == 2 ? 1 : 3, 0)
  {} // GodunovImplemV0

  // destructor
  ~GodunovImplemV0() = default;

  /**
   * Static creation method called by the solver factory.
   */
  static GodunovImplemBase_t *
  create(ParallelEnv const &      par_env,
         HydroParams const &      params,
         ConfigMap const &        config_map,
         ProfilingManager &       profiling_manager,
         AMRmesh<dim> const &     amr_mesh,
         MeshMap<dim, device_t> & mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
         ,
         MeshGhostsExchanger<dim, real_t, device_t> & mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
  )
  {
#ifdef KALYPSSO_CORE_USE_MPI
    GodunovImplemV0<dim, device_t> * impl = new GodunovImplemV0<dim, device_t>(
      par_env, params, config_map, profiling_manager, amr_mesh, mesh_map, mesh_ghosts_exchanger);
#else
    GodunovImplemV0<dim, device_t> * impl = new GodunovImplemV0<dim, device_t>(
      par_env, params, config_map, profiling_manager, amr_mesh, mesh_map);
#endif // KALYPSSO_CORE_USE_MPI

    return impl;
  }

  /**
   * Public interface
   */

  //! resize only auxiliary data array (implementation dependent)
  void
  resize_auxiliary_data() override;

  //! memory footprint monitoring
  uint64_t
  total_mem_size_in_bytes() override;

  void
  do_time_step(DataArrayBlock_t const &     U,
               DataArrayBlock_t const &     U2,
               FaceDataArrayBlock_t const & Bface,
               FaceDataArrayBlock_t const & Bface2,
               real_t                       dt) override;

private:
  /*
   * ghosted block arrays used for piece wise computation
   */
  //! hydrodynamics primitive - ghosted block - number of octants : owned + ghost + ghost outside
  DataArrayGhostedBlock_t m_Q_ghosted;

  //! hydrodynamics primitive at time t_{n+1/2} - ghosted block
  //! number of octants : owned + ghost + ghost outside
  DataArrayGhostedBlock_t m_Q2_ghosted;

  //! hydrodynamics primitive - ghosted block - number of octants: MPI mirrors + MPI ghosts
  //! \note the suffix _mg is meant to tell the developer that this array "lives" on
  //! mirror (m) and ghosts(g)
  DataArrayGhostedBlock_t m_Q_ghosted_mg;

  //! magnetic field (face-centered) - ghosted block with ghostwidth=2 - number of octants : owned +
  //! ghost + ghost outside
  FaceDataArrayBlock_t m_Bface_ghosted;

  //! slopes along X dir - ghosted block array of octant's block data - owned + ghosts
  DataArrayGhostedBlock_t m_Slopes_x;

  //! slopes along Y dir - ghosted block array of octant's block data - owned + ghosts
  DataArrayGhostedBlock_t m_Slopes_y;

  //! slopes along Z dir - ghosted block array of octant's block data - owned + ghosts
  DataArrayGhostedBlock_t m_Slopes_z;

  //! Temporary buffer used to stored hydrodynamics flux (owned + ghost + outside block).
  //! It will be reshaped as needed to store either fluxes along X, Y or Z direction.
  DataArrayBlock_t m_Fluxes;

  //! electric field - ghosted block array of octant's block data
  //! owned + ghosts
  //! ghost width of 2
  //! electric field has only one component in 2D (Ez), and 3 components in 3D (Ex,Ey,Ez)
  DataArrayGhostedBlock_t m_elec_field;

  //! source term face magnetic field - ghosted block array of octant's block data
  //! owned + ghost
  //! source term for face magnetic field has 2 components in 2D, and 3 components in 3D
  DataArrayGhostedBlock_t m_sFaceMag;

  //! electromotive forces (for magnetic field update).
  //! - in 2d, there is only one emf (only edge along Z are contributing to update Bx,By magnetic
  //! field components)
  //! - in 3d, there 3 emf (one per type of edge)
  DataArrayBlock_t m_emf;
  DataArrayBlock_t m_emf2;

  //! convert naked Bface block array into ghosted block array
  void
  fill_Bface_ghosted(FaceDataArrayBlock_t const & Bface);

  //! Convert conservative variables to primitive variables in mirror quadrants.
  //! Fills m_Q_ghosted_mg
  //!
  //! \param[in] U conservative variables array (owned + ghosts)
  //! \param[in] Bface magnetic field (owned + ghosts)
  //!
  void
  convert_to_primitives_in_mirror_quads(DataArrayBlock_t const &     U,
                                        FaceDataArrayBlock_t const & Bface);

  //! Convert conservative variables to primitive variables in owned octants + copy ghost octants
  //!
  //! \param[in] U conservative variables (owned + ghosts)
  //! \param[in] Bface magnetic field (owned + ghosts)
  void
  convert_to_primitives(DataArrayBlock_t const & U, FaceDataArrayBlock_t const & Bface);

  //! compute limited slopes in owned and ghosts quadrants
  void
  compute_limited_slopes_in_owned_and_ghosts();

  //! compute primitive variables time predictor at t_{n+1/2}
  void
  compute_primitives_predictor(real_t dt);

  //! Compute hydro fluxes in all (owned and ghosts) quadrants.
  void
  compute_fluxes_and_store_in_owned_and_ghosts(real_t dt, int direction);

  //! compute viscous fluxes in all (owned and ghosts) quadrants.
  void
  compute_viscous_fluxes_and_store_in_owned_and_ghosts(real_t dt, int direction);

  //! Update conservative variable in owned quadrants.
  void
  read_fluxes_and_update_in_owned(DataArrayBlock_t const &     u_out,
                                  FaceDataArrayBlock_t const & b_out,
                                  real_t                       dt,
                                  int                          direction);

  //! compute electric field in owned and ghost octants
  void
  compute_elec_field_in_owned_and_ghosts();

  //! compute face magnetic field source term
  void
  compute_sFaceMag_in_owned_and_ghosts(real_t dt);

  //! perform magnetic update (constraint transport).
  void
  compute_emf_and_store_in_owned_and_ghosts(real_t dt);

  //! Update magnetic field.
  void
  read_emf_and_update_in_owned(FaceDataArrayBlock_t const & b_out);

}; // class GodunovImplemV0

extern template class GodunovImplemV0<2, kalypsso::DefaultDevice>;
extern template class GodunovImplemV0<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_GODUNOV_IMPLEM_V0_H_
