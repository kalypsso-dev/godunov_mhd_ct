// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeUpdatedPrimitiveVariablesFunctor.h
 *
 * \brief Perform half a time step (time integration) of the cell-centered primitive variables
 * using a ghosted blocked array.
 */
#ifndef KALYPSSO_GODUNOV_MHD_COMPUTE_UPDATED_PRIMITIVE_VARIABLES_FUNCTOR_H_
#define KALYPSSO_GODUNOV_MHD_COMPUTE_UPDATED_PRIMITIVE_VARIABLES_FUNCTOR_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/orchard_key_base.h>

// hydro utils (conservative versus primitive variable, equation of state, ...)
#include <godunov_mhd_ct/models/MHD.h>
#include <godunov_mhd_ct/models/MHDState.h>
#include <godunov_mhd_ct/models/mhd_utils.h>
#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/GravityField.h>
#include <kalypsso/core/ViscosityParams.h>
#include <kalypsso/core/TimeIntegratorConfig.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute time-update (half time-step) of primitive variables in a range of octants.
 *
 * \note
 * This functor has two modes of operation:
 * - either we compute in a group of owned octants (batch mode, see TagComputeAllQuad)
 * - either we compute in ghost octants (see TagComputeGhostQuad)
 *
 * In batch mode, primitive variables array and slopes arrays must have the same size (number of
 * octants). When computing in ghosts octant, the slopes array are sized upon the number of ghost
 * octants.
 *
 * \note
 * For each octant containing a (bx,by,bz) cells grid, primitive variables are updated in a
 * (bx+1,by+1,bz+1) block of cells (original block equipped with one ghost cell all around); the
 * primitive variables ghosted array has block size (bx+2, by+2, bz+2).
 *
 * \note
 * The updated primitive variables are stored in a ghosted block array (ghostwidth = 1), to be
 * reused later by the Godunov functor (to compute hydro fluxes).
 *
 */
template <size_t dim, typename device_t>
class ComputeUpdatedPrimitiveVariablesFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  // using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using MHD = models::MHD;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! a ghosted block array of primitive variables (ghost width is 2) - nb_var_mhd_face variables
  DataArrayGhostedBlock_t m_q;

  //! a ghosted block array of updated primitive variables (ghost width is 1) - nb_var_mhd_face
  //! variables
  DataArrayGhostedBlock_t m_q2;

  //! ghosted block data arrays (ghost width is 1) - slopes along X - nb_var_mhd variables
  DataArrayGhostedBlock_t m_slopes_x;

  //! ghosted block data arrays (ghost width is 1) - slopes along Y - nb_var_mhd variables
  DataArrayGhostedBlock_t m_slopes_y;

  //! ghosted block data arrays (ghost width is 1) - slopes along Z  - nb_var_mhd variables
  //! only used when dim=3
  DataArrayGhostedBlock_t m_slopes_z;

  //! field manager
  FieldMap<models::MHD> m_fm;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  //! hydro settings (EOS parameters)
  MHDSettings m_mhd_settings;

  //! time step
  const real_t m_dt;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  //! gravity source term enabled ?
  const bool m_gravity_enabled;

  //! uniform gravity field
  const UniformGravityField<dim> m_gravity_field;

  //! viscosity parameters (needed for the Muscl-Hancock predictor)
  const ViscosityParams m_viscosity;

  //! time integrator
  const TimeIntegrator m_time_integrator;

public:
  struct TagComputeGhostQuad
  {};
  struct TagComputeAllQuadInGroup
  {};

  // ====================================================================
  // ====================================================================
  /**
   * Compute limited slopes functor.
   *
   * \param[in] prim_var : primitives variables at t_{n} (ghosted block, width=2)
   * \param[out] updated_prim_var : primitives variables at t_{n+1/2} (ghosted block, width=1)
   * \param[in] slopes_x : limited slopes along x (ghosted block width=1)
   * \param[in] slopes_y : limited slopes along y (ghosted block width=1)
   * \param[in] slopes_z : limited slopes along z (ghosted block width=1) - not used in 2d
   * \param[in] iOct_begin is the first octant index to process wrt the primitives variables array
   *            - when computing slopes in a group of owned octant, iOct_begin must be 0
   *               (because primitive var and slopes array are the same size)
   *            - when computing slopes in ghost octants, we don't compute slopes in mirror octant,
   *              only in ghosts, so we need to skip mirror octants
   *
   * \param[in] num_octants is the number of octant to process
   * \param[in] mhd_settings contains specific useful parameters like slope type
   *
   *
   */
  ComputeUpdatedPrimitiveVariablesFunctor(ConfigMap const &                config_map,
                                          orchard_key_view_t const &       orchard_keys,
                                          DataArrayGhostedBlock_t const &  prim_var,
                                          DataArrayGhostedBlock_t const &  updated_prim_var,
                                          DataArrayGhostedBlock_t const &  slopes_x,
                                          DataArrayGhostedBlock_t const &  slopes_y,
                                          DataArrayGhostedBlock_t const &  slopes_z,
                                          FieldMap<models::MHD>            fm,
                                          int32_t                          iOct_begin,
                                          int32_t                          num_octants,
                                          MHDSettings const &              mhd_settings,
                                          real_t                           dt,
                                          bool                             gravity_enabled,
                                          UniformGravityField<dim> const & gravity_field,
                                          ViscosityParams const &          viscosity,
                                          TimeIntegrator const &           time_integrator);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply_on_group(ConfigMap const &               config_map,
                 orchard_key_view_t const &      orchard_keys,
                 DataArrayGhostedBlock_t const & primitive_vars,
                 DataArrayGhostedBlock_t const & updated_primitive_vars,
                 DataArrayGhostedBlock_t const & slopes_x,
                 DataArrayGhostedBlock_t const & slopes_y,
                 DataArrayGhostedBlock_t const & slopes_z,
                 FieldMap<models::MHD>           fm,
                 int32_t                         num_octants,
                 MHDSettings const &             mhd_settings,
                 ViscosityParams const &         viscosity,
                 real_t                          dt);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in ghost quadrants.
  static void
  apply_on_ghosts(ConfigMap const &               config_map,
                  orchard_key_view_t const &      orchard_keys,
                  DataArrayGhostedBlock_t const & primitive_vars,
                  DataArrayGhostedBlock_t const & updated_primitive_vars,
                  DataArrayGhostedBlock_t const & slopes_x,
                  DataArrayGhostedBlock_t const & slopes_y,
                  DataArrayGhostedBlock_t const & slopes_z,
                  FieldMap<models::MHD>           fm,
                  int32_t                         num_mirrors,
                  int32_t                         num_ghosts,
                  MHDSettings const &             mhd_settings,
                  ViscosityParams const &         viscosity,
                  real_t                          dt);

  // ====================================================================
  // ====================================================================
  /**
   * Add viscous force predictor to reconstructed primitive variables state.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  add_viscous_predictor_2d(real_t &        u,
                           real_t &        v,
                           int32_t const & iq,
                           int32_t const & jq,
                           int32_t const & iOct,
                           real_t const &  dtdx,
                           real_t const &  dtdy,
                           real_t const &  dx,
                           real_t const &  dy) const;

  // ====================================================================
  // ====================================================================
  /**
   * Add viscous force predictor to reconstructed primitive variables state.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  add_viscous_predictor_3d(real_t &        u,
                           real_t &        v,
                           real_t &        w,
                           int32_t const & iq,
                           int32_t const & jq,
                           int32_t const & kq,
                           int32_t const & iOct,
                           real_t const &  dtdx,
                           real_t const &  dtdy,
                           real_t const &  dtdz,
                           real_t const &  dx,
                           real_t const &  dy,
                           real_t const &  dz) const;

  // ==============================================================
  // ==============================================================
  /**
   * Update primitive variables in all cells of ghosted block array (slopes_x, ...)
   *
   * \param[in] cell index integer used to map a cell inside a block
   * \param[in] iOct_in index identify an octant in input array
   * \param[in] iOct_out index identify an octant in output array
   *
   * \note iOct_in may be used to enumerate
   * - a owned local quadrant, in that case iOct_out = iOct_in
   * - a ghost quadrant, in that case iOct_out must be equal to iOct_in - offset where offset is
   * offset to the first ghost quadrant
   *
   * \note watchout index shift m_q has a ghost width of 2, while
   * slopes_x (and other) have a ghost width of 1
   */
  KOKKOS_INLINE_FUNCTION void
  update_primitive_variables(index_t const & cell_index,
                             index_t const   iOct_in,
                             index_t const   iOct_out) const;

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor for computing limited slopes in all quadrants of a group of quadrants.
   */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagComputeAllQuadInGroup const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor when computing only ghosts quadrants.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagComputeGhostQuad const &, const index_t & global_index) const;

}; // class ComputeUpdatedPrimitiveVariablesFunctor

// explicit template instantiation
extern template class ComputeUpdatedPrimitiveVariablesFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeUpdatedPrimitiveVariablesFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_COMPUTE_UPDATED_PRIMITIVE_VARIABLES_FUNCTOR_H_
