// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFaceMagFunctor.h for MHD using cell-centered primitive variables.
 */
#ifndef KALYPSSO_GODUNOV_MHD_COMPUTE_SOURCE_FACE_MAG_H_
#define KALYPSSO_GODUNOV_MHD_COMPUTE_SOURCE_FACE_MAG_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/orchard_key_base.h>

// hydro utils (conservative versus primitive variable, equation of state, ...)
#include <kalypsso/core/models/MHD.h>
#include <kalypsso/core/models/MHDState.h>
#include <kalypsso/core/models/mhd_utils.h>
#include <kalypsso/core/utils_block.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute source term used in state reconstruction on face and edge in a range of octants.
 *
 * \note
 * This functor has two modes of operation:
 * - either we compute source term in a group of owned octants (batch mode, see
 * TagComputeAllQuadInGroup)
 * - either we compute source term in ghost octants (see TagComputeGhostQuad)
 *
 * \note
 * Remember that each octant containing a (bx,by,bz) cells grid:
 * - the primitive variables ghosted array has block size (bx+2, by+2, bz+2)
 * - electric field must be computed with ghost width of one on the left, and two on the right.
 * - source term for magnetic field on face uses a ghostwidth of 1
 *
 * So here we use ghostwidth of 1.
 */
template <size_t dim, typename device_t>
class ComputeSourceFaceMagFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

private:
  //! a ghosted block array of  source term for mag field (ghost width is 1)
  //! 2 components in 2d
  //! 3 components in 3d
  DataArrayGhostedBlock_t m_sFaceMag;

  //! ghosted block data arrays (ghost width is 2) - electric field
  DataArrayGhostedBlock_t m_elec_field;

  //! field manager
  FieldMap<core::models::MHD> m_fm;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  // //! block sizes
  // const block_size_t<dim> m_block_sizes;

  // //! block sizes with ghost (of width 1)
  // const block_size_t<dim> m_ghosted_block_sizes;

  // //! number of cells per leaf block
  // const int32_t m_nbCellsPerLeaf;

  // //! number of cells per leaf block (ghost included)
  // const int32_t m_nbCellsPerGhostedLeaf;

  //! time step
  const real_t m_dt;

  // get geometrical scaling factor (necessary to compute dx => cell size)
  const real_t m_scaling_factor;

  //! list of orchard key of the mesh (necessary to obtain AMR level)
  orchard_key_view_t m_orchard_keys_device;

public:
  struct TagComputeGhostQuad
  {};
  struct TagComputeAllQuadInGroup
  {};

  // ====================================================================
  // ====================================================================
  /**
   * Compute source term for magnetic field 1/2 time step integration.
   *
   * \param[out] sFaceMag : source term for mag field on face (ghosted block, width=1)
   * \param[in] electric_field  (ghosted block width=2)
   * \param[in] iOct_begin is the first octant index to process wrt the primitives variables array
   *            - when computing slopes in a group of owned octant, iOct_begin must be 0
   *               (because primitive var and slopes array are the same size)
   *            - when computing slopes in ghost octants, we don't compute slopes in mirror octant,
   *              only in ghosts, so we need to skip mirror octants
   *
   * \param[in] num_octants is the number of octant to process
   */
  ComputeSourceFaceMagFunctor(DataArrayGhostedBlock_t const & sFaceMag,
                              DataArrayGhostedBlock_t const & elec_field,
                              FieldMap<core::models::MHD>     fm,
                              int32_t                         iOct_begin,
                              int32_t                         num_octants,
                              real_t                          dt,
                              real_t                          scaling_factor,
                              orchard_key_view_t const &      orchard_keys);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing source term in a group of octant
  static void
  apply_on_group(DataArrayGhostedBlock_t const & sFaceMag,
                 DataArrayGhostedBlock_t const & elec_field,
                 FieldMap<core::models::MHD>     fm,
                 int32_t                         num_octants,
                 real_t                          dt,
                 ConfigMap const &               config_map,
                 orchard_key_view_t const &      orchard_keys);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing source term in ghost quadrants.
  static void
  apply_on_ghosts(DataArrayGhostedBlock_t const & sFaceMag,
                  DataArrayGhostedBlock_t const & elec_field,
                  FieldMap<core::models::MHD>     fm,
                  int32_t                         num_mirrors,
                  int32_t                         num_ghosts,
                  real_t                          dt,
                  ConfigMap const &               config_map,
                  orchard_key_view_t const &      orchard_keys);

  // ==============================================================
  // ==============================================================
  /**
   * Compute source term in all cells of ghosted block array.
   *
   * \param[in] cell index integer used to map a cell inside a block
   * \param[in] iOct_in index identify an octant in input array
   * \param[in] iOct_out index identify an octant in output array
   *
   * \note watchout elec_field has a ghost width of 2, and sFaceMag a ghost width of 1
   *
   */
  KOKKOS_INLINE_FUNCTION void
  compute_sFaceMag(index_t const & cell_index,
                   index_t const & iOct_in,
                   index_t const & iOct_out) const;

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor for computing face magnetic source term in all quadrants of a group of
   * owned quadrants.
   */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagComputeAllQuadInGroup const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor for computing face magnetic source only in ghosts quadrants.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagComputeGhostQuad const &, const index_t & global_index) const;

}; // class ComputeSourceFaceMagFunctor

// explicit template instantiation
extern template class ComputeSourceFaceMagFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeSourceFaceMagFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_COMPUTE_SOURCE_FACE_MAG_H_
