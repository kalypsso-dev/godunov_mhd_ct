// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeElectricFieldFunctor.h for MHD using cell-centered primitive variables.
 */
#ifndef KALYPSSO_GODUNOV_MHD_COMPUTE_ELECTRIC_FIELD_H_
#define KALYPSSO_GODUNOV_MHD_COMPUTE_ELECTRIC_FIELD_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>

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
 * Compute electric field in a range of octants.
 *
 * Electric field components are computed in the middle of each cell edge, necessary to reconstruct
 * Riemann solver input state at cell edges.
 *
 * \note
 * This functor has two modes of operation:
 * - either we compute electric field in a group of owned octants (batch mode, see
 *   TagComputeAllQuad)
 * - either we compute electric field in ghost octants (see TagComputeGhostQuad)
 * In batch mode, primitive variables array and electric field arrays must have the same sizes.
 * When computing in a ghost octant, the electric field array is sized upon the number of ghost
 * octants.
 *
 *
 * \note
 * Remember that each octant containing a (bx,by,bz) cells grid:
 * - the primitive variables ghosted array has block size (bx+2, by+2, bz+2)
 * - electric field must be computed with ghost width of one on the left, and two on the right.
 *
 * So here we use ghostwidth of 2.
 */
template <size_t dim, typename device_t>
class ComputeElectricFieldFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // data array related type aliases
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

private:
  //! a ghosted block array of primitive variables (ghost width is 2) - nb_var_mhd_face variables
  DataArrayGhostedBlock_t m_q;

  //! ghosted block data arrays (ghost width is 2) - electric field
  DataArrayGhostedBlock_t m_elec_field;

  //! field manager
  FieldMap<core::models::MHD> m_fm;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

public:
  //! Tag used for computing only ghost quadrants (in q_mg)
  struct TagComputeGhostQuad
  {};

  //! Tag used for computing only a subset (group) of quadrants
  struct TagComputeAllQuadInGroup
  {};

  // ====================================================================
  // ====================================================================
  /**
   * Compute electric field.
   *
   * \param[in] prim_var : primitives variables (ghosted block, width=2)
   * \param[out] electric_field  (ghosted block width=2)
   * \param[in] iOct_begin is the first octant index to process wrt the primitives variables array
   *            - when computing elec field in a group of owned octant, iOct_begin must be 0
   *               (because primitive var and elec field array are the same size)
   *            - when computing elec field in ghost octants, we don't compute elec field in mirror
   *               octant, only in ghosts, so we need to skip mirror octants
   *
   * \param[in] num_octants is the number of octant to process
   */
  ComputeElectricFieldFunctor(DataArrayGhostedBlock_t const & prim_var,
                              DataArrayGhostedBlock_t const & elec_field,
                              FieldMap<core::models::MHD>     fm,
                              int32_t                         iOct_begin,
                              int32_t                         num_octants);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing electric field in a group of octant
  static void
  apply_on_group(DataArrayGhostedBlock_t const & primitive_vars,
                 DataArrayGhostedBlock_t const & elec_field,
                 FieldMap<core::models::MHD>     fm,
                 int32_t                         num_octants);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing electric field in ghost quadrants.
  static void
  apply_on_ghosts(DataArrayGhostedBlock_t const & primitive_vars,
                  DataArrayGhostedBlock_t const & elec_field,
                  FieldMap<core::models::MHD>     fm,
                  int32_t                         num_mirrors,
                  int32_t                         num_ghosts);

  // ==============================================================
  // ==============================================================
  /**
   * Compute electric field in all cells of ghosted block array.
   *
   * \param[in] cell index integer used to map a cell inside a block
   * \param[in] iOct_in index identify an octant in input array
   * \param[in] iOct_out index identify an octant in output array
   *
   * \note watchout m_q and elec_field have both a ghost width of 2, but electric field can only be
   * filled with 2 ghost cells on the right, and one ghost cell on the left side.
   *
   */
  KOKKOS_INLINE_FUNCTION void
  compute_elec_field(index_t const & cell_index,
                     index_t const   iOct_in,
                     index_t const   iOct_out) const;

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor for computing limited electric field in all quadrants of a group of
   * quadrants.
   */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagComputeAllQuadInGroup const &, const index_t & global_index) const;

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor when computing only ghosts quadrants.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagComputeGhostQuad const &, const index_t & global_index) const;

}; // class ComputeElectricFieldFunctor

// explicit template instantiation
extern template class ComputeElectricFieldFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeElectricFieldFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_COMPUTE_ELECTRIC_FIELD_H_
