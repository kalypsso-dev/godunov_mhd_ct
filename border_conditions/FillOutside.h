// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillOutside.h
 *
 * Implement border conditions (other than periodic) for MHD.
 */
#ifndef KALYPSSO_GODUNOV_MHD_FILLOUTSIDE_H_
#define KALYPSSO_GODUNOV_MHD_FILLOUTSIDE_H_

#include <kalypsso/core/FillOutside_utils.h>

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex
#include <kalypsso/core/mesh_utils.h> // for Face::face_t
#include <kalypsso/core/AMRMeshInfo.h>

#include <kalypsso/core/models/MHD.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/**
 * Define a (better) enum to list supported boundary conditions.
 *
 * - NONE: don't fill this border
 * - PERIODIC : don't fill this border (it will be filled by MPI ghost exchange operations).
 * - ZERO_GRADIENT: it means here that we fill outside cells with a copy of the "last" cell inside
 *                  domain.
 * - WALL : it means here that we fill outside cells with wall boundary condition (normal
 *          velocity is negated, other variable are copied using planar symmetry; at edge we use
 *          axial symmetry and at corners, central symmetry).
 * - BC_CUSTOM : provide a pointwise functor (TODO)
 *
 * \todo should we consider :
 *  - DIRICHLET (constant value at border) ?
 *  - NEUMANN (constant gradient at border) ?
 *
 * \note a (better) enum cannot be defined inside a class (that is a bit annoying)
 */
// clang-format off
BETTER_ENUM(BC_MHD, uint32_t,
  NONE,
  PERIODIC,
  ZERO_GRADIENT,
  WALL,
  CUSTOM
  )
// clang-format on

// ==============================================================================
// ==============================================================================
/**
 * \class FillOutsideCellFunctor
 *
 * A simple prototyping boundary condition class.
 *
 * \note periodic boundary condition is a special case that is not managed here.
 * \note either all sides are periodic, or none.
 *
 * How are filled cells in corner (2d) or edge+corner in 3d ?
 *
 * Here we do something different from godunov_hydro since it may create problem with magnetic
 field.
 * corner and edge octants are only visited once.
 *
 * We apply the border condition in a directionally split manner.
 *
 * --------------------------------
 * |    |         3          |    |
 * |____|____________________|____|
 * |    |                    |    |
 * |    |                    |    |
 * | 1  |                    | 2  |
 * |    |                    |    |
 * |    |                    |    |
 * |____|____________________|____|
 * |    |         4          |    |
 * |____|____________________|____|
 *
 * In you need a border condition with more control of how corner blocks are filled, you need to
 * modify/customize this prototype border condition functor.

 */
template <size_t dim, typename device_t>
class FillOutsideCellFunctor
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  using bc_array_t = BorderConditionsConfig<BC_MHD>::bc_array_t<dim>;

  struct TagFillHydroVar
  {};
  struct TagFillMagField
  {};

  // ==============================================================
  // ==============================================================
  /** Fill outside octant cells.
   *
   * \param[in,out] userdata_out data array which we want to fill the block ghosts cells
   * \param[in] amr_info gives the number of owned, ghost, outside, ghost_outside quads
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] config_map application parameter map
   */
  static void
  apply(DataArrayBlock_t const &            hydro,
        FaceDataArrayBlock_t const &        Bface,
        AMRMeshInfo const &                 amr_mesh_info,
        orchard_key_view_t const &          orchard_keys,
        amr_hashmap_t const &               amr_hashmap,
        FieldMap<core::models::MHD> const & fm,
        ConfigMap const &                   config_map,
        ParallelEnv const &                 par_env);

  // ==============================================================
  // ==============================================================
  /**
   * Range policy functor to sweep outside leaves and fill values according to given border
   * conditions.
   *
   * Just explaining inside/outside block/quadrant mirroring.
   *                       _
   *  ____________________|1|________
   * |                    |2|        |
   * |                               |
   * |                               |
   * |                     _         |
   * |____________________|3|________|
   *                      |4|
   *
   * Reminder:
   *
   * - 1 and 4 are outside quadrants
   * - 2 and 3 are inside  quadrants
   * - 1 and 3 have the same orchard key, except the outside status bits
   * - 2 and 4 have the same orchard key, except the outside status bits
   *
   * Suppose we have orchard key of 1 (say key1) and we want to compute orchard key of quadrant 2:
   *
   * 1. from key1, compute key3 (just change outside status bits)
   * 2. compute outside normal from key3 (which the same as outside normal of key1, since
   * key1==key3 and compute normal does not depend on outside status)
   * 3. use outside normal to compute key4 from key3
   * 4. deduce key2 from key4 (same orchard key, different outside bit status)
   *
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagFillHydroVar const &, const index_t & i_global) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  operator()(TagFillMagField const &, const index_t & i_global) const;

private:
  /**
   * Fill outside octant cells.
   *
   * \param[in,out] hydro cell-center data array which we want to fill the block ghosts cells
   * \param[in,out] mag_field face-center data array which we want to fill the block ghosts cells
   * \param[in] amr_info gives the number of owned, ghost, outside, ghost_outside quads
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   *
   * This constructor only specify BC type for faces. Corners (and edges) will be filled with zero
   * gradient (default).
   *
   *
   */
  FillOutsideCellFunctor(DataArrayBlock_t const &            hydro,
                         FaceDataArrayBlock_t const &        Bface,
                         AMRMeshInfo const &                 amr_mesh_info,
                         orchard_key_view_t const &          orchard_keys,
                         amr_hashmap_t const &               amr_hashmap,
                         FieldMap<core::models::MHD> const & fm,
                         ConfigMap const &                   config_map,
                         ParallelEnv const &                 par_env);

  //! a block data array (no ghosts, sizes= bx,by,bz) of cell-centered hydro variables
  DataArrayBlock_t m_hydro;

  //! a block data array (no ghosts) of face-centered magnetic field components
  FaceDataArrayBlock_t m_Bface;

  //! AMR mesh information (number of owned quadrants, number of ghosts quadrants, number of
  //! outside quads, number of outside ghosts, etc...)
  AMRMeshInfo m_amr_mesh_info;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! Variables to index mapping
  FieldMap<core::models::MHD> m_fm;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_size;

  //! foer each direction, state if mesh is periodic
  const Kokkos::Array<bool, dim> m_brick_periodicity;

  //! border conditions array (one for each face, edge, corner)
  const bc_array_t m_bc_types;

}; // class FillOutsideCellFunctor

// explicit template instantiation
extern template class FillOutsideCellFunctor<2, kalypsso::DefaultDevice>;
extern template class FillOutsideCellFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_FILLOUTSIDE_H_
