// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CorrectEmfAtBlockBorderFunctor.h
 *
 * The purpose of this functor is to modify emf's corresponding to a non-conformal location.
 *
 * Along a given direction (Z in 2D, or X,Y,Z in 3), 4 block shared a common "edge".
 * Depending on the AMR levels of block A,B,C,D there are multiple situations.
 *
 * - if all block are at the same AMR level nothing to be done.
 *
 * - if there is at least one block at a different level than the other, this common edge is
 * non-conformal, we must correct all emf's associated to the non-conformality.
 *
 * In the following example; at all nodes (edges are seen from above) marked with an "x" in the
 * coarse blocks, we must modify emf with data from the fine block
 *
 *  Block A
 *  _______________x
 * |   |   |   |   |
 * |___|___|___|___x
 * |   |   |   |   |    Block B
 * |___|___|___|___x     ________
 * |   |   |   |   |    |_|_|_|_|
 * |___|___|___|___x    |_|_|_|_|
 * |   |   |   |   |    |_|_|_|_|
 * |___|___|___|___x    |_|_|_|_|
 *
 * Block C              Block D
 *  _______________x    x___x___x___x___x
 * |   |   |   |   |    |   |   |   |   |
 * |___|___|___|___|    |___|___|___|___|
 * |   |   |   |   |    |   |   |   |   |
 * |___|___|___|___|    |___|___|___|___|
 * |   |   |   |   |    |   |   |   |   |
 * |___|___|___|___|    |___|___|___|___|
 * |   |   |   |   |    |   |   |   |   |
 * |___|___|___|___|    |___|___|___|___|
 *
 * \note please note that this correction has to be applied in owned blocks not in ghost blocks,
 * because only owned blocks will be used in ReadEmfAndUpdateFunctor to update magnetic field
 * components. For example, if block A and B are ghost block and C, D, owned blocks, then the
 * correction is only applied in C and D non-conformal edges.
 *
 */
#ifndef KALYPSSO_GODUNOV_MHD_CORRECT_EMF_AT_BLOCK_BORDER_FUNCTOR_H_
#define KALYPSSO_GODUNOV_MHD_CORRECT_EMF_AT_BLOCK_BORDER_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>

#include <kalypsso/core/EdgeDataArrayBlock_utils.h>
#include <kalypsso/core/ConformalFaceStatus.h>

#include <type_traits>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Correct emf (electromotive forces) at every block border that is non-conforming (the coarse side
 * must be modified with values from the fine side).
 *
 */
template <size_t dim, typename device_t>
class CorrectEmfAtBlockBorderFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // hashmap related type aliases
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using conformal_status_view_type = conformal_status_view_t<dim, device_t>;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

  using EdgeLocation_t = EdgeLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

private:
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! conformal status view
  conformal_status_view_type m_conformal_status;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  const AMRMeshInfo m_amr_mesh_info;

  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

  //! emf array (in and out); will be modified at non-corformal edges
  DataArrayBlock_t m_emf;
  DataArrayBlock_t m_emf_out;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! edge flat index offsets
  const edge_flat_index_offset_t m_edge_flat_index_offsets;

  //! MPI comm rank from parallel environment (maybe unused, but useful for debug)
  const int m_mpi_comm_rank;

public:
  /**
   * Perform time integration (induction equation) of magnetic field.
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  CorrectEmfAtBlockBorderFunctor(ConfigMap const &                  config_map,
                                 StencilHelper_t const &            stencil_helper,
                                 orchard_key_view_t const &         orchard_keys,
                                 conformal_status_view_type const & conformal_status,
                                 AMRMeshInfo const &                amr_mesh_info,
                                 block_size_t<dim> const &          bSize,
                                 DataArrayBlock_t const &           emf,
                                 DataArrayBlock_t const &           emf_out,
                                 const int                          mpi_comm_rank);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply(ConfigMap const &                  config_map,
        amr_hashmap_t const &              amr_hashmap,
        orchard_key_view_t const &         orchard_keys,
        conformal_status_view_type const & conformal_status,
        AMRMeshInfo const &                amr_mesh_info,
        block_size_t<dim> const &          bSize,
        DataArrayBlock_t const &           emf,
        DataArrayBlock_t const &           emf_out,
        brick_size_t<dim> const &          brick_sizes,
        Kokkos::Array<bool, dim> const &   is_brick_periodic,
        ParallelEnv const &                par_env);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION void
  correct_emf(index_t const & edge_flatindex, int32_t const & iOct) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  correct_emf_3d(const index_t & edge_flatindex, const int32_t & iOct) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // CorrectEmfAtBlockBorderFunctor

// explicit template instantiation
extern template class CorrectEmfAtBlockBorderFunctor<2, kalypsso::DefaultDevice>;
extern template class CorrectEmfAtBlockBorderFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_READ_EMF_AND_UPDATE_FUNCTOR_H_
