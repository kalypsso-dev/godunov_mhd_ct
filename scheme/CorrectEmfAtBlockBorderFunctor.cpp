// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CorrectEmfAtBlockBorderFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/CorrectEmfAtBlockBorderFunctor.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
template <size_t dim, typename device_t>
CorrectEmfAtBlockBorderFunctor<dim, device_t>::CorrectEmfAtBlockBorderFunctor(
  ConfigMap const &                  config_map,
  StencilHelper_t const &            stencil_helper,
  orchard_key_view_t const &         orchard_keys,
  conformal_status_view_type const & conformal_status,
  AMRMeshInfo const &                amr_mesh_info,
  block_size_t<dim> const &          bSize,
  DataArrayBlock_t const &           emf,
  DataArrayBlock_t const &           emf_out,
  const int                          mpi_comm_rank)
  : m_stencil_helper(stencil_helper)
  , m_orchard_keys_device(orchard_keys)
  , m_conformal_status(conformal_status)
  , m_amr_mesh_info(amr_mesh_info)
  , m_block_sizes(bSize)
  , m_emf(emf)
  , m_emf_out(emf_out)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_edge_flat_index_offsets(compute_edge_flat_index_offsets_emf<dim>(bSize))
  , m_mpi_comm_rank(mpi_comm_rank)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
CorrectEmfAtBlockBorderFunctor<dim, device_t>::apply(
  ConfigMap const &                  config_map,
  amr_hashmap_t const &              amr_hashmap,
  orchard_key_view_t const &         orchard_keys,
  conformal_status_view_type const & conformal_status,
  AMRMeshInfo const &                amr_mesh_info,
  block_size_t<dim> const &          block_sizes,
  DataArrayBlock_t const &           emf,
  DataArrayBlock_t const &           emf_out,
  brick_size_t<dim> const &          brick_sizes,
  Kokkos::Array<bool, dim> const &   is_brick_periodic,
  ParallelEnv const &                par_env)
{
  // Important note: the caller is responsible for providing a flux array with right shape.
  {
    [[maybe_unused]] auto emf_block_sizes = block_sizes + 1;
    assertm(emf_block_sizes == emf.block_size(), "EMF array has incompatible shape.");
  }

  auto stencil_helper =
    StencilHelper_t(amr_hashmap, orchard_keys, block_sizes, brick_sizes, is_brick_periodic);

  CorrectEmfAtBlockBorderFunctor<dim, device_t> functor(config_map,
                                                        stencil_helper,
                                                        orchard_keys,
                                                        conformal_status,
                                                        amr_mesh_info,
                                                        block_sizes,
                                                        emf,
                                                        emf_out,
                                                        par_env.rank());

  // number of owned quadrant x total number of edges (along Z only in 2d, or along all dir in 3D)
  // be careful emf is sized upon block_size + 1 in all direction, but
  // actually it is only required in the edge transverse directions, not in the edge longitudinal
  // direction
  const auto edge_flat_index_offsets = compute_edge_flat_index_offsets_emf<dim>(block_sizes);
  const auto nbIterations = amr_mesh_info.local_num_quadrants() * edge_flat_index_offsets[3];

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::CorrectEmfAtBlockBorderFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CorrectEmfAtBlockBorderFunctor<dim, device_t>::correct_emf(index_t const & edge_flatindex,
                                                           int32_t const & iOct) const
{
  constexpr auto surface_ratio = 1 << (dim - 1);

  auto const edge_indexes =
    edge_flat_index_unravel_emf<dim>(edge_flatindex, m_block_sizes, m_edge_flat_index_offsets);

  if constexpr (dim == 2)
  {
    KOKKOS_ASSERT(edge_indexes[dim] == IZ && "EMF can only be along Z in 2D");
  }
  auto const edge_dir = [](int i) {
    if constexpr (dim == 2)
      return ALONG_Z;
    else if constexpr (dim == 3)
      return i;
  }(edge_indexes[dim]);

  // current block AMR key
  const auto key_cur = m_stencil_helper.key(iOct);

  // create an edge location for current edge's cell (just round down edge index in transverse
  // direction)
  const EdgeLocation_t edge_loc{ edge_indexes, key_cur, iOct, false };
  const auto           level = edge_loc.level();

  // get sibling edge locations
  const auto edge_loc0 = m_stencil_helper.getEdgeSiblingLoc(edge_loc, 0);
  const auto edge_loc1 = m_stencil_helper.getEdgeSiblingLoc(edge_loc, 1);
  const auto edge_loc2 = m_stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

  const auto v =
    get_edge_outside_unit_vector<dim>(edge_indexes, m_block_sizes, EdgeNormalType::DIAGONAL);

  auto norm_v = v[IX] * v[IX] + v[IY] * v[IY];
  if constexpr (dim == 3)
    norm_v += v[IZ] * v[IZ];

  // determine if emf at current edge location must be corrected
  // i.e. if neighbor is at finer AMR level
  //
  // NB: an outside neighbor is always at same AMR level as current block

  if (norm_v > 0)
  {
    bool is_there_at_least_one_coarser_neighbor =
      (edge_loc0.level() < level and edge_loc0.is_valid) or
      (edge_loc1.level() < level and edge_loc1.is_valid) or
      (edge_loc2.level() < level and edge_loc2.is_valid);

    bool is_there_at_least_one_finer_neighbor =
      (edge_loc0.level() > level and edge_loc0.is_valid) or
      (edge_loc1.level() > level and edge_loc1.is_valid) or
      (edge_loc2.level() > level and edge_loc2.is_valid);

    bool all_edges_at_same_level = (edge_loc0.level() == level and edge_loc0.is_valid) and
                                   (edge_loc1.level() == level and edge_loc1.is_valid) and
                                   (edge_loc2.level() == level and edge_loc2.is_valid);

    const bool isNotOutside0 =
      edge_loc0.iOct < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

    const bool isNotOutside1 =
      edge_loc1.iOct < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

    const bool isNotOutside2 =
      edge_loc2.iOct < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

    if (is_there_at_least_one_coarser_neighbor)
    {
      // we average the emf's over neighbor edge locations at current level
      if constexpr (dim == 2)
      {
        auto emf_corrected = m_emf(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_dir, iOct);
        int  count = 1;
        if (edge_loc0.level() == level and edge_loc0.is_valid and isNotOutside0)
        {
          emf_corrected += m_emf(edge_loc0.ijk[IX], edge_loc0.ijk[IY], edge_dir, edge_loc0.iOct);
          count++;
        }
        if (edge_loc1.level() == level and edge_loc1.is_valid and isNotOutside1)
        {
          emf_corrected += m_emf(edge_loc1.ijk[IX], edge_loc1.ijk[IY], edge_dir, edge_loc1.iOct);
          count++;
        }
        if (edge_loc2.level() == level and edge_loc2.is_valid and isNotOutside2)
        {
          emf_corrected += m_emf(edge_loc2.ijk[IX], edge_loc2.ijk[IY], edge_dir, edge_loc2.iOct);
          count++;
        }
        m_emf_out(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_dir, iOct) =
          emf_corrected / static_cast<real_t>(count);
      }
      else if constexpr (dim == 3)
      {
        // we average the emf's over neighbor edge locations at current level
        // TODO : evaluate if we need to average emf's along edge direction (as it is done Athena)
        auto emf_corrected =
          m_emf(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_loc.ijk[IZ], edge_dir, iOct);
        int count = 1;
        if (edge_loc0.level() == level and edge_loc0.is_valid and isNotOutside0)
        {
          emf_corrected += m_emf(
            edge_loc0.ijk[IX], edge_loc0.ijk[IY], edge_loc0.ijk[IZ], edge_dir, edge_loc0.iOct);
          count++;
        }
        if (edge_loc1.level() == level and edge_loc1.is_valid and isNotOutside1)
        {
          emf_corrected += m_emf(
            edge_loc1.ijk[IX], edge_loc1.ijk[IY], edge_loc1.ijk[IZ], edge_dir, edge_loc1.iOct);
          count++;
        }
        if (edge_loc2.level() == level and edge_loc2.is_valid and isNotOutside2)
        {
          emf_corrected += m_emf(
            edge_loc2.ijk[IX], edge_loc2.ijk[IY], edge_loc2.ijk[IZ], edge_dir, edge_loc2.iOct);
          count++;
        }
        m_emf_out(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_loc.ijk[IZ], edge_dir, iOct) =
          emf_corrected / static_cast<real_t>(count);
      }
    }
    else if (is_there_at_least_one_finer_neighbor)
    {
      // we average the emf over neighbor edge locations at finer level
      // in 3d we also average emf's along edge direction

      if constexpr (dim == 2)
      {
        auto emf_corrected = ZERO_F;
        int  count = 0;
        if (edge_loc0.level() > level and edge_loc0.is_valid and isNotOutside0)
        {
          emf_corrected += m_emf(edge_loc0.ijk[IX], edge_loc0.ijk[IY], edge_dir, edge_loc0.iOct);
          count++;
        }
        if (edge_loc1.level() > level and edge_loc1.is_valid and isNotOutside1)
        {
          emf_corrected += m_emf(edge_loc1.ijk[IX], edge_loc1.ijk[IY], edge_dir, edge_loc1.iOct);
          count++;
        }
        if (edge_loc2.level() > level and edge_loc2.is_valid and isNotOutside2)
        {
          emf_corrected += m_emf(edge_loc2.ijk[IX], edge_loc2.ijk[IY], edge_dir, edge_loc2.iOct);
          count++;
        }

        KOKKOS_ASSERT(
          (count > 0) &&
          "Catastrophic error; there should at least one edge neighbor at finer AMR level.");

        m_emf_out(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_dir, iOct) =
          emf_corrected / static_cast<real_t>(count) / surface_ratio;
      }
      else if constexpr (dim == 3)
      {
        // unit vector along edge direction
        const auto v2 = get_unit_vector<dim>(edge_dir);

        auto emf_corrected = ZERO_F;
        int  count = 0;
        if (edge_loc0.level() > level and edge_loc0.is_valid and isNotOutside0)
        {
          // clang-format off
          emf_corrected += m_emf(
            edge_loc0.ijk[IX]       , edge_loc0.ijk[IY]       , edge_loc0.ijk[IZ]       , edge_dir, edge_loc0.iOct);
          emf_corrected += m_emf(
            edge_loc0.ijk[IX]+v2[IX], edge_loc0.ijk[IY]+v2[IY], edge_loc0.ijk[IZ]+v2[IZ], edge_dir, edge_loc0.iOct);
          // clang-format on
          count++;
        }
        if (edge_loc1.level() > level and edge_loc1.is_valid and isNotOutside1)
        {
          // clang-format off
          emf_corrected += m_emf(
            edge_loc1.ijk[IX]       , edge_loc1.ijk[IY]       , edge_loc1.ijk[IZ]       , edge_dir, edge_loc1.iOct);
          emf_corrected += m_emf(
            edge_loc1.ijk[IX]+v2[IX], edge_loc1.ijk[IY]+v2[IY], edge_loc1.ijk[IZ]+v2[IZ], edge_dir, edge_loc1.iOct);
          // clang-format off
          count++;
        }
        if (edge_loc2.level() > level and edge_loc2.is_valid and isNotOutside2)
        {
          // clang-format off
          emf_corrected += m_emf(
            edge_loc2.ijk[IX]       , edge_loc2.ijk[IY]       , edge_loc2.ijk[IZ]       , edge_dir, edge_loc2.iOct);
          emf_corrected += m_emf(
            edge_loc2.ijk[IX]+v2[IX], edge_loc2.ijk[IY]+v2[IY], edge_loc2.ijk[IZ]+v2[IZ], edge_dir, edge_loc2.iOct);
          // clang-format off
          count++;
        }

        KOKKOS_ASSERT(
          (count > 0) &&
          "Catastrophic error; there should at least one edge neighbor at finer AMR level.");

        m_emf_out(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_loc.ijk[IZ], edge_dir, iOct) =
          emf_corrected / static_cast<real_t>(count) / surface_ratio;
      }
    }

    if (all_edges_at_same_level)
    {
      if constexpr (dim == 2)
      {
        auto emf_corrected = ZERO_F;
        int  count = 1;
        emf_corrected += m_emf(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_dir, edge_loc.iOct);

        if (edge_loc0.is_valid and isNotOutside0)
        {
          emf_corrected += m_emf(edge_loc0.ijk[IX], edge_loc0.ijk[IY], edge_dir, edge_loc0.iOct);
          count++;
        }

        if (edge_loc1.is_valid and isNotOutside1)
        {
          emf_corrected += m_emf(edge_loc1.ijk[IX], edge_loc1.ijk[IY], edge_dir, edge_loc1.iOct);
          count++;
        }

        if (edge_loc2.is_valid and isNotOutside2)
        {
          emf_corrected += m_emf(edge_loc2.ijk[IX], edge_loc2.ijk[IY], edge_dir, edge_loc2.iOct);
          count++;
        }
        m_emf_out(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_dir, iOct) = emf_corrected / static_cast<real_t>(count);
      }
      else if constexpr (dim == 3)
      {
        auto emf_corrected = ZERO_F;
        int  count = 1;
        emf_corrected +=
          m_emf(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_loc.ijk[IZ], edge_dir, edge_loc.iOct);

        if (edge_loc0.is_valid and isNotOutside0)
        {
          emf_corrected += m_emf(
            edge_loc0.ijk[IX], edge_loc0.ijk[IY], edge_loc0.ijk[IZ], edge_dir, edge_loc0.iOct);
          count++;
        }

        if (edge_loc1.is_valid and isNotOutside1)
        {
          emf_corrected += m_emf(
            edge_loc1.ijk[IX], edge_loc1.ijk[IY], edge_loc1.ijk[IZ], edge_dir, edge_loc1.iOct);
          count++;
        }

        if (edge_loc2.is_valid and isNotOutside2)
        {
          emf_corrected += m_emf(
            edge_loc2.ijk[IX], edge_loc2.ijk[IY], edge_loc2.ijk[IZ], edge_dir, edge_loc2.iOct);
          count++;
        }

        m_emf_out(edge_loc.ijk[IX], edge_loc.ijk[IY], edge_loc.ijk[IZ], edge_dir, iOct) =
          emf_corrected / static_cast<real_t>(count);
      }
    }

  } // end norm_v > 0

} // correct_emf

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
CorrectEmfAtBlockBorderFunctor<dim, device_t>::correct_emf_3d(index_t const & edge_flatindex,
                                                              int32_t const & iOct) const
{
  // this routine is not used anymore see correct_emf just above

  constexpr auto surface_ratio = 1 << (dim - 1);

  auto const edge_indexes =
    edge_flat_index_unravel_emf<3>(edge_flatindex, m_block_sizes, m_edge_flat_index_offsets);
  auto const & i = edge_indexes[IX];
  auto const & j = edge_indexes[IY];
  auto const & k = edge_indexes[IZ];
  auto const & edge_dir = edge_indexes[dim];

  // current block AMR key
  const auto key_cur = m_stencil_helper.key(iOct);

  // create an edge location for current edge's cell (just round down edge index in transverse
  // direction)
  const EdgeLocation_t edge_loc{ edge_indexes, key_cur, iOct, false };

  const auto v =
    get_edge_outside_unit_vector<dim>(edge_indexes, m_block_sizes, EdgeNormalType::DIAGONAL);

  auto norm_v = v[IX] * v[IX] + v[IY] * v[IY] + v[IZ] * v[IZ];

  // unit vector along edge direction
  const auto v2 = get_unit_vector<dim>(edge_dir);

  // determine if emf at current edge location must be corrected
  // i.e. if neighbor is at finer AMR level
  //
  // NB: an outside neighbor is always at same AMR level as current block
  if (norm_v == 1)
  {
    const auto edge_loc_neigh = m_stencil_helper.getBorderEdgeLocSymmetric(edge_loc);

    if (edge_loc_neigh.level() > edge_loc.level())
    {
      auto const & i2 = edge_loc_neigh.ijk[IX];
      auto const & j2 = edge_loc_neigh.ijk[IY];
      auto const & k2 = edge_loc_neigh.ijk[IZ];
      auto const & iOct2 = edge_loc_neigh.iOct;

      // clang-format off
      auto new_value = ( m_emf(i2         , j2         , k2         , edge_dir, iOct2) +
                         m_emf(i2 + v2[IX], j2 + v2[IY], k2 + v2[IZ], edge_dir, iOct2) ) / surface_ratio;
      // clang-format on

      m_emf_out(i, j, k, edge_dir, iOct) = new_value;
    }
  } // end norm_v == 1
  else if (norm_v == 2)
  {
    // we are dealing with an edge of a block
    // we explore the 3 possibilities (first neighbor across edge in diagonal, then the 2
    // neighbors across the faces)

    // diagonal neighbor
    const auto edge_loc_neigh = m_stencil_helper.getBorderEdgeLocSymmetric(edge_loc);
    if (edge_loc_neigh.level() > edge_loc.level())
    {
      auto const & i2 = edge_loc_neigh.ijk[IX];
      auto const & j2 = edge_loc_neigh.ijk[IY];
      auto const & k2 = edge_loc_neigh.ijk[IZ];
      auto const & iOct2 = edge_loc_neigh.iOct;

      const bool isNotOutside =
        iOct2 < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

      if (isNotOutside)
      {
        // clang-format off
        auto new_value = ( m_emf(i2         , j2         , k2         , edge_dir, iOct2) +
                           m_emf(i2 + v2[IX], j2 + v2[IY], k2 + v2[IZ], edge_dir, iOct2) ) / surface_ratio;
        // clang-format on

        m_emf_out(i, j, k, edge_dir, iOct) = new_value;
        return;
      }
    }

    // first direct neighbor
    const auto edge_loc_neigh1 =
      m_stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIR1);
    if (edge_loc_neigh1.level() > edge_loc.level())
    {
      auto const & i2 = edge_loc_neigh1.ijk[IX];
      auto const & j2 = edge_loc_neigh1.ijk[IY];
      auto const & k2 = edge_loc_neigh1.ijk[IZ];
      auto const & iOct2 = edge_loc_neigh1.iOct;

      const bool isNotOutside =
        iOct2 < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

      if (isNotOutside)
      {
        // clang-format off
        auto new_value = ( m_emf(i2         , j2         , k2         , edge_dir, iOct2) +
                           m_emf(i2 + v2[IX], j2 + v2[IY], k2 + v2[IZ], edge_dir, iOct2) ) / surface_ratio;
        // clang-format on

        m_emf_out(i, j, k, edge_dir, iOct) = new_value;
        return;
      }
    }

    // second direct neighbor
    const auto edge_loc_neigh2 =
      m_stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIR2);
    if (edge_loc_neigh2.level() > edge_loc.level())
    {
      auto const & i2 = edge_loc_neigh2.ijk[IX];
      auto const & j2 = edge_loc_neigh2.ijk[IY];
      auto const & k2 = edge_loc_neigh2.ijk[IZ];
      auto const & iOct2 = edge_loc_neigh2.iOct;

      const bool isNotOutside =
        iOct2 < (m_amr_mesh_info.local_num_quadrants() + m_amr_mesh_info.local_num_ghosts());

      if (isNotOutside)
      {
        // clang-format off
        auto new_value = ( m_emf(i2         , j2         , k2         , edge_dir, iOct2) +
                           m_emf(i2 + v2[IX], j2 + v2[IY], k2 + v2[IZ], edge_dir, iOct2) ) / surface_ratio;
        // clang-format on

        m_emf_out(i, j, k, edge_dir, iOct) = new_value;
        return;
      }
    }

  } // end norm_v == 2

} // correct_emf_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
CorrectEmfAtBlockBorderFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto num_edges = m_edge_flat_index_offsets[3];

  // retrieve local octant index
  auto const iOct_local = global_index / num_edges;
  auto const edge_index = global_index - iOct_local * num_edges;

  correct_emf(edge_index, iOct_local);

} // operator ()

// explicit template instantiation
template class CorrectEmfAtBlockBorderFunctor<2, kalypsso::DefaultDevice>;
template class CorrectEmfAtBlockBorderFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
