// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ReadFluxesAndConservativeUpdateFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/ReadFluxesAndConservativeUpdateFunctor.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::ReadFluxesAndConservativeUpdateFunctor(
  ConfigMap const &                  config_map,
  StencilHelper_t const &            stencil_helper,
  orchard_key_view_t const &         orchard_keys,
  conformal_status_view_type const & conformal_status,
  AMRMeshInfo const &                amr_mesh_info,
  DataArrayBlock_t const &           u_out,
  FaceDataArrayBlock_t const &       b_out,
  DataArrayBlock_t const &           fluxes,
  FieldMap<core::models::MHD>        fm,
  int                                direction,
  MHDSettings const &                mhd_settings,
  real_t                             dt)
  : m_stencil_helper(stencil_helper)
  , m_orchard_keys_device(orchard_keys)
  , m_conformal_status(conformal_status)
  , m_amr_mesh_info(amr_mesh_info)
  , m_Uout(u_out)
  , m_Bout(b_out)
  , m_Fluxes(fluxes)
  , m_fm(fm)
  , m_direction(direction)
  , m_num_owned(amr_mesh_info.local_num_quadrants())
  , m_num_ghosts(amr_mesh_info.local_num_ghosts())
  , m_block_sizes(u_out.block_size())
  , m_mhd_settings(mhd_settings)
  , m_dt(dt)
  , m_scaling_factor(get_scaling_factor(config_map))
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::apply(
  ConfigMap const &                  config_map,
  amr_hashmap_t const &              amr_hashmap,
  orchard_key_view_t const &         orchard_keys,
  conformal_status_view_type const & conformal_status,
  AMRMeshInfo const &                amr_mesh_info,
  DataArrayBlock_t const &           Uout,
  FaceDataArrayBlock_t const &       Bout,
  DataArrayBlock_t const &           fluxes,
  FieldMap<core::models::MHD>        fm,
  int                                direction,
  brick_size_t<dim> const &          brick_sizes,
  Kokkos::Array<bool, dim> const &   is_brick_periodic,
  MHDSettings const &                mhd_settings,
  real_t                             dt)
{
  // Important note: the caller is responsible for providing a flux array with right shape.
  {
    [[maybe_unused]] auto flux_block_sizes = Uout.block_size();
    flux_block_sizes[static_cast<size_t>(direction)]++;
    assertm(flux_block_sizes == fluxes.shape(), "Flux array has incompatible shape.");
  }

  auto stencil_helper =
    StencilHelper_t(amr_hashmap, orchard_keys, Uout.block_size(), brick_sizes, is_brick_periodic);

  ReadFluxesAndConservativeUpdateFunctor<dim, device_t> functor(config_map,
                                                                stencil_helper,
                                                                orchard_keys,
                                                                conformal_status,
                                                                amr_mesh_info,
                                                                Uout,
                                                                Bout,
                                                                fluxes,
                                                                fm,
                                                                direction,
                                                                mhd_settings,
                                                                dt);

  // number of owned quadrant x number of block cells
  const auto nbIterations = amr_mesh_info.local_num_quadrants() * Uout.num_cells();

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::ReadFluxesAndConservativeUpdateFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::read_fluxes_and_update_2d(
  index_t const & cell_index,
  index_t const & iOct) const
{
  constexpr auto volume_ratio = 1 << dim;

  auto const   coords = cell_index_unravel<2>(cell_index, m_Uout.shape());
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  /*
   * Update with flux along IX
   */
  if (m_direction == IX)
  {
    auto accum_flux = init_kokkos_array<real_t, MHDStateCell::size()>(ZERO_F);

    const auto face_xmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    const auto face_xmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // accumulate flux
    // - if neighbor is same size or larger, do a regular update
    // - if neighbor is smaller, we need to accumulate flux from fine neighbors
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY])
    {
      //
      // left coming flux
      //
      if (i == 0 and face_xmin_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ -1, 0 };
        constexpr bool         use_right_flux = true;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux + get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux + read_flux(i, j, iOct);
      }

      //
      // right coming flux
      //
      if (i == m_block_sizes[IX] - 1 and face_xmax_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 1, 0 };
        constexpr bool         use_right_flux = false;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux - get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux - read_flux(i + 1, j, iOct);
      }

      update_U(i, j, iOct, accum_flux);
    }
  } // end direction IX

  /*
   * Update with flux along IY
   */
  if (m_direction == IY)
  {
    auto accum_flux = init_kokkos_array<real_t, MHDStateCell::size()>(ZERO_F);

    const auto face_ymin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    const auto face_ymax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // accumulate flux
    // - if neighbor is same size or larger, do a regular update
    // - if neighbor is smaller, we need to accumulate flux from fine neighbors
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY])
    {
      //
      // left coming flux
      //
      if (j == 0 and face_ymin_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 0, -1 };
        constexpr bool         use_right_flux = true;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux + get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux + read_flux(i, j, iOct);
      }

      //
      // right coming flux
      //
      if (j == m_block_sizes[IY] - 1 and face_ymax_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 0, 1 };
        constexpr bool         use_right_flux = false;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux - get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux - read_flux(i, j + 1, iOct);
      }

      update_U(i, j, iOct, accum_flux);
    }
  } // end direction IY

} // read_fluxes_and_update_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::read_fluxes_and_update_3d(
  index_t const & cell_index,
  index_t const & iOct) const
{
  constexpr auto volume_ratio = 1 << dim;

  auto const   coords = cell_index_unravel<3>(cell_index, m_Uout.shape());
  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  /*
   * Update with flux along IX
   */
  if (m_direction == IX)
  {
    auto accum_flux = init_kokkos_array<real_t, MHDStateCell::size()>(ZERO_F);

    const auto face_xmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    const auto face_xmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // accumulate flux
    // - if neighbor is same size or larger, do a regular update
    // - if neighbor is smaller, we need to accumulate flux from fine neighbors
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      //
      // left coming flux
      //
      if (i == 0 and face_xmin_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ -1, 0, 0 };
        constexpr bool         use_right_flux = true;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux + get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux + read_flux(i, j, k, iOct);
      }

      //
      // right coming flux
      //
      if (i == m_block_sizes[IX] - 1 and face_xmax_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 1, 0, 0 };
        constexpr bool         use_right_flux = false;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux - get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux - read_flux(i + 1, j, k, iOct);
      }

      update_U(i, j, k, iOct, accum_flux);
    }
  } // end direction IX

  /*
   * Update with flux along IY
   */
  if (m_direction == IY)
  {
    auto accum_flux = init_kokkos_array<real_t, MHDStateCell::size()>(ZERO_F);

    const auto face_ymin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    const auto face_ymax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // accumulate flux
    // - if neighbor is same size or larger, do a regular update
    // - if neighbor is smaller, we need to accumulate flux from fine neighbors
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      //
      // left coming flux
      //
      if (j == 0 and face_ymin_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 0, -1, 0 };
        constexpr bool         use_right_flux = true;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux + get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux + read_flux(i, j, k, iOct);
      }

      //
      // right coming flux
      //
      if (j == m_block_sizes[IY] - 1 and face_ymax_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 0, 1, 0 };
        constexpr bool         use_right_flux = false;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux - get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux - read_flux(i, j + 1, k, iOct);
      }

      update_U(i, j, k, iOct, accum_flux);
    }
  } // end direction IY

  /*
   * Update with flux along IZ
   */
  if (m_direction == IZ)
  {
    auto accum_flux = init_kokkos_array<real_t, MHDStateCell::size()>(ZERO_F);

    const auto face_zmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_zmin(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    const auto face_zmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_zmax(m_conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // accumulate flux
    // - if neighbor is same size or larger, do a regular update
    // - if neighbor is smaller, we need to accumulate flux from fine neighbors
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      //
      // left coming flux
      //
      if (k == 0 and face_zmin_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 0, 0, -1 };
        constexpr bool         use_right_flux = true;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux + get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux + read_flux(i, j, k, iOct);
      }

      //
      // right coming flux
      //
      if (k == m_block_sizes[IZ] - 1 and face_zmax_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 0, 0, 1 };
        constexpr bool         use_right_flux = false;

        // Note: current cell is larger than neighbor (1 AMR level difference), so volume is 2^dim
        // times larger than neighbor cell volume
        accum_flux = accum_flux - get_flux_from_fine_neighbor(iOct, coords, shift, use_right_flux) /
                                    volume_ratio;
      }
      else
      {
        accum_flux = accum_flux - read_flux(i, j, k + 1, iOct);
      }

      update_U(i, j, k, iOct, accum_flux);
    }
  } // end direction IZ

} // read_fluxes_and_update_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::operator()(
  const index_t & global_index) const
{
  const auto num_cells = m_Uout.num_cells();

  // retrieve local octant index
  auto const iOct_local = global_index / num_cells;
  auto const cell_index = global_index - iOct_local * num_cells;

  if constexpr (dim == 2)
  {
    read_fluxes_and_update_2d(cell_index, iOct_local);
  }
  else if constexpr (dim == 3)
  {
    read_fluxes_and_update_3d(cell_index, iOct_local);
  }

} // operator()

// explicit template instantiation
template class ReadFluxesAndConservativeUpdateFunctor<2, kalypsso::DefaultDevice>;
template class ReadFluxesAndConservativeUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
