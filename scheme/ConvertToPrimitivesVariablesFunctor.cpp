// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConvertToPrimitivesVariablesFunctor.cpp
 *
 * MHD variant.
 */
#include <godunov_mhd_ct/scheme/ConvertToPrimitivesVariablesFunctor.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
ConvertToPrimitivesVariablesFunctor<dim, device_t>::ConvertToPrimitivesVariablesFunctor(
  StencilHelper_t const &         stencil_helper,
  AMRMeshInfo const &             amr_mesh_info,
  int32_t                         iOct_begin,
  DataArrayBlock_t const &        userdata_in,
  FaceDataArrayBlock_t const &    Bface_ghosted,
  DataArrayGhostedBlock_t const & userdata_out,
  MHDSettings const &             mhd_settings,
  ProlongationParam const &       prolongation,
  const int                       mpi_comm_rank)
  : m_stencil_helper(stencil_helper)
  , m_mirror_orchard_keys_device()
  , m_amr_mesh_info(amr_mesh_info)
  , m_iOct_begin(iOct_begin)
  , m_userdata_in(userdata_in)
  , m_Bface_ghosted(Bface_ghosted)
  , m_userdata_out(userdata_out)
  , m_block_sizes(userdata_out.block_size())
  , m_mhd_settings(mhd_settings)
  , m_prolongation(prolongation)
  , m_mpi_comm_rank(mpi_comm_rank)
{

  // make sure Bface_ghosted (magnetic field) has valid block sizes.
  KOKKOS_ASSERT(Bface_ghosted.cell_block_size_inner() == m_userdata_in.block_size() &&
                "Bface_ghosted has wrong inner block sizes");

  KOKKOS_ASSERT(Bface_ghosted.cell_block_size() == (m_userdata_in.block_size() + 4) &&
                "Bface_ghosted has wrong outer block sizes");
}

// ==============================================================
// ==============================================================
// same as above, but specifying also the mirror keys array
template <size_t dim, typename device_t>
ConvertToPrimitivesVariablesFunctor<dim, device_t>::ConvertToPrimitivesVariablesFunctor(
  StencilHelper_t const &         stencil_helper,
  orchard_key_view_t const &      mirror_orchard_keys,
  AMRMeshInfo const &             amr_mesh_info,
  DataArrayBlock_t const &        userdata_in,
  FaceDataArrayBlock_t const &    Bface_ghosted,
  DataArrayGhostedBlock_t const & userdata_out,
  MHDSettings const &             mhd_settings,
  ProlongationParam const &       prolongation,
  const int                       mpi_comm_rank)
  : m_stencil_helper(stencil_helper)
  , m_mirror_orchard_keys_device(mirror_orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_iOct_begin(0) // not used when processing mirrors quad
  , m_userdata_in(userdata_in)
  , m_Bface_ghosted(Bface_ghosted)
  , m_userdata_out(userdata_out)
  , m_block_sizes(userdata_out.block_size())
  , m_mhd_settings(mhd_settings)
  , m_prolongation(prolongation)
  , m_mpi_comm_rank(mpi_comm_rank)
{}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_on_group(
  ConfigMap const &                config_map,
  amr_hashmap_t const &            amr_hashmap,
  orchard_key_view_t const &       orchard_keys,
  AMRMeshInfo const &              amr_mesh_info,
  int32_t                          iOct_begin,
  int32_t                          num_octants_in_group,
  DataArrayBlock_t const &         userdata_in,
  FaceDataArrayBlock_t const &     Bface_ghosted,
  DataArrayGhostedBlock_t const &  userdata_out,
  brick_size_t<dim> const &        brick_sizes,
  Kokkos::Array<bool, dim> const & is_brick_periodic,
  MHDSettings const &              mhd_settings,
  ParallelEnv const &              par_env)
{
  // make sure the range of octants to process is valid
  assertm((iOct_begin + num_octants_in_group) <= amr_mesh_info.local_num_quadrants(),
          "Invalid range of octants to process");

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, userdata_in.block_size(), brick_sizes, is_brick_periodic);

  ConvertToPrimitivesVariablesFunctor<dim, device_t> functor(stencil_helper,
                                                             amr_mesh_info,
                                                             iOct_begin,
                                                             userdata_in,
                                                             Bface_ghosted,
                                                             userdata_out,
                                                             mhd_settings,
                                                             ProlongationParam(config_map),
                                                             par_env.rank());

  const auto nbCellsPerGhostedLeaf = userdata_out.num_cells();
  const auto nbCellsTotal = num_octants_in_group * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ConvertToPrimitivesVariablesFunctor",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuad>(0, nbCellsTotal),
                       functor);

} // apply_on_group

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_in_mirrors(
  ConfigMap const &                config_map,
  amr_hashmap_t const &            amr_hashmap,
  orchard_key_view_t const &       orchard_keys,
  orchard_key_view_t const &       mirror_orchard_keys,
  AMRMeshInfo const &              amr_mesh_info,
  DataArrayBlock_t const &         userdata_in,
  FaceDataArrayBlock_t const &     Bface_ghosted,
  DataArrayGhostedBlock_t const &  userdata_out,
  brick_size_t<dim> const &        brick_sizes,
  Kokkos::Array<bool, dim> const & is_brick_periodic,
  MHDSettings const &              mhd_settings,
  ParallelEnv const &              par_env)
{

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, userdata_in.block_size(), brick_sizes, is_brick_periodic);

  ConvertToPrimitivesVariablesFunctor<dim, device_t> functor(stencil_helper,
                                                             mirror_orchard_keys,
                                                             amr_mesh_info,
                                                             userdata_in,
                                                             Bface_ghosted,
                                                             userdata_out,
                                                             mhd_settings,
                                                             ProlongationParam(config_map),
                                                             par_env.rank());

  const auto num_mirrors = mirror_orchard_keys.extent(0);
  const auto nbCellsPerGhostedLeaf = userdata_out.num_cells();
  const auto nbCellsTotal = num_mirrors * static_cast<size_t>(nbCellsPerGhostedLeaf);

  assertm(num_mirrors == static_cast<size_t>(amr_mesh_info.local_num_mirrors()),
          "wrong number of mirror quads.");

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ConvertToPrimitivesVariablesFunctor",
                       Kokkos::RangePolicy<exec_space, TagComputeMirrorQuad>(0, nbCellsTotal),
                       functor);

} // apply_in_mirrors

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION MHDStateCell
ConvertToPrimitivesVariablesFunctor<dim, device_t>::get_conservative_vars(
  const int32_t        cellindex,
  coord_t<dim> const & iCoord,
  const iOct_t         iOct) const
{
  MHDStateCell uLoc; // conservative variables in current cell

  uLoc[MHD::ID] = m_userdata_in(cellindex, MHD::ID, iOct);
  uLoc[MHD::IP] = m_userdata_in(cellindex, MHD::IP, iOct);
  uLoc[MHD::IU] = m_userdata_in(cellindex, MHD::IU, iOct);
  uLoc[MHD::IV] = m_userdata_in(cellindex, MHD::IV, iOct);
  uLoc[MHD::IW] = m_userdata_in(cellindex, MHD::IW, iOct);

  // clang-format off
  if constexpr (dim == 2)
  {
    auto const & ig = iCoord[IX];
    auto const & jg = iCoord[IY];
    uLoc[MHD::IA] = HALF_F * (m_Bface_ghosted(ig, jg, IX, iOct) + m_Bface_ghosted(ig + 1, jg    , IX, iOct));
    uLoc[MHD::IB] = HALF_F * (m_Bface_ghosted(ig, jg, IY, iOct) + m_Bface_ghosted(ig    , jg + 1, IY, iOct));
    uLoc[MHD::IC] = HALF_F * (m_Bface_ghosted(ig, jg, IZ, iOct) + m_Bface_ghosted(ig    , jg    , IZ, iOct));
  }
  else if constexpr (dim == 3)
  {
    auto const & ig = iCoord[IX];
    auto const & jg = iCoord[IY];
    auto const & kg = iCoord[IZ];
    uLoc[MHD::IA] = HALF_F * (m_Bface_ghosted(ig, jg, kg, IX, iOct) + m_Bface_ghosted(ig + 1, jg    , kg    , IX, iOct));
    uLoc[MHD::IB] = HALF_F * (m_Bface_ghosted(ig, jg, kg, IY, iOct) + m_Bface_ghosted(ig    , jg + 1, kg    , IY, iOct));
    uLoc[MHD::IC] = HALF_F * (m_Bface_ghosted(ig, jg, kg, IZ, iOct) + m_Bface_ghosted(ig    , jg    , kg + 1, IZ, iOct));
  }
  // clang-format on

  return uLoc;

} // get_conservative_vars

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION MHDStateCell
ConvertToPrimitivesVariablesFunctor<dim, device_t>::get_conservative_vars(
  CellLocation_t const & cell_loc_in) const
{
  const auto cellindex_in = cell_loc_in.cellindex(m_block_sizes);

  return get_conservative_vars(cellindex_in, cell_loc_in.ijk, cell_loc_in.iOct);

} // get_conservative_vars

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION MHDStateCell
ConvertToPrimitivesVariablesFunctor<dim, device_t>::get_conservative_vars_restriction(
  coord_t<dim> const &   coord_out,
  CellLocation_t const & cell_loc_out,
  CellLocation_t const & cell_loc_in) const
{
  MHDStateCell uLoc; // cell-centered conservative variables in current cell

  uLoc[MHD::ID] =
    m_stencil_helper.compute_siblings_average(cell_loc_in, m_block_sizes, MHD::ID, m_userdata_in);
  uLoc[MHD::IP] =
    m_stencil_helper.compute_siblings_average(cell_loc_in, m_block_sizes, MHD::IP, m_userdata_in);
  uLoc[MHD::IU] =
    m_stencil_helper.compute_siblings_average(cell_loc_in, m_block_sizes, MHD::IU, m_userdata_in);
  uLoc[MHD::IV] =
    m_stencil_helper.compute_siblings_average(cell_loc_in, m_block_sizes, MHD::IV, m_userdata_in);
  uLoc[MHD::IW] =
    m_stencil_helper.compute_siblings_average(cell_loc_in, m_block_sizes, MHD::IW, m_userdata_in);

  //  const auto coord_out = cellindex_to_coord<dim>(
  //  cellindex_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  // now average face-centered magnetic field and make it cell-centered
  if constexpr (dim == 2)
  {
    auto const & ig = coord_out[IX];
    auto const & jg = coord_out[IY];

    // clang-format off
    uLoc[MHD::IA] =
      HALF_F * (m_Bface_ghosted(ig, jg, IX, cell_loc_out.iOct) + m_Bface_ghosted(ig + 1, jg    , IX, cell_loc_out.iOct));
    uLoc[MHD::IB] =
      HALF_F * (m_Bface_ghosted(ig, jg, IY, cell_loc_out.iOct) + m_Bface_ghosted(ig    , jg + 1, IY, cell_loc_out.iOct));
    uLoc[MHD::IC] =
                m_Bface_ghosted(ig, jg, IZ, cell_loc_out.iOct);
    // clang-format on
  }
  else if constexpr (dim == 3)
  {
    auto const & ig = coord_out[IX];
    auto const & jg = coord_out[IY];
    auto const & kg = coord_out[IZ];

    // clang-format off
    uLoc[MHD::IA] = HALF_F * (m_Bface_ghosted(ig    , jg    , kg    , IX, cell_loc_out.iOct) +
                              m_Bface_ghosted(ig + 1, jg    , kg    , IX, cell_loc_out.iOct));
    uLoc[MHD::IB] = HALF_F * (m_Bface_ghosted(ig    , jg    , kg    , IY, cell_loc_out.iOct) +
                              m_Bface_ghosted(ig    , jg + 1, kg    , IY, cell_loc_out.iOct));
    uLoc[MHD::IC] = HALF_F * (m_Bface_ghosted(ig    , jg    , kg    , IZ, cell_loc_out.iOct) +
                              m_Bface_ghosted(ig    , jg    , kg + 1, IZ, cell_loc_out.iOct));
    // clang-format on
  }

  return uLoc;

} // get_conservative_vars_restriction

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION FaceMagState
ConvertToPrimitivesVariablesFunctor<dim, device_t>::get_face_mag(
  CellLocation_t const & cell_loc_in) const
{
  FaceMagState bLoc; // face-centered magnetic field

  if constexpr (dim == 2)
  {
    auto const & ig = cell_loc_in.ijk[IX];
    auto const & jg = cell_loc_in.ijk[IY];

    // clang-format off
    bLoc[MHD::AL] = m_Bface_ghosted(ig    , jg    , IX, cell_loc_in.iOct);
    bLoc[MHD::AR] = m_Bface_ghosted(ig + 1, jg    , IX, cell_loc_in.iOct);
    bLoc[MHD::BL] = m_Bface_ghosted(ig    , jg    , IY, cell_loc_in.iOct);
    bLoc[MHD::BR] = m_Bface_ghosted(ig    , jg + 1, IY, cell_loc_in.iOct);
    bLoc[MHD::CL] = m_Bface_ghosted(ig    , jg    , IZ, cell_loc_in.iOct);
    bLoc[MHD::CR] = m_Bface_ghosted(ig    , jg    , IZ, cell_loc_in.iOct); // same as MHD::CL
    // clang-format on
  }
  else if constexpr (dim == 3)
  {
    auto const & ig = cell_loc_in.ijk[IX];
    auto const & jg = cell_loc_in.ijk[IY];
    auto const & kg = cell_loc_in.ijk[IZ];

    // clang-format off
    bLoc[MHD::AL] = m_Bface_ghosted(ig    , jg    , kg    , IX, cell_loc_in.iOct);
    bLoc[MHD::AR] = m_Bface_ghosted(ig + 1, jg    , kg    , IX, cell_loc_in.iOct);
    bLoc[MHD::BL] = m_Bface_ghosted(ig    , jg    , kg    , IY, cell_loc_in.iOct);
    bLoc[MHD::BR] = m_Bface_ghosted(ig    , jg + 1, kg    , IY, cell_loc_in.iOct);
    bLoc[MHD::CL] = m_Bface_ghosted(ig    , jg    , kg    , IZ, cell_loc_in.iOct);
    bLoc[MHD::CR] = m_Bface_ghosted(ig    , jg    , kg + 1, IZ, cell_loc_in.iOct);
    // clang-format on
  }

  return bLoc;

} // get_face_mag

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION FaceMagState
ConvertToPrimitivesVariablesFunctor<dim, device_t>::get_face_mag(coord_t<dim> const & coord_out,
                                                                 iOct_t const & iOct_out) const
{
  FaceMagState bLoc; // face-centered magnetic field

  if constexpr (dim == 2)
  {
    auto const & ig = coord_out[IX];
    auto const & jg = coord_out[IY];

    // clang-format off
    bLoc[MHD::AL] = m_Bface_ghosted(ig    , jg    , IX, iOct_out);
    bLoc[MHD::AR] = m_Bface_ghosted(ig + 1, jg    , IX, iOct_out);
    bLoc[MHD::BL] = m_Bface_ghosted(ig    , jg    , IY, iOct_out);
    bLoc[MHD::BR] = m_Bface_ghosted(ig    , jg + 1, IY, iOct_out);
    bLoc[MHD::CL] = m_Bface_ghosted(ig    , jg    , IZ, iOct_out);
    bLoc[MHD::CR] = bLoc[MHD::CL];
    // clang-format on
  }
  else if constexpr (dim == 3)
  {
    auto const & ig = coord_out[IX];
    auto const & jg = coord_out[IY];
    auto const & kg = coord_out[IZ];

    // clang-format off
    bLoc[MHD::AL] = m_Bface_ghosted(ig    , jg    , kg    , IX, iOct_out);
    bLoc[MHD::AR] = m_Bface_ghosted(ig + 1, jg    , kg    , IX, iOct_out);
    bLoc[MHD::BL] = m_Bface_ghosted(ig    , jg    , kg    , IY, iOct_out);
    bLoc[MHD::BR] = m_Bface_ghosted(ig    , jg + 1, kg    , IY, iOct_out);
    bLoc[MHD::CL] = m_Bface_ghosted(ig    , jg    , kg    , IZ, iOct_out);
    bLoc[MHD::CR] = m_Bface_ghosted(ig    , jg    , kg + 1, IZ, iOct_out);
    // clang-format on
  }

  return bLoc;

} // get_face_mag

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::set_primitive_vars(
  const int32_t                         cellindex_out_g,
  [[maybe_unused]] coord_t<dim> const & coord_in,
  [[maybe_unused]] const iOct_t         iOct_in,
  const iOct_t                          iOct_out,
  MHDStateCell const &                  q,
  FaceMagState const &                  face_mag) const
{

  m_userdata_out(cellindex_out_g, MHD::ID, iOct_out) = q[MHD::ID];
  m_userdata_out(cellindex_out_g, MHD::IP, iOct_out) = q[MHD::IP];
  m_userdata_out(cellindex_out_g, MHD::IU, iOct_out) = q[MHD::IU];
  m_userdata_out(cellindex_out_g, MHD::IV, iOct_out) = q[MHD::IV];
  m_userdata_out(cellindex_out_g, MHD::IW, iOct_out) = q[MHD::IW];

  m_userdata_out(cellindex_out_g, MHD::IAL, iOct_out) = face_mag[MHD::AL];
  m_userdata_out(cellindex_out_g, MHD::IAR, iOct_out) = face_mag[MHD::AR];
  m_userdata_out(cellindex_out_g, MHD::IBL, iOct_out) = face_mag[MHD::BL];
  m_userdata_out(cellindex_out_g, MHD::IBR, iOct_out) = face_mag[MHD::BR];
  m_userdata_out(cellindex_out_g, MHD::ICL, iOct_out) = face_mag[MHD::CL];
  m_userdata_out(cellindex_out_g, MHD::ICR, iOct_out) = face_mag[MHD::CR];

} // set_primitive_vars

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::fill_inner(coord_t<dim> const & coord_in,
                                                               int32_t              cellindex_in,
                                                               int32_t              cellindex_out,
                                                               iOct_t               iOct_global,
                                                               iOct_t               iOct_out) const
{
  // read conservative variables (hydro) + magnetic field
  const auto uLoc = get_conservative_vars(cellindex_in, coord_in, iOct_global);

  // compute primitive variables in current cell
  auto qLoc = core::models::mhd::computePrimitives(uLoc, m_mhd_settings);

  // fake current location (AMR key is not needed here, just it to zero)
  const CellLocation_t cell_loc{ coord_in, 0, iOct_global, false };

  // read magnetic field (face-centered values) at current location
  FaceMagState bLoc = get_face_mag(cell_loc);

  // write primitive variables
  set_primitive_vars(cellindex_out, coord_in, iOct_global, iOct_out, qLoc, bLoc);

} // fill_inner

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::fill_ghost_copy(
  CellLocation_t const & cell_loc_out,
  CellLocation_t const & cell_loc_in,
  index_t const &        cellindex_out,
  coord_t<dim> const &   coord_out,
  iOct_t const &         iOct_out) const
{

  const bool do_restriction = cell_loc_in.level() == (cell_loc_out.level() + 1);

  // read cell-centered conservative variables + compute cell-centered magnetic field
  const auto uLoc = do_restriction
                      ? get_conservative_vars_restriction(coord_out, cell_loc_out, cell_loc_in)
                      : get_conservative_vars(cell_loc_in);

  // compute cell-centered primitive variables in current cell
  auto qLoc = core::models::mhd::computePrimitives(uLoc, m_mhd_settings);

  // write cell-centered primitive variables
  m_userdata_out(cellindex_out, MHD::ID, iOct_out) = qLoc[MHD::ID];
  m_userdata_out(cellindex_out, MHD::IP, iOct_out) = qLoc[MHD::IP];
  m_userdata_out(cellindex_out, MHD::IU, iOct_out) = qLoc[MHD::IU];
  m_userdata_out(cellindex_out, MHD::IV, iOct_out) = qLoc[MHD::IV];
  m_userdata_out(cellindex_out, MHD::IW, iOct_out) = qLoc[MHD::IW];

  //
  // write magnetic field (face-centered values)
  //
  const auto bLoc = get_face_mag(coord_out, cell_loc_out.iOct);

  m_userdata_out(cellindex_out, MHD::IAL, iOct_out) = bLoc[MHD::AL];
  m_userdata_out(cellindex_out, MHD::IAR, iOct_out) = bLoc[MHD::AR];
  m_userdata_out(cellindex_out, MHD::IBL, iOct_out) = bLoc[MHD::BL];
  m_userdata_out(cellindex_out, MHD::IBR, iOct_out) = bLoc[MHD::BR];
  m_userdata_out(cellindex_out, MHD::ICL, iOct_out) = bLoc[MHD::CL];
  m_userdata_out(cellindex_out, MHD::ICR, iOct_out) = bLoc[MHD::CR];

} // fill_ghost_copy

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::
  linear_extrapolate_hydro_vars_using_limited_slopes(CellLocation<2> const & cell_loc_neigh,
                                                     coord_t<2> const &      coord_in,
                                                     iOct_t const &          iOct_global,
                                                     index_t const &         cellindex_out,
                                                     coord_t<2> const &      coord_out,
                                                     iOct_t const &          iOct_out) const
{
  const real_t slope_type = 1;
  const auto   nbvar_hydro = 5;

  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in -1, +1
  const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
  const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;

  MHDStateCell uLoc; // conservative variables in current cell

  for (int32_t ivar = 0; ivar < nbvar_hydro; ++ivar)
  {
    // compute limited slopes
    auto const dudx = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, ivar, m_userdata_in, slope_type);
    auto const dudy = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, ivar, m_userdata_in, slope_type);

    // extrapolate conservative variables
    uLoc[static_cast<size_t>(ivar)] =
      m_userdata_in(cell_loc_neigh.cellindex(m_block_sizes), ivar, cell_loc_neigh.iOct) +
      KALYPSSO_NUM(0.25) * static_cast<real_t>(ix) * dudx +
      KALYPSSO_NUM(0.25) * static_cast<real_t>(iy) * dudy;
  }

  // read face-centered magnetic field (already prolongated !)
  // and convert it into cell-centered values
  // const auto bLoc = get_face_mag(coord_out, iOct_out);
  const auto bLoc = get_face_mag(coord_out, iOct_global);

  uLoc[MHD::IA] = HALF_F * (bLoc[MHD::AL] + bLoc[MHD::AR]);
  uLoc[MHD::IB] = HALF_F * (bLoc[MHD::BL] + bLoc[MHD::BR]);
  uLoc[MHD::IC] = bLoc[MHD::CL];

  // compute primitive variables in current cell
  auto qLoc = core::models::mhd::computePrimitives(uLoc, m_mhd_settings);

  //
  // write hydro primitive variables
  //
  for (int32_t ivar = 0; ivar < nbvar_hydro; ++ivar)
  {
    m_userdata_out(cellindex_out, ivar, iOct_out) = qLoc[static_cast<size_t>(ivar)];
  }

  //
  // write magnetic field (face-centered values)
  //
  m_userdata_out(cellindex_out, MHD::IAL, iOct_out) = bLoc[MHD::AL];
  m_userdata_out(cellindex_out, MHD::IAR, iOct_out) = bLoc[MHD::AR];
  m_userdata_out(cellindex_out, MHD::IBL, iOct_out) = bLoc[MHD::BL];
  m_userdata_out(cellindex_out, MHD::IBR, iOct_out) = bLoc[MHD::BR];
  m_userdata_out(cellindex_out, MHD::ICL, iOct_out) = bLoc[MHD::CL];
  m_userdata_out(cellindex_out, MHD::ICR, iOct_out) = bLoc[MHD::CR];

} // linear_extrapolate_hydro_vars_using_limited_slopes - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::
  linear_extrapolate_hydro_vars_using_limited_slopes(CellLocation<3> const & cell_loc_neigh,
                                                     coord_t<3> const &      coord_in,
                                                     iOct_t const &          iOct_global,
                                                     index_t const &         cellindex_out,
                                                     coord_t<3> const &      coord_out,
                                                     iOct_t const &          iOct_out) const
{

  const real_t slope_type = 1; // TODO : investigate if a better value should be searched for
  const auto   nbvar_hydro = 5;

  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  const auto cell_loc_left_z =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-ZDIR));
  const auto cell_loc_right_z =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+ZDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in -1, +1
  const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
  const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;
  const int iz = 2 * (coord_in[IZ] - 2 * (coord_in[IZ] / 2)) - 1;

  MHDStateCell uLoc; // conservative variables in current cell

  for (int32_t ivar = 0; ivar < nbvar_hydro; ++ivar)
  {
    // compute limited slopes
    auto const dudx = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, ivar, m_userdata_in, slope_type);
    auto const dudy = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, ivar, m_userdata_in, slope_type);
    auto const dudz = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_z, cell_loc_left_z, ivar, m_userdata_in, slope_type);

    // extrapolate conservative variables
    uLoc[static_cast<size_t>(ivar)] =
      m_userdata_in(cell_loc_neigh.cellindex(m_block_sizes), ivar, cell_loc_neigh.iOct) +
      KALYPSSO_NUM(0.25) * static_cast<real_t>(ix) * dudx +
      KALYPSSO_NUM(0.25) * static_cast<real_t>(iy) * dudy +
      KALYPSSO_NUM(0.25) * static_cast<real_t>(iz) * dudz;
  }

  // read face-centered magnetic field (already prolongated !)
  // and convert it into cell-centered values
  // const auto bLoc = get_face_mag(coord_out, iOct_out);
  const auto bLoc = get_face_mag(coord_out, iOct_global);

  uLoc[MHD::IA] = HALF_F * (bLoc[MHD::AL] + bLoc[MHD::AR]);
  uLoc[MHD::IB] = HALF_F * (bLoc[MHD::BL] + bLoc[MHD::BR]);
  uLoc[MHD::IC] = HALF_F * (bLoc[MHD::CL] + bLoc[MHD::CR]);

  // compute primitive variables in current cell
  auto qLoc = core::models::mhd::computePrimitives(uLoc, m_mhd_settings);

  //
  // write hydro primitive variables
  //
  for (int32_t ivar = 0; ivar < nbvar_hydro; ++ivar)
  {
    m_userdata_out(cellindex_out, ivar, iOct_out) = qLoc[static_cast<size_t>(ivar)];
  }

  //
  // write magnetic field (face-centered values)
  //
  m_userdata_out(cellindex_out, MHD::IAL, iOct_out) = bLoc[MHD::AL];
  m_userdata_out(cellindex_out, MHD::IAR, iOct_out) = bLoc[MHD::AR];
  m_userdata_out(cellindex_out, MHD::IBL, iOct_out) = bLoc[MHD::BL];
  m_userdata_out(cellindex_out, MHD::IBR, iOct_out) = bLoc[MHD::BR];
  m_userdata_out(cellindex_out, MHD::ICL, iOct_out) = bLoc[MHD::CL];
  m_userdata_out(cellindex_out, MHD::ICR, iOct_out) = bLoc[MHD::CR];

} // linear_extrapolate_hydro_vars_using_limited_slopes - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::fill_ghosts(index_t const &      cellindex_out,
                                                                coord_t<dim> const & coord_out,
                                                                iOct_t const &       iOct_global,
                                                                iOct_t const &       iOct_out) const
{
  const auto & b = m_block_sizes;

  // coordinates of source cell (where to read data)
  coord_t<dim> coord_in;
  const auto   dir = ghosted_coords_to_inner_coords(coord_in, coord_out, b);

  int32_t cellindex_in = coord_to_cellindex<dim>(coord_in, m_userdata_in.block_size());

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  if (dir_norm == 0)
  {
    // current cell is inside current block
    fill_inner(coord_in, cellindex_in, cellindex_out, iOct_global, iOct_out);
  }
  else
  {
    // current cell is a ghost cell (thus belonging to a neighbor block)

    /*
     * fill actual ghosts with data from a neighbor block.
     */

    // get orchard key of current octant
    auto key_cur = m_stencil_helper.key(iOct_global);

    shift_t<dim> shift;
    shift[IX] = b[IX] * dir[IX];
    shift[IY] = b[IY] * dir[IY];
    if constexpr (dim == 3)
    {
      shift[IZ] = b[IZ] * dir[IZ];
    }

    const CellLocation_t cell_loc_cur{ coord_in, key_cur, iOct_global, false };
    const auto           cell_loc_neigh = m_stencil_helper.getNeighLoc(cell_loc_cur, shift);

    /*
     * Dealing with the 3 possibilities:
     * - neighbor octant is at same    AMR level : doing a simple copy
     * - neighbor octant is at finer   AMR level : doing a restriction (average values)
     * - neighbor octant is at coarser AMR level : doing a prolongation
     */
    if (cell_loc_neigh.level() >= cell_loc_cur.level())
    {
      // doing a simple copy or doing a restriction when neighbor is at higher AMR level (hydro and
      // magnetic field)
      fill_ghost_copy(cell_loc_cur, cell_loc_neigh, cellindex_out, coord_out, iOct_out);
    }
    else if (cell_loc_neigh.level() + 1 == cell_loc_cur.level())
    {
      // doing a prolongation (because neighbor is coarser)

      if (m_prolongation.m_cell == +CellCenteredProlongationType::SIMPLE_COPY)
      {
        // simple copy of the coarse value
        fill_ghost_copy(cell_loc_cur, cell_loc_neigh, cellindex_out, coord_out, iOct_out);
      }
      else if (m_prolongation.m_cell == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
      {
        linear_extrapolate_hydro_vars_using_limited_slopes(
          cell_loc_neigh, coord_in, iOct_global, cellindex_out, coord_out, iOct_out);
      }
    }
    else
    {
      KOKKOS_ASSERT(false && "Logic error: neighbor octant not found (Kernel Panic !)");
    }

  } // end if (dir_norm ==0)

} // fill_ghosts

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::operator()(TagComputeAllQuad const &,
                                                               const index_t & global_index) const
{
  const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

  // retrieve local octant index (this is where we want to write data)
  const auto iOct_local = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iOct_local * nbCellsPerGhostedLeaf;

  // retrieve global octant index
  const auto iOct_global = m_iOct_begin + iOct_local;

  // compute cartesian coordinates inside ghosted block
  const auto coord_out = cellindex_to_coord<dim>(
    cell_index_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  fill_ghosts(cell_index_out, coord_out, iOct_global, iOct_local);

} // operator() - TagComputeAllQuad

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::operator()(TagComputeMirrorQuad const &,
                                                               const index_t & global_index) const
{
  const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

  // retrieve mirror index (where we want to write)
  const auto iMirror = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iMirror * nbCellsPerGhostedLeaf;

  // retrieve key associated to that mirror index
  const auto mirror_key = m_mirror_orchard_keys_device(iMirror);

  // make sure the key is in the hashmap and retrieve value
  const auto mirror_hashindex = m_stencil_helper.m_amr_hashmap_device.find(mirror_key);
  [[maybe_unused]] const auto valid =
    m_stencil_helper.m_amr_hashmap_device.valid_at(mirror_hashindex);

  KOKKOS_ASSERT(
    valid && "(mirror quadrant) key doesn't exist in hashmap (this is in principle not possible, "
             "since mirror keys are computed from p4est ghosts.)");

  // retrieve iOct associated to that mirror quadrant
  const auto iOct_global = m_stencil_helper.m_amr_hashmap_device.value_at(mirror_hashindex);

  // compute cartesian coordinates inside ghosted block
  const auto coord_out = cellindex_to_coord<dim>(
    cell_index_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  fill_ghosts(cell_index_out, coord_out, iOct_global, iMirror);

} // operator() - TagComputeMirrorQuad

// explicit template instantiation
template class ConvertToPrimitivesVariablesFunctor<2, kalypsso::DefaultDevice>;
template class ConvertToPrimitivesVariablesFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
