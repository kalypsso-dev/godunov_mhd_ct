// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeSourceFaceMagFunctor.cpp for MHD using cell-centered primitive variables.
 */
#include <godunov_mhd_ct/scheme/ComputeSourceFaceMagFunctor.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
ComputeSourceFaceMagFunctor<dim, device_t>::ComputeSourceFaceMagFunctor(
  DataArrayGhostedBlock_t const & sFaceMag,
  DataArrayGhostedBlock_t const & elec_field,
  FieldMap<core::models::MHD>     fm,
  int32_t                         iOct_begin,
  int32_t                         num_octants,
  real_t                          dt,
  real_t                          scaling_factor,
  orchard_key_view_t const &      orchard_keys)
  : m_sFaceMag(sFaceMag)
  , m_elec_field(elec_field)
  , m_fm(fm)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
  , m_dt(dt)
  , m_scaling_factor(scaling_factor)
  , m_orchard_keys_device(orchard_keys)
{}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeSourceFaceMagFunctor<dim, device_t>::apply_on_group(
  DataArrayGhostedBlock_t const & sFaceMag,
  DataArrayGhostedBlock_t const & elec_field,
  FieldMap<core::models::MHD>     fm,
  int32_t                         num_octants,
  real_t                          dt,
  ConfigMap const &               config_map,
  orchard_key_view_t const &      orchard_keys)
{

  ComputeSourceFaceMagFunctor<dim, device_t> functor(sFaceMag,
                                                     elec_field,
                                                     fm,
                                                     0,           // first index to compute
                                                     num_octants, // number of octants to process
                                                     dt,
                                                     get_scaling_factor(config_map),
                                                     orchard_keys);

  const auto nbCellsPerGhostedLeaf = sFaceMag.num_cells();
  const auto nbCellsTotal = num_octants * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ComputeSourceFaceMagFunctor - All quadrants",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuadInGroup>(0, nbCellsTotal),
                       functor);

} // apply_on_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeSourceFaceMagFunctor<dim, device_t>::apply_on_ghosts(
  DataArrayGhostedBlock_t const & sFaceMag,
  DataArrayGhostedBlock_t const & elec_field,
  FieldMap<core::models::MHD>     fm,
  int32_t                         num_mirrors,
  int32_t                         num_ghosts,
  real_t                          dt,
  ConfigMap const &               config_map,
  orchard_key_view_t const &      orchard_keys)
{

  // we expect primitive_vars to be of size num_ghosts
  KOKKOS_ASSERT(sFaceMag.num_quadrants() == num_ghosts &&
                "[ComputeSourceFaceMagFunctor] primitive_vars has wrong sizes");

  // we expect elec_field to be of size num_ghosts
  KOKKOS_ASSERT(elec_field.num_quadrants() == num_ghosts &&
                "[ComputeSourceFaceMagFunctor] elec_field has wrong sizes");

  ComputeSourceFaceMagFunctor<dim, device_t> functor(sFaceMag,
                                                     elec_field,
                                                     fm,
                                                     num_mirrors, // first index to compute
                                                     num_ghosts,  // number of octant to compute
                                                     dt,
                                                     get_scaling_factor(config_map),
                                                     orchard_keys);

  const auto nbCellsPerGhostedLeaf = sFaceMag.num_cells();
  const auto nbCellsTotal = num_ghosts * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ComputeSourceFaceMagFunctor - ghost quadrants only",
                       Kokkos::RangePolicy<exec_space, TagComputeGhostQuad>(0, nbCellsTotal),
                       functor);

} // apply_on_ghosts

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeSourceFaceMagFunctor<dim, device_t>::compute_sFaceMag(index_t const & cell_index,
                                                             index_t const & iOct_in,
                                                             index_t const & iOct_out) const
{

  // get AMR level
  auto const level = orchard_key_t<dim>::level(m_orchard_keys_device(iOct_in));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<dim>(level, m_sFaceMag.block_size()[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;


  // compute cartesian coordinates inside ghosted block
  const auto coord =
    cellindex_to_coord<dim>(cell_index, m_sFaceMag.ghosted_block_size(), m_sFaceMag.shift());

  if constexpr (dim == 2)
  {
    auto const & dtdx = dtdS_over_dV_cur;
    auto const & dtdy = dtdS_over_dV_cur;

    auto const & i = coord[IX];
    auto const & j = coord[IY];

    auto const & ie = i;
    auto const & je = j;

    // compute source term for magnetic field dt/2 integration

    // clang-format off

    // sAL0 = +(ELR - ELL) * 0.5 * dtdy
    m_sFaceMag(i, j, IX, iOct_out) =  (m_elec_field(ie    , je + 1, 0, iOct_in) -
                                       m_elec_field(ie    , je    , 0, iOct_in)) * HALF_F * dtdy;

    // sBL0 = -(ERL - ELL) * 0.5 * dtdx;
    m_sFaceMag(i, j, IY, iOct_out) = -(m_elec_field(ie + 1, je    , 0, iOct_in) -
                                       m_elec_field(ie    , je    , 0, iOct_in)) * HALF_F * dtdx;
    // clang-format on
  }
  else if constexpr (dim == 3)
  {
    auto const & dtdx = dtdS_over_dV_cur;
    auto const & dtdy = dtdS_over_dV_cur;
    auto const & dtdz = dtdS_over_dV_cur;

    auto const & i = coord[IX];
    auto const & j = coord[IY];
    auto const & k = coord[IZ];

    auto const & ie = i;
    auto const & je = j;
    auto const & ke = k;

    // compute source term for magnetic field dt/2 integration

    // clang-format off

    // sAL0 = +(GLR - GLL) * dtdy * HALF_F - (FLR - FLL) * dtdz * HALF_F
    m_sFaceMag(i, j, k, IX, iOct_out) =
      (m_elec_field(ie, je + 1, ke    , IZ, iOct_in) -
       m_elec_field(ie, je    , ke    , IZ, iOct_in)) * HALF_F * dtdy -
      (m_elec_field(ie, je    , ke + 1, IY, iOct_in) -
       m_elec_field(ie, je    , ke    , IY, iOct_in)) * HALF_F * dtdz;

    // sBL0 = -(GRL - GLL) * dtdx * HALF_F + (ELR - ELL) * dtdz * HALF_F
    m_sFaceMag(i, j, k, IY, iOct_out) =
      -(m_elec_field(ie + 1, je, ke    , IZ, iOct_in) -
        m_elec_field(ie    , je, ke    , IZ, iOct_in)) * HALF_F * dtdx +
      ( m_elec_field(ie    , je, ke + 1, IX, iOct_in) -
        m_elec_field(ie    , je, ke    , IX, iOct_in)) * HALF_F * dtdz;

    // sCL0 = +(FRL - FLL) * dtdx * HALF_F - (ERL - ELL) * dtdy * HALF_F
    m_sFaceMag(i, j, k, IZ, iOct_out) =
      (m_elec_field(ie + 1, je    , ke, IY, iOct_in) -
       m_elec_field(ie    , je    , ke, IY, iOct_in)) * HALF_F * dtdx -
      (m_elec_field(ie    , je + 1, ke, IX, iOct_in) -
       m_elec_field(ie    , je    , ke, IX, iOct_in)) * HALF_F * dtdy;

    // clang-format on

  } // end dim == 3

} // compute_sFaceMag

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeSourceFaceMagFunctor<dim, device_t>::operator()(TagComputeAllQuadInGroup const &,
                                                       const index_t & global_index) const
{

  // retrieve local octant index (local to group)
  const auto iOct_local = global_index / m_sFaceMag.num_cells();
  const auto cell_index = global_index - iOct_local * m_sFaceMag.num_cells();

  // index in (to access m_elec_field) is iOct_local
  // index out (to access m_sFaceMag) is iOct_local
  compute_sFaceMag(cell_index, iOct_local, iOct_local);

} // operator() - TagComputeAllQuadInGroup

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeSourceFaceMagFunctor<dim, device_t>::operator()(TagComputeGhostQuad const &,
                                                       const index_t & global_index) const
{

  // retrieve ghost index - iGhost should take values between 0 and m_num_octants-1 and
  // in this context, m_num_octants must be equal to the number of ghosts quadrants
  int32_t    iGhost = global_index / m_sFaceMag.num_cells();
  const auto cell_index = global_index - iGhost * m_sFaceMag.num_cells();

  // index in (to access m_elec_field) is iGhost
  // index out (to access m_sFaceMag) is iGhost
  compute_sFaceMag(cell_index, iGhost, iGhost);

} // operator() - TagComputeGhostQuad

// explicit template instantiation
template class ComputeSourceFaceMagFunctor<2, kalypsso::DefaultDevice>;
template class ComputeSourceFaceMagFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
