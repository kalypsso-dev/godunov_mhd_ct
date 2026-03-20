// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeElectricFieldFunctor.cpp for MHD using cell-centered primitive variables.
 */
#include <godunov_mhd_ct/scheme/ComputeElectricFieldFunctor.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
ComputeElectricFieldFunctor<dim, device_t>::ComputeElectricFieldFunctor(
  DataArrayGhostedBlock_t const & prim_var,
  DataArrayGhostedBlock_t const & elec_field,
  FieldMap<core::models::MHD>     fm,
  int32_t                         iOct_begin,
  int32_t                         num_octants)
  : m_q(prim_var)
  , m_elec_field(elec_field)
  , m_fm(fm)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
{}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeElectricFieldFunctor<dim, device_t>::apply_on_group(
  DataArrayGhostedBlock_t const & primitive_vars,
  DataArrayGhostedBlock_t const & elec_field,
  FieldMap<core::models::MHD>     fm,
  int32_t                         num_quads)
{

  ComputeElectricFieldFunctor<dim, device_t> functor(primitive_vars,
                                                     elec_field,
                                                     fm,
                                                     0,        // first index to compute
                                                     num_quads // number of quads to process
  );

  const auto nbCellsPerGhostedLeaf = elec_field.num_cells();
  const auto nbCellsTotal = num_quads * nbCellsPerGhostedLeaf;

  Kokkos::parallel_for("ComputeElectricFieldFunctor - group of quadrants",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuadInGroup>(0, nbCellsTotal),
                       functor);

} // apply_on_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeElectricFieldFunctor<dim, device_t>::apply_on_ghosts(
  DataArrayGhostedBlock_t const & primitive_vars,
  DataArrayGhostedBlock_t const & elec_field,
  FieldMap<core::models::MHD>     fm,
  int32_t                         num_mirrors,
  int32_t                         num_ghosts)
{

  // we expect primitive_vars to be of size num_mirrors + num_ghosts
  KOKKOS_ASSERT(primitive_vars.num_quadrants() == num_mirrors + num_ghosts &&
                "[ComputeElectricFieldFunctor] primitive_vars has wrong sizes");

  // we expect elec_field to be of size num_ghosts
  KOKKOS_ASSERT(elec_field.num_quadrants() == num_ghosts &&
                "[ComputeElectricFieldFunctor] elec_field has wrong sizes");

  ComputeElectricFieldFunctor<dim, device_t> functor(primitive_vars,
                                                     elec_field,
                                                     fm,
                                                     num_mirrors, // first index to compute
                                                     num_ghosts   // number of quads to compute
  );

  const auto nbCellsPerGhostedLeaf = elec_field.num_cells();
  const auto nbCellsTotal = num_ghosts * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ComputeElectricFieldFunctor - ghost quadrants only",
                       Kokkos::RangePolicy<exec_space, TagComputeGhostQuad>(0, nbCellsTotal),
                       functor);

} // apply_on_ghosts

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeElectricFieldFunctor<dim, device_t>::compute_elec_field(index_t const & cell_index,
                                                               index_t const   iOct_in,
                                                               index_t const   iOct_out) const
{
  KOKKOS_ASSERT(iOct_in < m_q.num_quadrants() && "Invalid access to m_q view.");

  // compute cartesian coordinates inside ghosted block
  const auto coord =
    cellindex_to_coord<dim>(cell_index, m_elec_field.ghosted_block_size(), m_elec_field.shift());

  if constexpr (dim == 2)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];

    auto const & iq = i;
    auto const & jq = j;

    // compute velocity at edge

    if (iq > -2 and jq > -2)
    {
      // clang-format off
      const real_t u =
        ONE_FOURTH_F * (m_q(iq - 1, jq - 1, MHD::IU, iOct_in) +
                        m_q(iq - 1, jq    , MHD::IU, iOct_in) +
                        m_q(iq    , jq - 1, MHD::IU, iOct_in) +
                        m_q(iq    , jq    , MHD::IU, iOct_in));
      const real_t v =
        ONE_FOURTH_F * (m_q(iq - 1, jq - 1, MHD::IV, iOct_in) +
                        m_q(iq - 1, jq    , MHD::IV, iOct_in) +
                        m_q(iq    , jq - 1, MHD::IV, iOct_in) +
                        m_q(iq    , jq    , MHD::IV, iOct_in));
      // clang-format on

      // compute magnetic field at edge

      // clang-format off
      const real_t A =
        HALF_F * (m_q(iq, jq - 1, MHD::IAL, iOct_in) +
                  m_q(iq, jq    , MHD::IAL, iOct_in));
      const real_t B =
        HALF_F * (m_q(iq - 1, jq, MHD::IBL, iOct_in) +
                  m_q(iq    , jq, MHD::IBL, iOct_in));
      // clang-format on

      // edge-centered electric field
      m_elec_field(i, j, 0, iOct_out) = u * B - v * A;
    }
  }
  else if constexpr (dim == 3)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];
    auto const & k = coord[IZ];

    auto const & iq = i;
    auto const & jq = j;
    auto const & kq = k;

    // compute Ex
    if (iq > -2 and jq > -2 and kq > -2)
    {
      // clang-format off
      const real_t v = ONE_FOURTH_F * (m_q(iq, jq - 1, kq - 1, MHD::IV, iOct_in) +
                                       m_q(iq, jq - 1, kq    , MHD::IV, iOct_in) +
                                       m_q(iq, jq    , kq - 1, MHD::IV, iOct_in) +
                                       m_q(iq, jq    , kq    , MHD::IV, iOct_in));

      const real_t w = ONE_FOURTH_F * (m_q(iq, jq - 1, kq - 1, MHD::IW, iOct_in) +
                                       m_q(iq, jq - 1, kq    , MHD::IW, iOct_in) +
                                       m_q(iq, jq    , kq - 1, MHD::IW, iOct_in) +
                                       m_q(iq, jq    , kq    , MHD::IW, iOct_in));

      const real_t B =
        HALF_F * (m_q(iq, jq    , kq - 1, MHD::IBL, iOct_in) +
                  m_q(iq, jq    , kq    , MHD::IBL, iOct_in));
      const real_t C =
        HALF_F * (m_q(iq, jq - 1, kq    , MHD::ICL, iOct_in) +
                  m_q(iq, jq    , kq    , MHD::ICL, iOct_in));
      // clang-format on

      m_elec_field(i, j, k, IX, iOct_out) = v * C - w * B;
    }

    // compute Ey
    if (iq > -2 and jq > -2 and kq > -2)
    {
      // clang-format off
      const real_t u = ONE_FOURTH_F * (m_q(iq - 1, jq, kq - 1, MHD::IU, iOct_in) +
                                       m_q(iq - 1, jq, kq    , MHD::IU, iOct_in) +
                                       m_q(iq    , jq, kq - 1, MHD::IU, iOct_in) +
                                       m_q(iq    , jq, kq    , MHD::IU, iOct_in));

      const real_t w = ONE_FOURTH_F * (m_q(iq - 1, jq, kq - 1, MHD::IW, iOct_in) +
                                       m_q(iq - 1, jq, kq    , MHD::IW, iOct_in) +
                                       m_q(iq    , jq, kq - 1, MHD::IW, iOct_in) +
                                       m_q(iq    , jq, kq    , MHD::IW, iOct_in));

      const real_t A =
        HALF_F * (m_q(iq    , jq, kq - 1, MHD::IAL, iOct_in) +
                  m_q(iq    , jq, kq    , MHD::IAL, iOct_in));
      const real_t C =
        HALF_F * (m_q(iq - 1, jq, kq    , MHD::ICL, iOct_in) +
                  m_q(iq    , jq, kq    , MHD::ICL, iOct_in));
      // clang-format on

      m_elec_field(i, j, k, IY, iOct_out) = w * A - u * C;
    }

    // compute Ez
    if (iq > -2 and jq > -2 and kq > -2)
    {
      // clang-format off
      const real_t u = ONE_FOURTH_F * (m_q(iq - 1, jq - 1, kq, MHD::IU, iOct_in) +
                                       m_q(iq - 1, jq    , kq, MHD::IU, iOct_in) +
                                       m_q(iq    , jq - 1, kq, MHD::IU, iOct_in) +
                                       m_q(iq    , jq    , kq, MHD::IU, iOct_in));

      const real_t v = ONE_FOURTH_F * (m_q(iq - 1, jq - 1, kq, MHD::IV, iOct_in) +
                                       m_q(iq - 1, jq    , kq, MHD::IV, iOct_in) +
                                       m_q(iq    , jq - 1, kq, MHD::IV, iOct_in) +
                                       m_q(iq    , jq    , kq, MHD::IV, iOct_in));

      const real_t A =
        HALF_F * (m_q(iq    , jq - 1, kq, MHD::IAL, iOct_in) +
                  m_q(iq    , jq    , kq, MHD::IAL, iOct_in));
      const real_t B =
        HALF_F * (m_q(iq - 1, jq    , kq, MHD::IBL, iOct_in) +
                  m_q(iq    , jq    , kq, MHD::IBL, iOct_in));
      // clang-format on

      m_elec_field(i, j, k, IZ, iOct_out) = u * B - v * A;
    }
  } // end dim == 3

} // compute_elec_field

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeElectricFieldFunctor<dim, device_t>::operator()(TagComputeAllQuadInGroup const &,
                                                       const index_t & global_index) const
{

  // retrieve local octant index (local to group)
  const auto iOct_local = global_index / m_elec_field.num_cells();
  const auto cell_index = global_index - iOct_local * m_elec_field.num_cells();

  // index in (to access m_q) is iOct_local
  // index out (to access m_elec_field, ...) is iOct_local
  compute_elec_field(cell_index, iOct_local, iOct_local);

} // operator() - TagComputeAllQuadInGroup

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeElectricFieldFunctor<dim, device_t>::operator()(TagComputeGhostQuad const &,
                                                       const index_t & global_index) const
{

  // just creating an alias for code clarity
  // iOct_begin is actually the index of the first ghost
  auto const & first_ghost = m_iOct_begin;

  // retrieve ghost index - iGhost should take values between 0 and m_num_octants-1 and
  // in this context, m_num_octants must be equal to the number of ghosts quadrants
  int32_t    iGhost = global_index / m_elec_field.num_cells();
  const auto cell_index = global_index - iGhost * m_elec_field.num_cells();

  // index in (to access m_q) is first_ghost + iGhost
  // index out (to access m_elec_field) is iGhost
  compute_elec_field(cell_index, first_ghost + iGhost, iGhost);

} // operator() - TagComputeGhostQuad

// explicit template instantiation
template class ComputeElectricFieldFunctor<2, kalypsso::DefaultDevice>;
template class ComputeElectricFieldFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
