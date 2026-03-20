// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeLimitedSlopesFunctor.cpp for MHD using cell-centered primitive variables.
 */
#include <godunov_mhd_ct/scheme/ComputeLimitedSlopesFunctor.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
ComputeLimitedSlopesFunctor<dim, device_t>::ComputeLimitedSlopesFunctor(
  DataArrayGhostedBlock_t const & prim_var,
  DataArrayGhostedBlock_t const & slopes_x,
  DataArrayGhostedBlock_t const & slopes_y,
  DataArrayGhostedBlock_t const & slopes_z,
  FieldMap<core::models::MHD>     fm,
  int32_t                         iOct_begin,
  int32_t                         num_octants,
  MHDSettings const &             mhd_settings)
  : m_q(prim_var)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_fm(fm)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
  , m_mhd_settings(mhd_settings)
{}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeLimitedSlopesFunctor<dim, device_t>::apply_on_group(
  DataArrayGhostedBlock_t const & primitive_vars,
  DataArrayGhostedBlock_t const & slopes_x,
  DataArrayGhostedBlock_t const & slopes_y,
  DataArrayGhostedBlock_t const & slopes_z,
  FieldMap<core::models::MHD>     fm,
  int32_t                         num_quads,
  MHDSettings const &             mhd_settings)
{

  ComputeLimitedSlopesFunctor<dim, device_t> functor(primitive_vars,
                                                     slopes_x,
                                                     slopes_y,
                                                     slopes_z,
                                                     fm,
                                                     0,         // first index to compute
                                                     num_quads, // number of quads to process
                                                     mhd_settings);

  const auto nbCellsPerGhostedLeaf = slopes_x.num_cells();
  const auto nbCellsTotal = num_quads * nbCellsPerGhostedLeaf;

  Kokkos::parallel_for("ComputeLimitedSlopesFunctor - All quadrants",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuadInGroup>(0, nbCellsTotal),
                       functor);

} // apply_on_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeLimitedSlopesFunctor<dim, device_t>::apply_on_ghosts(
  DataArrayGhostedBlock_t const & primitive_vars_mg,
  DataArrayGhostedBlock_t const & slopes_x,
  DataArrayGhostedBlock_t const & slopes_y,
  DataArrayGhostedBlock_t const & slopes_z,
  FieldMap<core::models::MHD>     fm,
  int32_t                         num_mirrors,
  int32_t                         num_ghosts,
  MHDSettings const &             mhd_settings)
{

  // we expect primitive_vars to be of size num_mirrors + num_ghosts
  KOKKOS_ASSERT(primitive_vars_mg.num_quadrants() == num_mirrors + num_ghosts &&
                "[ComputeLimitedSlopesFunctor] primitive_vars has wrong sizes");

  // we expect slopes_x, slopes_y and slopes_z to be of size num_ghosts
  KOKKOS_ASSERT(slopes_x.num_quadrants() == num_ghosts &&
                "[ComputeLimitedSlopesFunctor] slopes_x has wrong sizes");
  KOKKOS_ASSERT(slopes_y.num_quadrants() == num_ghosts &&
                "[ComputeLimitedSlopesFunctor] slopes_y has wrong sizes");
  if constexpr (dim == 3)
  {
    KOKKOS_ASSERT(slopes_z.num_quadrants() == num_ghosts &&
                  "[ComputeLimitedSlopesFunctor] slopes_z has wrong sizes");
  }

  ComputeLimitedSlopesFunctor<dim, device_t> functor(primitive_vars_mg,
                                                     slopes_x,
                                                     slopes_y,
                                                     slopes_z,
                                                     fm,
                                                     num_mirrors, // first index to compute in q_mg
                                                     num_ghosts,  // number of quads to compute
                                                     mhd_settings);

  const auto nbCellsPerGhostedLeaf = slopes_x.num_cells();
  const auto nbCellsTotal = num_ghosts * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ComputeLimitedSlopesFunctor - ghost quadrants only",
                       Kokkos::RangePolicy<exec_space, TagComputeGhostQuad>(0, nbCellsTotal),
                       functor);

} // apply_on_ghosts

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
ComputeLimitedSlopesFunctor<dim, device_t>::slope_unsplit_scalar(real_t q,
                                                                 real_t qPlus,
                                                                 real_t qMinus) const
{
  const real_t slope_type = m_mhd_settings.hydro.slope_type;

  real_t dq = 0;

  if (slope_type == 1 or slope_type == 2)
  {
    const real_t dlft = slope_type * (q - qMinus);
    const real_t drgt = slope_type * (qPlus - q);
    const real_t dcen = HALF_F * (qPlus - qMinus);
    const real_t dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    const real_t slop = fmin(fabs(dlft), fabs(drgt));
    const real_t dlim = (dlft * drgt) <= ZERO_F ? ZERO_F : slop;
    dq = dsgn * fmin(dlim, fabs(dcen));
  }

  return dq;

} // slope_unsplit_scalar

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
ComputeLimitedSlopesFunctor<dim, device_t>::slope_unsplit_scalar_mag(
  int32_t const   i,
  int32_t const   j,
  int32_t const   k,
  int             dir,
  const MHD::Id_t ivar1,
  const MHD::Id_t ivar2,
  index_t const & iOct_local) const
{
  // compute unit vector
  const auto uv = [](int direction) {
    Kokkos::Array<int, dim> v;
    for (int idir = 0; idir < static_cast<int>(dim); ++idir)
    {
      v[idir] = idir == direction ? 1 : 0;
    }
    return v;
  }(dir);

  if constexpr (dim == 2)
  {
    // special treatment for face-normal component
    if ((ivar1 == MHD::IAL and ivar2 == MHD::IAR and dir == IX) or
        (ivar1 == MHD::IBL and ivar2 == MHD::IBR and dir == IY))
    {
      return m_q(i, j, ivar2, iOct_local) - m_q(i, j, ivar1, iOct_local);
    }
    else
    {
      // clang-format off
        const auto datac =
          (m_q(i, j, ivar1, iOct_local) +
           m_q(i, j, ivar2, iOct_local)) * HALF_F;
        const auto datap =
          (m_q(i + uv[IX], j + uv[IY], ivar1, iOct_local) +
           m_q(i + uv[IX], j + uv[IY], ivar2, iOct_local)) * HALF_F;
        const auto datam =
          (m_q(i - uv[IX], j - uv[IY], ivar1, iOct_local) +
           m_q(i - uv[IX], j - uv[IY], ivar2, iOct_local)) * HALF_F;
      // clang-format on
      return slope_unsplit_scalar(datac, datap, datam);
    }
  }
  else if constexpr (dim == 3)
  {
    // special treatment for face-normal component
    if ((ivar1 == MHD::IAL and ivar2 == MHD::IAR and dir == IX) or
        (ivar1 == MHD::IBL and ivar2 == MHD::IBR and dir == IY) or
        (ivar1 == MHD::ICL and ivar2 == MHD::ICR and dir == IZ))
    {
      return m_q(i, j, k, ivar2, iOct_local) - m_q(i, j, k, ivar1, iOct_local);
    }
    else
    {
      // clang-format off
        const auto datac =
          (m_q(i, j, k, ivar1, iOct_local) +
           m_q(i, j, k, ivar2, iOct_local)) * HALF_F;
        const auto datap =
          (m_q(i + uv[IX], j + uv[IY], k + uv[IZ], ivar1, iOct_local) +
           m_q(i + uv[IX], j + uv[IY], k + uv[IZ], ivar2, iOct_local)) * HALF_F;
        const auto datam =
          (m_q(i - uv[IX], j - uv[IY], k - uv[IZ], ivar1, iOct_local) +
           m_q(i - uv[IX], j - uv[IY], k - uv[IZ], ivar2, iOct_local)) * HALF_F;
      // clang-format on
      return slope_unsplit_scalar(datac, datap, datam);
    }
  } // end dim == 3

  // we should never be here (dim is always 2 or 3)
  return 0.0;
} // slope_unsplit_scalar_mag

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeLimitedSlopesFunctor<dim, device_t>::compute_limited_slopes(index_t const & cell_index,
                                                                   index_t const   iOct_in,
                                                                   index_t const   iOct_out) const
{
  // compute cartesian coordinates inside ghosted block
  const auto coord =
    cellindex_to_coord<dim>(cell_index, m_slopes_x.ghosted_block_size(), m_slopes_x.shift());

  if constexpr (dim == 2)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];

    for (int ivar = 0; ivar < nbvar_hydro_only<dim>(); ++ivar)
    {
      // clang-format off
      m_slopes_x(i, j, ivar, iOct_out) = slope_unsplit_scalar(m_q(i + 0, j, ivar, iOct_in),
                                                              m_q(i + 1, j, ivar, iOct_in),
                                                              m_q(i - 1, j, ivar, iOct_in));

      m_slopes_y(i, j, ivar, iOct_out) = slope_unsplit_scalar(m_q(i, j + 0, ivar, iOct_in),
                                                              m_q(i, j + 1, ivar, iOct_in),
                                                              m_q(i, j - 1, ivar, iOct_in));
      // clang-format on
    }

    // cell-centered magnetic slopes
    m_slopes_x(i, j, MHD::IA, iOct_out) =
      slope_unsplit_scalar_mag(i, j, 0, IX, MHD::IAL, MHD::IAR, iOct_in);
    m_slopes_y(i, j, MHD::IA, iOct_out) =
      slope_unsplit_scalar_mag(i, j, 0, IY, MHD::IAL, MHD::IAR, iOct_in);

    m_slopes_x(i, j, MHD::IB, iOct_out) =
      slope_unsplit_scalar_mag(i, j, 0, IX, MHD::IBL, MHD::IBR, iOct_in);
    m_slopes_y(i, j, MHD::IB, iOct_out) =
      slope_unsplit_scalar_mag(i, j, 0, IY, MHD::IBL, MHD::IBR, iOct_in);

    m_slopes_x(i, j, MHD::IC, iOct_out) =
      slope_unsplit_scalar_mag(i, j, 0, IX, MHD::ICL, MHD::ICR, iOct_in);
    m_slopes_y(i, j, MHD::IC, iOct_out) =
      slope_unsplit_scalar_mag(i, j, 0, IY, MHD::ICL, MHD::ICR, iOct_in);
  }
  else if constexpr (dim == 3)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];
    auto const & k = coord[IZ];

    for (int ivar = 0; ivar < nbvar_hydro_only<dim>(); ++ivar)
    {
      m_slopes_x(i, j, k, ivar, iOct_out) = slope_unsplit_scalar(m_q(i + 0, j, k, ivar, iOct_in),
                                                                 m_q(i + 1, j, k, ivar, iOct_in),
                                                                 m_q(i - 1, j, k, ivar, iOct_in));

      m_slopes_y(i, j, k, ivar, iOct_out) = slope_unsplit_scalar(m_q(i, j + 0, k, ivar, iOct_in),
                                                                 m_q(i, j + 1, k, ivar, iOct_in),
                                                                 m_q(i, j - 1, k, ivar, iOct_in));

      m_slopes_z(i, j, k, ivar, iOct_out) = slope_unsplit_scalar(m_q(i, j, k + 0, ivar, iOct_in),
                                                                 m_q(i, j, k + 1, ivar, iOct_in),
                                                                 m_q(i, j, k - 1, ivar, iOct_in));
    }

    // cell-centered magnetic slopes
    m_slopes_x(i, j, k, MHD::IA, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IX, MHD::IAL, MHD::IAR, iOct_in);
    m_slopes_y(i, j, k, MHD::IA, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IY, MHD::IAL, MHD::IAR, iOct_in);
    m_slopes_z(i, j, k, MHD::IA, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IZ, MHD::IAL, MHD::IAR, iOct_in);

    m_slopes_x(i, j, k, MHD::IB, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IX, MHD::IBL, MHD::IBR, iOct_in);
    m_slopes_y(i, j, k, MHD::IB, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IY, MHD::IBL, MHD::IBR, iOct_in);
    m_slopes_z(i, j, k, MHD::IB, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IZ, MHD::IBL, MHD::IBR, iOct_in);

    m_slopes_x(i, j, k, MHD::IC, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IX, MHD::ICL, MHD::ICR, iOct_in);
    m_slopes_y(i, j, k, MHD::IC, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IY, MHD::ICL, MHD::ICR, iOct_in);
    m_slopes_z(i, j, k, MHD::IC, iOct_out) =
      slope_unsplit_scalar_mag(i, j, k, IZ, MHD::ICL, MHD::ICR, iOct_in);

  } // end dim == 3

} // compute_limited_slopes

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeLimitedSlopesFunctor<dim, device_t>::operator()(TagComputeAllQuadInGroup const &,
                                                       const index_t & global_index) const
{

  // retrieve local octant index (local to group)
  const auto iOct_local = global_index / m_slopes_x.num_cells();
  const auto cell_index = global_index - iOct_local * m_slopes_x.num_cells();

  // index in (to access m_q) is iOct_local
  // index out (to access m_slopes_x, ...) is iOct_local
  compute_limited_slopes(cell_index, iOct_local, iOct_local);

} // operator() - TagComputeAllQuadInGroup

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeLimitedSlopesFunctor<dim, device_t>::operator()(TagComputeGhostQuad const &,
                                                       const index_t & global_index) const
{

  // just creating an alias for code clarity
  // iOct_begin is actually the index of the first ghost
  auto const & first_ghost = m_iOct_begin;

  // retrieve ghost index - iGhost should take values between 0 and m_num_octants-1 and
  // in this context, m_num_octants must be equal to the number of ghosts quadrants
  int32_t    iGhost = global_index / m_slopes_x.num_cells();
  const auto cell_index = global_index - iGhost * m_slopes_x.num_cells();

  // index in (to access m_q) is first_ghost + iGhost
  // index out (to access m_slopes_x, ...) is iGhost
  compute_limited_slopes(cell_index, first_ghost + iGhost, iGhost);

} // operator() - TagComputeGhostQuad

// explicit template instantiation
template class ComputeLimitedSlopesFunctor<2, kalypsso::DefaultDevice>;
template class ComputeLimitedSlopesFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
