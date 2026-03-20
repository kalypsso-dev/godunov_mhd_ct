// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ReadEmfAndUpdateFunctor.cpp
 */
#include <godunov_mhd_ct/scheme/ReadEmfAndUpdateFunctor.h>
#include <kalypsso/core/utils_block.h> // for face_to_cell_coords

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ReadEmfAndUpdateFunctor<dim, device_t>::ReadEmfAndUpdateFunctor(AMRMeshInfo const & amr_mesh_info,
                                                                FaceDataArrayBlock_t const & b_out,
                                                                DataArrayBlock_t const &     emf)
  : m_amr_mesh_info(amr_mesh_info)
  , m_Bout(b_out)
  , m_emf(emf)
  , m_block_sizes(b_out.cell_block_size())
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ReadEmfAndUpdateFunctor<dim, device_t>::apply(AMRMeshInfo const &          amr_mesh_info,
                                              FaceDataArrayBlock_t const & Bout,
                                              DataArrayBlock_t const &     emf)
{
  // Important note: the caller is responsible for providing a flux array with right shape.
  {
    [[maybe_unused]] auto emf_block_sizes = Bout.cell_block_size() + 1;
    assertm(emf_block_sizes == emf.block_size(), "EMF array has incompatible shape.");
  }

  ReadEmfAndUpdateFunctor<dim, device_t> functor(amr_mesh_info, Bout, emf);

  // number of owned quadrant x number of faces (Bx,By,Bz)
  const auto nbIterations = amr_mesh_info.local_num_quadrants() * Bout.num_elements_per_octant();

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::ReadEmfAndUpdateFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ReadEmfAndUpdateFunctor<dim, device_t>::read_emf_and_update_2d(index_t const & face_index,
                                                               iOct_t const &  iOct) const
{
  auto const face_indexes =
    face_flat_index_unravel<2>(face_index, m_block_sizes, m_Bout.offsets(), m_Bout.shift());
  auto const & i = face_indexes[IX];
  auto const & j = face_indexes[IY];
  auto const & ivar = face_indexes[dim];

  /*
   * Update Bx
   */
  if (ivar == IX)
  {
    // clang-format off
    auto emf0 = m_emf(i, j    , ALONG_Z, iOct);
    auto emf1 = m_emf(i, j + 1, ALONG_Z, iOct);
    // clang-format on

    m_Bout(i, j, IX, iOct) += (emf1 - emf0);

  } // end ivar == IX

  /*
   * Update By
   */
  else if (ivar == IY)
  {
    // clang-format off
    auto emf1 = m_emf(i + 1, j, ALONG_Z, iOct);
    auto emf0 = m_emf(i    , j, ALONG_Z, iOct);
    // clang-format on

    m_Bout(i, j, IY, iOct) -= (emf1 - emf0);

  } // end ivar == IY

} // read_emf_and_update_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ReadEmfAndUpdateFunctor<dim, device_t>::read_emf_and_update_3d(index_t const & face_index,
                                                               iOct_t const &  iOct) const
{
  auto const face_indexes =
    face_flat_index_unravel<3>(face_index, m_block_sizes, m_Bout.offsets(), m_Bout.shift());
  auto const & i = face_indexes[IX];
  auto const & j = face_indexes[IY];
  auto const & k = face_indexes[IZ];
  auto const & ivar = face_indexes[dim];

  /*
   * Update Bx
   */
  if (ivar == IX)
  {
    real_t dBx = 0;

    // clang-format off
    dBx += (m_emf(i, j + 1, k    , IZ, iOct) - m_emf(i, j, k, IZ, iOct));
    dBx -= (m_emf(i, j    , k + 1, IY, iOct) - m_emf(i, j, k, IY, iOct));
    // clang-format on

    m_Bout(i, j, k, IX, iOct) += dBx;

  } // end ivar == IX

  /*
   * Update By
   */
  if (ivar == IY)
  {
    real_t dBy = 0;

    // clang-format off
    dBy += (m_emf(i    , j, k + 1, IX, iOct) - m_emf(i, j, k, IX, iOct));
    dBy -= (m_emf(i + 1, j, k    , IZ, iOct) - m_emf(i, j, k, IZ, iOct));
    // clang-format on

    m_Bout(i, j, k, IY, iOct) += dBy;

  } // end ivar == IY

  /*
   * Update Bz
   */
  if (ivar == IZ)
  {
    real_t dBz = 0;

    // clang-format off
    dBz += (m_emf(i + 1, j    , k, IY, iOct) - m_emf(i, j, k, IY, iOct));
    dBz -= (m_emf(i    , j + 1, k, IX, iOct) - m_emf(i, j, k, IX, iOct));
    // clang-format on

    m_Bout(i, j, k, IZ, iOct) += dBz;

  } // end ivar == IZ

} // read_emf_and_update_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ReadEmfAndUpdateFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto num_faces = m_Bout.num_elements_per_octant();

  // retrieve local octant index
  auto const iOct_local = global_index / num_faces;
  auto const face_index = static_cast<int32_t>(global_index - iOct_local * num_faces);

  if constexpr (dim == 2)
  {
    read_emf_and_update_2d(face_index, iOct_local);
  }
  else if constexpr (dim == 3)
  {
    read_emf_and_update_3d(face_index, iOct_local);
  }

} // operator ()

// explicit template instantiation
template class ReadEmfAndUpdateFunctor<2, kalypsso::DefaultDevice>;
template class ReadEmfAndUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
