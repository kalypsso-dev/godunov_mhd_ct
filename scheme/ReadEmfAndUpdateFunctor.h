// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ReadEmfAndUpdateFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_READ_EMF_AND_UPDATE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_MHD_READ_EMF_AND_UPDATE_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/models/MHDState.h>
#include <kalypsso/core/AMRMeshInfo.h>

// utils mhd
#include <kalypsso/core/models/mhd_utils.h>

#include <type_traits>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Read emf (electromotive forces) and perform a CONSERVATIVE update magnetic field components.
 *
 * \sa ReadFluxesAndConservativeUpdateFunctor
 */
template <size_t dim, typename device_t>
class ReadEmfAndUpdateFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // hashmap related type aliases
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

private:
  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! user data - magnetic field - entire mesh - in/out
  FaceDataArrayBlock_t m_Bout;

  //! fluxes - owned and ghost quadrants
  DataArrayBlock_t m_emf;

  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

public:
  /**
   * Perform time integration (induction equation) of magnetic field.
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ReadEmfAndUpdateFunctor(AMRMeshInfo const &          amr_mesh_info,
                          FaceDataArrayBlock_t const & B_out,
                          DataArrayBlock_t const &     emf);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply(AMRMeshInfo const &          amr_mesh_info,
        FaceDataArrayBlock_t const & Bout,
        DataArrayBlock_t const &     emf);

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  read_emf_and_update_2d(index_t const & face_index, iOct_t const & iOct) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  read_emf_and_update_3d(index_t const & face_index, iOct_t const & iOct) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ReadEmfAndUpdateFunctor

// explicit template instantiation
extern template class ReadEmfAndUpdateFunctor<2, kalypsso::DefaultDevice>;
extern template class ReadEmfAndUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_READ_EMF_AND_UPDATE_FUNCTOR_H_
