// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ReadFluxesAndConservativeUpdateFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_READ_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_MHD_READ_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/FieldMap.h>
#include <godunov_mhd_ct/models/MHDState.h>
#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>

// utils hydro
#include <godunov_mhd_ct/models/mhd_utils.h>

#include <type_traits>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Read fluxes (on conservative variables) and perform a CONSERVATIVE update of conservative
 * variables (i.e. perform time integration using Godunov (e.g. MUSCL-Hancock) scheme).
 *
 * Input data is Ugroup (containing ghosted block data)
 *
 * We compute fluxes (using Riemann solver) and perform
 * update directly in external array U.
 * Loop through all cell (sub-)faces.
 *
 * \note This functor actually assumes the slopes array to be ghosted array with ghost width of 1.
 * Conservative variable array is assume to be a block array (no ghost).
 *
 * \todo routines like reconstruct_state_2d/3d could probably be
 * moved outside to alleviate this class.
 *
 */
template <size_t dim, typename device_t>
class ReadFluxesAndConservativeUpdateFunctor
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
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using MHD = models::MHD;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

  using CellLocation_t = CellLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

private:
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! conformal status view
  conformal_status_view_type m_conformal_status;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! user data - hydrodynamics variables - entire mesh - in/out
  DataArrayBlock_t m_Uout;

  //! user data - magnetic field - entire mesh - in/out
  FaceDataArrayBlock_t m_Bout;

  //! fluxes - owned and ghost quadrants
  DataArrayBlock_t m_Fluxes;

  //! field manager
  FieldMap<models::MHD> m_fm;

  //! flux direction (IX, IY or IZ)
  int m_direction;

  //! number of owned quadrants
  const int32_t m_num_owned;

  //! number of ghost quadrants
  const int32_t m_num_ghosts;

  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

  //! hydro settings (EOS parameters)
  MHDSettings m_mhd_settings;

  //! time step
  real_t m_dt;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

public:
  /**
   * Perform time integration (MUSCL Godunov).
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ReadFluxesAndConservativeUpdateFunctor(ConfigMap const &                  config_map,
                                         StencilHelper_t const &            stencil_helper,
                                         orchard_key_view_t const &         orchard_keys,
                                         conformal_status_view_type const & conformal_status,
                                         AMRMeshInfo const &                amr_mesh_info,
                                         DataArrayBlock_t const &           u_out,
                                         FaceDataArrayBlock_t const &       B_out,
                                         DataArrayBlock_t const &           fluxes,
                                         FieldMap<models::MHD>        fm,
                                         int                                direction,
                                         MHDSettings const &                mhd_settings,
                                         real_t                             dt);

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
        DataArrayBlock_t const &           Uout,
        FaceDataArrayBlock_t const &       Bout,
        DataArrayBlock_t const &           fluxes,
        FieldMap<models::MHD>        fm,
        int                                direction,
        brick_size_t<dim> const &          brick_sizes,
        Kokkos::Array<bool, dim> const &   is_brick_periodic,
        MHDSettings const &                mhd_settings,
        real_t                             dt);

  // ====================================================================
  // ====================================================================
  /**
   * Get conservative variables state vector.
   *
   * \param[in] i,j identifies location in flux array
   * \param[in] iOct identifies a quadrant (owned + ghosts)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  read_flux(int32_t i, int32_t j, int32_t iOct) const
  {

    MHDStateCell flux;

    flux[MHD::ID] = m_Fluxes(i, j, m_fm[MHD::ID], iOct);
    flux[MHD::IP] = m_Fluxes(i, j, m_fm[MHD::IP], iOct);
    flux[MHD::IU] = m_Fluxes(i, j, m_fm[MHD::IU], iOct);
    flux[MHD::IV] = m_Fluxes(i, j, m_fm[MHD::IV], iOct);
    flux[MHD::IW] = m_Fluxes(i, j, m_fm[MHD::IW], iOct);
    flux[MHD::IC] = m_Fluxes(i, j, m_fm[MHD::IC], iOct);

    return flux;

  } // read_flux

  // ====================================================================
  // ====================================================================
  /**
   * Get conservative variables state vector.
   *
   * \param[in] i,j,k identifies in flux array
   * \param[in] iOct identifies a quadrant (owned + ghosts)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  read_flux(int32_t i, int32_t j, int32_t k, int32_t iOct) const
  {

    MHDStateCell flux;

    flux[MHD::ID] = m_Fluxes(i, j, k, m_fm[MHD::ID], iOct);
    flux[MHD::IP] = m_Fluxes(i, j, k, m_fm[MHD::IP], iOct);
    flux[MHD::IU] = m_Fluxes(i, j, k, m_fm[MHD::IU], iOct);
    flux[MHD::IV] = m_Fluxes(i, j, k, m_fm[MHD::IV], iOct);
    flux[MHD::IW] = m_Fluxes(i, j, k, m_fm[MHD::IW], iOct);

    return flux;

  } // read_flux

  // ====================================================================
  // ====================================================================
  /**
   * Get flux from fine neighbor.
   *
   * \param[in] iOct_cur current octant id
   * \param[in] coords current cell cartesian coordinate (inside current block)
   * \param[in] shift indicates local direction to fine neighbor
   * \param[in] use_right_flux boolean value to tell if we want to use left or right flux
   *
   */
  KOKKOS_INLINE_FUNCTION auto
  get_flux_from_fine_neighbor(index_t const &      iOct_cur,
                              coord_t<dim> const & coords,
                              shift_t<dim> const & shift,
                              bool                 use_right_flux) const
  {
    const auto           key_cur = m_orchard_keys_device(iOct_cur);
    const CellLocation_t cell_loc{ coords, key_cur, iOct_cur, false };
    const auto           cell_loc_neigh = m_stencil_helper.getNeighLocFinerNearer(cell_loc, shift);

    MHDStateCell flux;

    flux[MHD::ID] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, m_fm[MHD::ID], m_Fluxes, use_right_flux);
    flux[MHD::IP] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, m_fm[MHD::IP], m_Fluxes, use_right_flux);
    flux[MHD::IU] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, m_fm[MHD::IU], m_Fluxes, use_right_flux);
    flux[MHD::IV] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, m_fm[MHD::IV], m_Fluxes, use_right_flux);
    flux[MHD::IW] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, m_fm[MHD::IW], m_Fluxes, use_right_flux);

    if constexpr (dim == 2)
    {
      flux[MHD::IC] = m_stencil_helper.compute_face_siblings_sum(
        cell_loc_neigh, m_fm[MHD::IC], m_Fluxes, use_right_flux);
    }

    return flux;

  } // get_flux_from_fine_neighbor_flux

  // ====================================================================
  // ====================================================================
  /**
   * Update state vector (hydro variables only).
   *
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  update_U(int32_t i, int32_t j, int32_t iOct, MHDStateCell const & flux) const
  {

    m_Uout(i, j, m_fm[MHD::ID], iOct) += flux[MHD::ID];
    m_Uout(i, j, m_fm[MHD::IP], iOct) += flux[MHD::IP];
    m_Uout(i, j, m_fm[MHD::IU], iOct) += flux[MHD::IU];
    m_Uout(i, j, m_fm[MHD::IV], iOct) += flux[MHD::IV];
    m_Uout(i, j, m_fm[MHD::IW], iOct) += flux[MHD::IW];
    m_Bout(i, j, IZ, iOct) += flux[MHD::IC];

  } // update_U - 2d

  // ====================================================================
  // ====================================================================
  /**
   * Update state vector (hydro variables only).
   *
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] k identifies location in the ghosted block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  update_U(int32_t i, int32_t j, int32_t k, int32_t iOct, MHDStateCell const & flux) const
  {

    m_Uout(i, j, k, m_fm[MHD::ID], iOct) += flux[MHD::ID];
    m_Uout(i, j, k, m_fm[MHD::IP], iOct) += flux[MHD::IP];
    m_Uout(i, j, k, m_fm[MHD::IU], iOct) += flux[MHD::IU];
    m_Uout(i, j, k, m_fm[MHD::IV], iOct) += flux[MHD::IV];
    m_Uout(i, j, k, m_fm[MHD::IW], iOct) += flux[MHD::IW];

  } // update_U - 3d

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  read_fluxes_and_update_2d(index_t const & cell_index, index_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  read_fluxes_and_update_3d(const index_t & cell_index, const index_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ReadFluxesAndConservativeUpdateFunctor

// explicit template instantiation
extern template class ReadFluxesAndConservativeUpdateFunctor<2, kalypsso::DefaultDevice>;
extern template class ReadFluxesAndConservativeUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_READ_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_
