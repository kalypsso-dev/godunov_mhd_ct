// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeHydroFluxesAndStoreFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_COMPUTE_HYDRO_FLUXES_AND_STORE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_MHD_COMPUTE_HYDRO_FLUXES_AND_STORE_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/models/MHDState.h>
#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/TimeIntegratorConfig.h>

// mhd
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
 * Compute fluxes (on conservative variables) and store. Actual update of conservative
 * variables (i.e. perform time integration using Godunov (e.g. MUSCL-Hancock) scheme) will be
 * done in a separate functor. This functor is designed to be used when performing a non-piecewise
 * godunov update (all block at once).
 *
 * Input data is Ugroup (containing ghosted block data)
 *
 * We compute fluxes (using Riemann solver) and store.
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
class ComputeHydroFluxesAndStoreFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int64_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // access quadrant <-> orchard key (hence AMR level and quadrant size)
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! fluxes (output)
  DataArrayBlock_t m_Fluxes;

  //! a ghosted block array of primitive variables (ghost width is 2)
  //! size :
  //! if implem version 0 : owned + ghost quadrants
  //! if implem version 1 : size of group of quadrants
  DataArrayGhostedBlock_t m_q;

  //! a ghosted block array of primitive variables at t_{n+1/2} (ghost width is 1)
  DataArrayGhostedBlock_t m_q2;

  //! ghosted block data arrays (ghost width is 1) - slopes along X
  DataArrayGhostedBlock_t m_slopes_x;

  //! ghosted block data arrays (ghost width is 1) - slopes along Y
  DataArrayGhostedBlock_t m_slopes_y;

  //! ghosted block data arrays (ghost width is 1) - slopes along Z - only used when dim=3
  DataArrayGhostedBlock_t m_slopes_z;

  //! a ghosted block array of  source term for mag field (ghost width is 1) - 1 component in 2d, 3
  //! components in 3D
  DataArrayGhostedBlock_t m_sFaceMag;

  //! field manager
  FieldMap<core::models::MHD> m_fm;

  //! offset to first octant in flux array where to write
  const int32_t m_iOct_flux_offset;

  //! number of quadrants to process
  const int32_t m_num_quads;

  //! flux direction (IX, IY or IZ)
  int m_direction;

  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

  //! number of cells per leaf block
  const int32_t m_nbCellsPerLeaf;

  //! hydro settings (EOS parameters)
  MHDSettings m_mhd_settings;

  //! time step
  real_t m_dt;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! time integrator
  const TimeIntegrator m_time_integrator;

public:
  /**
   * Compute Godunov fluxes along a given direction.
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ComputeHydroFluxesAndStoreFunctor(orchard_key_view_t const &      orchard_keys,
                                    AMRMeshInfo const &             amr_mesh_info,
                                    DataArrayBlock_t const &        fluxes,
                                    DataArrayGhostedBlock_t const & q_ghosted,
                                    DataArrayGhostedBlock_t const & q2_ghosted,
                                    DataArrayGhostedBlock_t const & slopes_x,
                                    DataArrayGhostedBlock_t const & slopes_y,
                                    DataArrayGhostedBlock_t const & slopes_z,
                                    DataArrayGhostedBlock_t const & sFaceMag,
                                    FieldMap<core::models::MHD>     fm,
                                    int32_t                         iOct_flux_offset,
                                    int32_t                         num_quads,
                                    int                             direction,
                                    MHDSettings const &             mhd_settings,
                                    real_t                          dt,
                                    real_t                          scaling_factor,
                                    TimeIntegrator const &          time_integrator);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  static void
  apply(ConfigMap const &               config_map,
        orchard_key_view_t const &      orchard_keys,
        AMRMeshInfo const &             amr_mesh_info,
        DataArrayBlock_t const &        fluxes,
        DataArrayGhostedBlock_t const & q_ghosted,
        DataArrayGhostedBlock_t const & q2_ghosted,
        DataArrayGhostedBlock_t const & slopes_x,
        DataArrayGhostedBlock_t const & slopes_y,
        DataArrayGhostedBlock_t const & slopes_z,
        DataArrayGhostedBlock_t const & sFaceMag,
        FieldMap<core::models::MHD>     fm,
        int32_t                         iOct_flux_offset,
        int32_t                         num_quads,
        int                             direction,
        MHDSettings const &             mhd_settings,
        real_t                          dt);

  // ====================================================================
  // ====================================================================
  /**
   * Select a slopes array.
   */
  KOKKOS_INLINE_FUNCTION auto
  get_slopes(int dir) const
  {
    if constexpr (dim == 2)
    {
      KOKKOS_ASSERT((dir == IX or dir == IY) && "Wrong direction value");
    }
    else if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((dir == IX or dir == IY or dir == IZ) && "Wrong direction value");
    }

    if (dir == IX)
    {
      return m_slopes_x;
    }
    else if (dir == IY)
    {
      return m_slopes_y;
    }
    else if (dir == IZ)
    {
      return m_slopes_z;
    }

    // default value (should never be here in debug mode)
    return m_slopes_x;

  } // get_slopes

  // ====================================================================
  // ====================================================================
  /**
   * Get state vector.
   *
   * \param[in] data a ghosted block data array (q2 or slopes)
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_state(DataArrayGhostedBlock_t const & data, int32_t i, int32_t j, int32_t iOct_local) const
  {

    MHDStateCell q;

    q[MHD::ID] = data(i, j, m_fm[MHD::ID], iOct_local);
    q[MHD::IP] = data(i, j, m_fm[MHD::IP], iOct_local);
    q[MHD::IU] = data(i, j, m_fm[MHD::IU], iOct_local);
    q[MHD::IV] = data(i, j, m_fm[MHD::IV], iOct_local);
    q[MHD::IW] = data(i, j, m_fm[MHD::IW], iOct_local);
    q[MHD::IA] = data(i, j, m_fm[MHD::IA], iOct_local);
    q[MHD::IB] = data(i, j, m_fm[MHD::IB], iOct_local);
    q[MHD::IC] = data(i, j, m_fm[MHD::IC], iOct_local);

    return q;

  } // get_state

  // ====================================================================
  // ====================================================================
  /**
   * Get state vector.
   *
   * \param[in] data a ghosted block data array
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] k identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_state(DataArrayGhostedBlock_t const & data,
            int32_t                         i,
            int32_t                         j,
            int32_t                         k,
            int32_t                         iOct_local) const
  {

    MHDStateCell q;

    q[MHD::ID] = data(i, j, k, m_fm[MHD::ID], iOct_local);
    q[MHD::IP] = data(i, j, k, m_fm[MHD::IP], iOct_local);
    q[MHD::IU] = data(i, j, k, m_fm[MHD::IU], iOct_local);
    q[MHD::IV] = data(i, j, k, m_fm[MHD::IV], iOct_local);
    q[MHD::IW] = data(i, j, k, m_fm[MHD::IW], iOct_local);
    q[MHD::IA] = data(i, j, k, m_fm[MHD::IA], iOct_local);
    q[MHD::IB] = data(i, j, k, m_fm[MHD::IB], iOct_local);
    q[MHD::IC] = data(i, j, k, m_fm[MHD::IC], iOct_local);

    return q;

  } // get_state

  // ====================================================================
  // ====================================================================
  /**
   * Set flux (hydro variables only).
   *
   * \param[in] i identifies location in the flux in block
   * \param[in] j identifies location in the flux in block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  set_flux(int32_t i, int32_t j, int32_t iOct, MHDStateCell const & flux) const
  {
    iOct += m_iOct_flux_offset;

    m_Fluxes(i, j, m_fm[MHD::ID], iOct) = flux[MHD::ID];
    m_Fluxes(i, j, m_fm[MHD::IP], iOct) = flux[MHD::IP];
    m_Fluxes(i, j, m_fm[MHD::IU], iOct) = flux[MHD::IU];
    m_Fluxes(i, j, m_fm[MHD::IV], iOct) = flux[MHD::IV];
    m_Fluxes(i, j, m_fm[MHD::IW], iOct) = flux[MHD::IW];
    m_Fluxes(i, j, m_fm[MHD::IC], iOct) = flux[MHD::IC];

  } // set_flux - 2d

  // ====================================================================
  // ====================================================================
  /**
   * Set flux (hydro variables only).
   *
   * \param[in] i identifies location in the flux in block
   * \param[in] j identifies location in the flux in block
   * \param[in] k identifies location in the flux in block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  set_flux(int32_t i, int32_t j, int32_t k, int32_t iOct, MHDStateCell const & flux) const
  {
    iOct += m_iOct_flux_offset;

    m_Fluxes(i, j, k, m_fm[MHD::ID], iOct) = flux[MHD::ID];
    m_Fluxes(i, j, k, m_fm[MHD::IP], iOct) = flux[MHD::IP];
    m_Fluxes(i, j, k, m_fm[MHD::IU], iOct) = flux[MHD::IU];
    m_Fluxes(i, j, k, m_fm[MHD::IV], iOct) = flux[MHD::IV];
    m_Fluxes(i, j, k, m_fm[MHD::IW], iOct) = flux[MHD::IW];
    m_Fluxes(i, j, k, m_fm[MHD::IA], iOct) = flux[MHD::IA];
    m_Fluxes(i, j, k, m_fm[MHD::IB], iOct) = flux[MHD::IB];
    m_Fluxes(i, j, k, m_fm[MHD::IC], iOct) = flux[MHD::IC];

  } // set_flux - 3d

  // ====================================================================
  // ====================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets.
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \note offsets are given in units dx/2, i.e. a vector containing only 1.0 or -1.0
   *
   * \param[in] q primitive variables at cell center, at time t_{n+1/2}
   * \param[in] i_s X coordinate to access slope array
   * \param[in] j_s Y coordinate to access slope array
   * \param[in] iOct_local index to octant in local array
   * \param[in] dir (IX or IY)
   * \param[in] face (FACE_LEFT or FACE_RIGHT)
   *
   * \return qr reconstructed state (primitive variables)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_2d(MHDStateCell const & q,
                       int32_t              i_s,
                       int32_t              j_s,
                       int32_t              iOct_local,
                       int                  dir,
                       face_type_t          face) const;

  // ====================================================================
  // ====================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets (3d version).
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \note offsets are given in units dx/2, i.e. a vector containing only 1.0 or -1.0
   *
   * \param[in] q primitive variables at cell center, at time t_{n+1/2}
   * \param[in] is X coordinate to access slope array
   * \param[in] js Y coordinate to access slope array
   * \param[in] ks Y coordinate to access slope array
   * \param[in] iOct_local index to octant in local array
   * \param[in] dir (IX or IY or IZ)
   * \param[in] face (FACE_LEFT or FACE_RIGHT)
   *
   * \return qr reconstructed state (primitive variables)
   *
   * \sa reconstruct_state_2d
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_3d(MHDStateCell const & q,
                       int32_t              is,
                       int32_t              js,
                       int32_t              ks,
                       int32_t              iOct_local,
                       int                  dir,
                       face_type_t          face) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_store_2d(int32_t const & cell_index, int32_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_store_3d(const int32_t & cell_index, const int32_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ComputeHydroFluxesAndStoreFunctor

// explicit template instantiation
extern template class ComputeHydroFluxesAndStoreFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeHydroFluxesAndStoreFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_COMPUTE_HYDRO_FLUXES_AND_STORE_FUNCTOR_H_
