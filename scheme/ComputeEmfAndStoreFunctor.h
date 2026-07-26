// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeEmfAndStoreFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_COMPUTE_EMF_AND_STORE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_MHD_COMPUTE_EMF_AND_STORE_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/models/MHDState.h>
#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/TimeIntegratorConfig.h>

#include <godunov_mhd_ct/models/RiemannSolvers_MHD.h>

// utils hydro
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
 * Compute electromotive forces (EMF) and perform a CONSERVATIVE (constraint transport)
 * update of (face-centered) magnetic field components using Maxwell-Faraday equation.
 *
 * \note Hydrodynamics (cell-center) variables are updated in ComputeHydroFluxesAndUpdateFunctor.
 *
 * \note This functor actually assumes the slopes array to be ghosted array with ghost width of 1.
 * Conservative variable array is assume to be a block array (no ghost).
 *
 * \todo routines like reconstruct_state_2d/3d could probably be
 * moved outside to alleviate this class.
 *
 */
template <size_t dim, typename device_t>
class ComputeEmfAndStoreFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  // access quadrant <-> orchard key (hence AMR level and quadrant size)
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! EMF (ElectroMotive forces)
  DataArrayBlock_t m_emf;

  //! a ghosted block array of primitive variables at t_n (ghost width is 2)
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

  //! offset to first octant where to write
  const int32_t m_iOct_emf_offset;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_quads;

  // //! block sizes (no ghost)
  // const block_size_t<dim> m_block_sizes;

  // //! emf block sizes
  // const block_size_t<dim> m_block_sizes_emf;

  // //! number of cells per leaf block
  // const int32_t m_nbCellsPerLeaf;

  // //! number of edge emfs per leaf block (just 1 more than the number of cells in all direction)
  // const int32_t m_nbEdgeEmfPerLeaf;

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
   * Perform time integration (MUSCL Godunov).
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ComputeEmfAndStoreFunctor(ConfigMap const &               config_map,
                            orchard_key_view_t const &      orchard_keys,
                            AMRMeshInfo const &             amr_mesh_info,
                            DataArrayBlock_t const &        emf,
                            DataArrayGhostedBlock_t const & q,
                            DataArrayGhostedBlock_t const & q2,
                            DataArrayGhostedBlock_t const & slopes_x,
                            DataArrayGhostedBlock_t const & slopes_y,
                            DataArrayGhostedBlock_t const & slopes_z,
                            DataArrayGhostedBlock_t const & sFaceMag,
                            FieldMap<core::models::MHD>     fm,
                            int32_t                         iOct_emf_offset,
                            int32_t                         num_quads,
                            MHDSettings const &             mhd_settings,
                            real_t                          dt,
                            TimeIntegrator const &          time_integrator);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply(ConfigMap const &               config_map,
        orchard_key_view_t const &      orchard_keys,
        AMRMeshInfo const &             amr_mesh_info,
        DataArrayBlock_t const &        emf,
        DataArrayGhostedBlock_t const & q,
        DataArrayGhostedBlock_t const & q2,
        DataArrayGhostedBlock_t const & slopes_x,
        DataArrayGhostedBlock_t const & slopes_y,
        DataArrayGhostedBlock_t const & slopes_z,
        DataArrayGhostedBlock_t const & sFaceMag,
        FieldMap<core::models::MHD>     fm,
        int32_t                         iOct_emf_offset,
        int32_t                         num_quads,
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
   * Get state vector but with a different interface.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  get_state(DataArrayGhostedBlock_t const & data,
            int32_t                         i,
            int32_t                         j,
            int32_t                         iOct_local,
            MHDStateCell &                  q) const
  {

    q[MHD::ID] = data(i, j, m_fm[MHD::ID], iOct_local);
    q[MHD::IP] = data(i, j, m_fm[MHD::IP], iOct_local);
    q[MHD::IU] = data(i, j, m_fm[MHD::IU], iOct_local);
    q[MHD::IV] = data(i, j, m_fm[MHD::IV], iOct_local);
    q[MHD::IW] = data(i, j, m_fm[MHD::IW], iOct_local);
    q[MHD::IA] = data(i, j, m_fm[MHD::IA], iOct_local);
    q[MHD::IB] = data(i, j, m_fm[MHD::IB], iOct_local);
    q[MHD::IC] = data(i, j, m_fm[MHD::IC], iOct_local);

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
   * Get state vector but with a different interface.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  get_state(DataArrayGhostedBlock_t const & data,
            int32_t                         i,
            int32_t                         j,
            int32_t                         k,
            int32_t                         iOct_local,
            MHDStateCell &                  q) const
  {

    q[MHD::ID] = data(i, j, k, m_fm[MHD::ID], iOct_local);
    q[MHD::IP] = data(i, j, k, m_fm[MHD::IP], iOct_local);
    q[MHD::IU] = data(i, j, k, m_fm[MHD::IU], iOct_local);
    q[MHD::IV] = data(i, j, k, m_fm[MHD::IV], iOct_local);
    q[MHD::IW] = data(i, j, k, m_fm[MHD::IW], iOct_local);
    q[MHD::IA] = data(i, j, k, m_fm[MHD::IA], iOct_local);
    q[MHD::IB] = data(i, j, k, m_fm[MHD::IB], iOct_local);
    q[MHD::IC] = data(i, j, k, m_fm[MHD::IC], iOct_local);

  } // get_state

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION real_t
  slope_unsplit_scalar(real_t q, real_t qPlus, real_t qMinus) const
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

  // ====================================================================
  // ====================================================================
  /**
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  compute_limited_slope(int32_t iq, int32_t jq, int32_t ivar, int32_t iOct_local, int dir) const
  {
    if (dir == IX)
    {
      return slope_unsplit_scalar(m_q(iq + 0, jq, ivar, iOct_local),
                                  m_q(iq + 1, jq, ivar, iOct_local),
                                  m_q(iq - 1, jq, ivar, iOct_local));
    }
    else if (dir == IY)
    {
      return slope_unsplit_scalar(m_q(iq, jq + 0, ivar, iOct_local),
                                  m_q(iq, jq + 1, ivar, iOct_local),
                                  m_q(iq, jq - 1, ivar, iOct_local));
    }

    return ZERO_F;

  } // compute_limited_slope - 2d

  // ====================================================================
  // ====================================================================
  /**
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  compute_limited_slope(int32_t iq,
                        int32_t jq,
                        int32_t kq,
                        int32_t ivar,
                        int32_t iOct_local,
                        int     dir) const
  {
    if (dir == IX)
    {
      return slope_unsplit_scalar(m_q(iq + 0, jq, kq, ivar, iOct_local),
                                  m_q(iq + 1, jq, kq, ivar, iOct_local),
                                  m_q(iq - 1, jq, kq, ivar, iOct_local));
    }
    else if (dir == IY)
    {
      return slope_unsplit_scalar(m_q(iq, jq + 0, kq, ivar, iOct_local),
                                  m_q(iq, jq + 1, kq, ivar, iOct_local),
                                  m_q(iq, jq - 1, kq, ivar, iOct_local));
    }
    else if (dir == IZ)
    {
      return slope_unsplit_scalar(m_q(iq, jq, kq + 0, ivar, iOct_local),
                                  m_q(iq, jq, kq + 1, ivar, iOct_local),
                                  m_q(iq, jq, kq - 1, ivar, iOct_local));
    }

    return ZERO_F;

  } // compute_limited_slope - 3d

  // ====================================================================
  // ====================================================================
  /**
   * Reconstruct an hydro state at a cell-face border.
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \param[in] is X coordinate to access slope array
   * \param[in] js Y coordinate to access slope array
   * \param[in] iOct_local index to octant in local array
   * \param[in] edge_loc specifies the type of edge reconstruction (from cell center)
   * \param[in,out] q primitive variables at cell center at time t_{n+1/2}
   *
   * \return qr reconstructed state (primitive variables)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_2d_at_edge(int32_t         is,
                               int32_t         js,
                               int32_t         iOct_local,
                               MHDEdgeLocation edge_loc,
                               MHDStateCell &  q) const;

  // ====================================================================
  // ====================================================================
  /**
   * Reconstruct an hydro state at a cell-face border.
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \param[in] is X coordinate to access slope array
   * \param[in] js Y coordinate to access slope array
   * \param[in] ks Z coordinate to access slope array
   * \param[in] iOct_local index to octant in local array
   * \param[in] edge_loc specifies the type of edge reconstruction (from cell center)
   * \param[in,out] q primitive variables at cell center at time t_{n+1/2}
   *
   * \return qr reconstructed state (primitive variables)
   *
   * \sa reconstruct_state_2d
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_3d_at_edge(int32_t         is,
                               int32_t         js,
                               int32_t         ks,
                               int32_t         iOct_local,
                               MHDEdgeLocation edge_loc,
                               int             dir0,
                               int             dir1,
                               MHDStateCell &  q) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_emf_and_store_2d(index_t const & edge_index, index_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_emf_and_store_3d(const index_t & edge_index, const index_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ComputeEmfAndStoreFunctor

// explicit template instantiation
extern template class ComputeEmfAndStoreFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeEmfAndStoreFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_COMPUTE_EMF_AND_STORE_FUNCTOR_H_
