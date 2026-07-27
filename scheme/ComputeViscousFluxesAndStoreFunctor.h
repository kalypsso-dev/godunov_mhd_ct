// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeViscousFluxesAndStoreFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_COMPUTE_VISCOUS_FLUXES_AND_STORE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_MHD_COMPUTE_VISCOUS_FLUXES_AND_STORE_FUNCTOR_H_

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
#include <kalypsso/core/ViscosityParams.h>

// utils mhd
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
 * Compute hydrodynamics fluxes (on conservative variables) due to viscous forces and store.
 *
 * As for Godunov hydrodynamics fluxes, the actual update of conservative
 * variables will be done in a separate functor (taking into account non-conformal mesh interface).
 *
 * This functor is designed to be used when performing a non-piecewise godunov update (all block at
 * once).
 *
 * Input data is q (containing primitive variables in a ghosted block data array)
 *
 */
template <size_t dim, typename device_t>
class ComputeViscousFluxesAndStoreFunctor
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int64_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using MHD = models::MHD;

  // makes enum Hydro::GradId available
  using Grad = models::MHD::GradTensorId;

  // access quadrant <-> orchard key (hence AMR level and quadrant size)
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

  using Velocity_t = Kokkos::Array<real_t, dim>;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! viscous fluxes (output)
  DataArrayBlock_t m_Fluxes;

  //! a ghosted block array of primitive variables (ghost width is 2)
  //! size :
  //! if implem version 0 : owned + ghost quadrants
  //! if implem version 1 : size of group of quadrants
  DataArrayGhostedBlock_t m_q;

  //! field manager
  FieldMap<models::MHD> m_fm;

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

  //! viscosity parameters
  ViscosityParams m_viscosity;

  //! time step
  real_t m_dt;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

public:
  /**
   * Compute viscous fluxes along a given direction.
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ComputeViscousFluxesAndStoreFunctor(orchard_key_view_t const &      orchard_keys,
                                      AMRMeshInfo const &             amr_mesh_info,
                                      DataArrayBlock_t const &        fluxes,
                                      DataArrayGhostedBlock_t const & q_ghosted,
                                      FieldMap<models::MHD>           fm,
                                      int32_t                         iOct_flux_offset,
                                      int32_t                         num_quads,
                                      int                             direction,
                                      ViscosityParams const &         viscosity,
                                      real_t                          dt,
                                      real_t                          scaling_factor);

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
        FieldMap<models::MHD>           fm,
        int32_t                         iOct_flux_offset,
        int32_t                         num_quads,
        int                             direction,
        ViscosityParams const &         viscosity,
        real_t                          dt);

  // ====================================================================
  // ====================================================================
  /**
   * Get velocities.
   *
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_velocity(int32_t i, int32_t j, int32_t iOct_local) const
  {

    Velocity_t q;

    q[IX] = m_q(i, j, m_fm[MHD::IU], iOct_local);
    q[IY] = m_q(i, j, m_fm[MHD::IV], iOct_local);

    return q;

  } // get_velocity

  // ====================================================================
  // ====================================================================
  /**
   * Get velocities.
   *
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] k identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_velocity(int32_t i, int32_t j, int32_t k, int32_t iOct_local) const
  {

    Velocity_t q;

    q[IX] = m_q(i, j, k, m_fm[MHD::IU], iOct_local);
    q[IY] = m_q(i, j, k, m_fm[MHD::IV], iOct_local);
    q[IZ] = m_q(i, j, k, m_fm[MHD::IW], iOct_local);

    return q;

  } // get_velocity

  // ====================================================================
  // ====================================================================
  /**
   * Set viscous flux.
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

    m_Fluxes(i, j, m_fm[MHD::ID], iOct) = ZERO_F;
    m_Fluxes(i, j, m_fm[MHD::IE], iOct) = flux[MHD::IE];
    m_Fluxes(i, j, m_fm[MHD::IU], iOct) = flux[MHD::IU];
    m_Fluxes(i, j, m_fm[MHD::IV], iOct) = flux[MHD::IV];
    m_Fluxes(i, j, m_fm[MHD::IW], iOct) = ZERO_F;
    m_Fluxes(i, j, m_fm[MHD::IC], iOct) = ZERO_F;

  } // set_flux - 2d

  // ====================================================================
  // ====================================================================
  /**
   * Set viscous flux.
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

    m_Fluxes(i, j, k, m_fm[MHD::ID], iOct) = ZERO_F;
    m_Fluxes(i, j, k, m_fm[MHD::IE], iOct) = flux[MHD::IE];
    m_Fluxes(i, j, k, m_fm[MHD::IU], iOct) = flux[MHD::IU];
    m_Fluxes(i, j, k, m_fm[MHD::IV], iOct) = flux[MHD::IV];
    m_Fluxes(i, j, k, m_fm[MHD::IW], iOct) = flux[MHD::IW];
    m_Fluxes(i, j, k, m_fm[MHD::IA], iOct) = ZERO_F;
    m_Fluxes(i, j, k, m_fm[MHD::IB], iOct) = ZERO_F;
    m_Fluxes(i, j, k, m_fm[MHD::IC], iOct) = ZERO_F;

  } // set_flux - 3d

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  compute_velocity_gradient_2d(int32_t i, int32_t j, int32_t iOct_local, real_t dx) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  compute_velocity_gradient_3d(int32_t i, int32_t j, int32_t k, int32_t iOct_local, real_t dx)
    const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_viscous_fluxes_and_store_2d(int32_t const & cell_index, int32_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_viscous_fluxes_and_store_3d(const int32_t & cell_index, const int32_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ComputeViscousFluxesAndStoreFunctor

// explicit template instantiation
extern template class ComputeViscousFluxesAndStoreFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeViscousFluxesAndStoreFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_COMPUTE_VISCOUS_FLUXES_AND_STORE_FUNCTOR_H_
