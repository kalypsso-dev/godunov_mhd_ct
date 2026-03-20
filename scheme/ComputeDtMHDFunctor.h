// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDtMHDFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_COMPUTE_DT_MHD_FUNCTOR_H_
#define KALYPSSO_GODUNOV_MHD_COMPUTE_DT_MHD_FUNCTOR_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/orchard_key_base.h>

// hydro utils (conservative versus primitive variable, equation of state, ...)
#include <kalypsso/core/models/MHDState.h>
// #include <kalypsso/core/models/mhd_utils.h>
#include <kalypsso/core/models/mhd_utils.h>
#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/GravityField.h>
#include <kalypsso/core/ViscosityParams.h>

namespace kalypsso
{
namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Simplest CFL computational functor for compressible mono fluid magneto-hydrodynamics.
 *
 * All cell, whatever level, contribute equally to the CFL condition.
 *
 * We actually compute inverse of cfl, the user is responsible to convert it to actual CFL.
 *
 * \tparam dim is space dimension (2 or 3)
 * \tparam device_t is the Kokkos device use for computation (CPU, GPU, ...)
 */
template <size_t dim, typename device_t>
class ComputeDtMHDFunctor
{

public:
  //! type alias for cell-centered data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! type alias for face-centered data array at block level (see kalypsso_data_container.h)
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  //! type alias for a (device) Kokkos view of orchard keys
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  //! global cell index
  using index_t = int32_t;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! EOS parameters
  MHDSettings m_mhd_settings;

  //! Viscosity parameters
  ViscosityParams m_viscosity_params;

  //! field manager
  FieldMap<core::models::MHD> m_fm;

  //! block sizes
  block_size_t<dim> m_block_sizes;

  //! number of cells per leaf
  const int32_t m_nbCellsPerLeaf;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  //! cell-centered conservative variables
  DataArrayBlock_t m_Udata;

  //! face-centered magnetic field components
  FaceDataArrayBlock_t m_Bface;

  //! gravity source term enabled ?
  const bool m_gravity_enabled;

  //! uniform gravity field
  const UniformGravityField<dim> m_gravity_field;

public:
  ComputeDtMHDFunctor(ConfigMap const &                config_map,
                      orchard_key_view_t const &       orchard_keys,
                      int32_t                          local_num_octants,
                      MHDSettings const &              mhd_settings,
                      FieldMap<core::models::MHD>      fm,
                      block_size_t<dim> const &        block_sizes,
                      DataArrayBlock_t const &         Udata,
                      FaceDataArrayBlock_t const &     Bface,
                      bool                             gravity_enabled,
                      UniformGravityField<dim> const & gravity_field);

  // ====================================================================
  // ====================================================================
  //! static method which does it all: create and execute functor using range policy
  //!
  //! \param[in] orchard_keys is a vector of all local (owned+ghost) octant orchard/morton keys
  //! \param[in] local_num_octants is the number of octants owned by current MPI process (ghost
  //!            excluded)
  //! \param[in] mhd_settings contains hydrodynamics parameter used to perform conservative to
  //! primitive
  //!            variable conversion (equation of state)
  //! \param[in] fm is the field map (TODO refactor this)
  //! \param[in] block_sizes is an array the cartesian block sizes
  //! \param[in,out] invDt is the inverse of time step, the output of this functor
  //!
  static void
  apply(ConfigMap const &            config_map,
        orchard_key_view_t const &   orchard_keys,
        int32_t                      local_num_octants,
        MHDSettings const &          mhd_settings,
        FieldMap<core::models::MHD>  fm,
        block_size_t<dim> const &    block_sizes,
        DataArrayBlock_t const &     Udata,
        FaceDataArrayBlock_t const & Bface,
        real_t &                     invDt);

  // ====================================================================
  // ====================================================================
  /**
   * Update reduced variable when visiting a cell.
   *
   * \param[in] iOct is the visited octant id
   * \param[in] cell_index is the visited local cell index (local to block)
   * \param[in,out] invDt is the reduced variable to update
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  compute_cfl(int32_t const & iOct, int32_t const & cell_index, real_t & invDt) const;

  KOKKOS_INLINE_FUNCTION void
  compute_cfl_with_gravity(int32_t const & iOct, int32_t const & cell_index, real_t & invDt) const;

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor for computing CFL condition.
   *
   * \param[in] global_index spans range from 0 to nbCellsPerLeaf * local_num_octants-1
   *            (i.e. total number of cells in current MPI process)
   * \param[in,out] invDt is the reduced variable to update
   */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index, real_t & invDt) const;

}; // class ComputeDtMHDFunctor

// explicit template instantiation
extern template class ComputeDtMHDFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeDtMHDFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_COMPUTE_DT_MHD_FUNCTOR_H_
