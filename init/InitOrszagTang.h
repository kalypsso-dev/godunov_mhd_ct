// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitOrszagTang.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_INIT_ORSZAG_TANG_H_
#define KALYPSSO_GODUNOV_MHD_CT_INIT_ORSZAG_TANG_H_

#include <godunov_mhd_ct/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/OrszagTangParams.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement user data initialization to solve Orszag-Tang vortex problem.
 *
 * This test asserts/requires periodic boundary conditions.
 *
 * \sa http://www.astro.virginia.edu/VITA/ATHENA/ot.html
 * \sa http://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
 *
 * This functor takes as input a mesh, already refined (after companion functor
 * InitOrszagTangRefineFunctor), and initializes user data on host.
 * Copying data from host to device, should be done outside.
 *
 * Initial conditions is refined near strong density gradients.
 *
 * \note this functor uses Tags for discriminating different loops (cell-center loops, or
 * face-center loops).
 *
 * \note we explicitly only allow square blocks, i.e. block such that bx=by=bz
 *
 * \sa InitOrszagTangRefineFunctor
 */
template <size_t dim, typename device_t>
class InitOrszagTangDataFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

private:
  //! hydrodynamics variables
  DataArrayBlock_t m_Udata;

  //! magnetic field components
  FaceDataArrayBlock_t m_Bface;

  //! field manager
  FieldMap<models::MHD> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  MHDSettings m_mhd_settings;

  //! OrszagTang problem specific parameters (used on device)
  OrszagTangParams m_otParams;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain upper right corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

public:
  //! init all hydro var, except total energy (partially initialized)
  struct TagInitHydroVar
  {};

  //! init magnetic field
  struct TagInitMagField
  {};

  //! update total energy with magnetic energy
  struct TagInitTotalEnergy
  {};


  InitOrszagTangDataFunctor(DataArrayBlock_t             Udata,
                            FaceDataArrayBlock_t         Bface,
                            FieldMap<models::MHD>        fm,
                            orchard_key_view_t<device_t> orchard_keys,
                            int32_t                      local_num_octants,
                            ConfigMap const &            config_map);

  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t             Udata,
        FaceDataArrayBlock_t         Bface,
        FieldMap<models::MHD>        fm,
        orchard_key_view_t<device_t> orchard_keys,
        int32_t                      local_num_octants,
        ConfigMap const &            config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitHydroVar, const int32_t & global_index) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitMagField, const int32_t & global_index) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitTotalEnergy, const int32_t & global_index) const;

}; // InitOrszagTangDataFunctor

// explicit template instantiation
extern template class InitOrszagTangDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitOrszagTangDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * Orszag-Tang vortex initialization.
 */
template <size_t dim, typename device_t>
class InitOrszagTang
{
public:
  static void
  apply(SolverGodunovMHD<dim, device_t> & solver);
};

// explicit template instantiation declaration to prevent implicit instantiation
extern template class InitOrszagTang<2, kalypsso::DefaultDevice>;
extern template class InitOrszagTang<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_INIT_ORSZAG_TANG_H_
