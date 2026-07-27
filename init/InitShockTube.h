// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitShockTube.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_INIT_SHOCK_TUBE_H_
#define KALYPSSO_GODUNOV_MHD_CT_INIT_SHOCK_TUBE_H_

#include <godunov_mhd_ct/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/MHDShockTubeParams.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement user data initialization to solve a shock tube problem.
 *
 *
 * This functor takes as input a mesh, already refined (after companion functor
 * InitShockTubeRefineFunctor), and initializes user data on host.
 * Copying data from host to device, should be done outside.
 *
 * Initial conditions is refined near strong density gradients.
 *
 * \sa InitShockTubeRefineFunctor
 */
template <size_t dim, typename device_t>
class InitShockTubeDataFunctor
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

  //! MHD settings (used on device)
  MHDSettings m_mhd_settings;

  //! ShockTube problem specific parameters (used on device)
  MHDShockTubeParams m_st_params;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

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

  InitShockTubeDataFunctor(DataArrayBlock_t             Udata,
                           FaceDataArrayBlock_t         Bface,
                           FieldMap<models::MHD>  fm,
                           orchard_key_view_t<device_t> orchard_keys,
                           int32_t                      local_num_octants,
                           ConfigMap const &            config_map);

  // ====================================================================
  // ====================================================================
  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t             Udata,
        FaceDataArrayBlock_t         Bface,
        FieldMap<models::MHD>  fm,
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

}; // class InitShockTubeDataFunctor

// explicit template instantiation
extern template class InitShockTubeDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitShockTubeDataFunctor<3, kalypsso::DefaultDevice>;

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initial refinement to solve ShockTube shock tube problem.
 *
 * This functor only performs mesh refinement, no user data init.
 * User data init is actually done in InitShockTubeDataFunctor
 *
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitShockTubeDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitShockTubeRefineFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a (device) Kokkos view of refinement flags
  using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

  struct TagRefineAlways
  {};
  struct TagRefineGeometric
  {};

private:
  //! hydrodynamics variables
  DataArrayBlock_t m_Udata;

  //! magnetic field components
  FaceDataArrayBlock_t m_Bface;

  //! field manager
  FieldMap<models::MHD> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! MHD settings (used on device)
  MHDSettings m_mhd_settings;

  //! ShockTube problem specific parameters (used on device)
  MHDShockTubeParams m_st_params;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  // ===========================================================
  // ===========================================================
  InitShockTubeRefineFunctor(DataArrayBlock_t             Udata,
                             FaceDataArrayBlock_t         Bface,
                             FieldMap<models::MHD>  fm,
                             orchard_key_view_t<device_t> orchard_keys,
                             amrflags_view_t              amrflags,
                             int32_t                      local_num_octants,
                             int                          level_refine,
                             ConfigMap const &            config_map);

  // ===========================================================
  // ===========================================================
  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t             Udata,
        FaceDataArrayBlock_t         Bface,
        FieldMap<models::MHD>  fm,
        orchard_key_view_t<device_t> orchard_keys,
        amrflags_view_t              amrflags,
        int32_t                      local_num_octants,
        int                          level_refine,
        ConfigMap const &            config_map);

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const iOct_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const iOct_t & iOct) const;

}; // class InitShockTubeRefineFunctor

// explicit template instantiation
extern template class InitShockTubeRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitShockTubeRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * MHD shock tube problem.
 *
 *
 * Initial condition is mostly done on host, the final refined initial
 * condition data are uploaded to kokkos device.
 *
 * Different initial refinement strategies are possible:
 * - no refinement at all
 * - geometric refinement, i.e. refine near interface
 * - regular gradient based refinement
 *
 * This is controlled input parameter in ini file : "amr"/"init_condition_refine_criterion"
 */
template <size_t dim, typename device_t>
class InitShockTube
{
public:
  static void
  apply(SolverGodunovMHD<dim, device_t> & solver);
}; // class InitShockTube

// explicit template instantiation declaration to prevent implicit instantiation
extern template class InitShockTube<2, kalypsso::DefaultDevice>;
extern template class InitShockTube<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_INIT_SHOCK_TUBE_H_
